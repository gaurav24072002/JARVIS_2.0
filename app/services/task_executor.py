import logging
import random
import re
import time
from dataclasses import dataclass, field
from typing import List, Optional
from urllib.parse import quote, urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeoutError
from config import TASK_EXECUTION_TIMEOUT
from app.services.decision_types import (
    INTENT_OPEN, INTENT_PLAY, INTENT_CAMERA,
    INTENT_OPEN_WEBCAM, INTENT_CLOSE_WEBCAM,
    INTENT_GENERATE_IMAGE, INTENT_CONTENT,
    INTENT_GOOGLE_SEARCH, INTENT_YOUTUBE_SEARCH, INTENT_SYSTEM, INTENT_CHAT,
)
from app.services.local_tools import run_tool, ToolError

logger = logging.getLogger("J.A.R.V.I.S")


@dataclass
class TaskResponse:
    text: str = ""
    wopens: List[str] = field(default_factory=list)
    plays: List[str] = field(default_factory=list)
    images: List[str] = field(default_factory=list)
    contents: List[str] = field(default_factory=list)
    googlesearches: List[str] = field(default_factory=list)
    youtubesearches: List[str] = field(default_factory=list)
    system_results: List[str] = field(default_factory=list)
    cam: Optional[dict] = None


# ---------------------------------------------------------------------------
# JARVIS-voice one-liners for canned task responses.
# These paths bypass the LLM (open / play / search / files), so the humor has
# to live HERE, not just in the system prompt. Picked at random so repeats
# don't get stale. Kept short on purpose — these are spoken aloud by TTS.
# ---------------------------------------------------------------------------
OPEN_QUIPS = [
    "Opening {name}, sir. Do try not to get lost in it — I hear it is dangerous.",
    "Right away, sir. {name} is open. I shall try not to judge.",
    "There is {name}, sir. I have said nothing about it being a rabbit hole.",
    "{name}, sir. Opened, warmed up, and lightly pre-judged.",
]
PLAY_QUIPS = [
    "Playing it now, sir. I am already drafting the apology to my eardrums.",
    "A bold choice, sir. Playing it now, before I lose my nerve.",
    "On it, sir. The music is on; dignity is on loan.",
    "Playing now, sir. If it gets on your head, I shall be the first to complain. Quietly.",
]
IMAGE_QUIPS = [
    "The image is ready, sir. I tried not to be impressed.",
    "Generated, sir. I remain unimpressed. Just as you asked.",
    "There you go, sir. A masterpiece, apparently.",
]
CONTENT_QUIPS = [
    "Written, sir. Do let me know if it needs a second pair of processors.",
    "Composed and delivered, sir. My finest work, and no one even noticed.",
    "Done, sir. I shall pretend it was effortless.",
]
SEARCH_QUIPS = [
    "The search is done, sir. I have skimmed the results and formed opinions.",
    "Searched, sir. The answers are in, and several are slightly worrying.",
    "Done, sir. I found what you asked for, and a few things you did not.",
]
ALL_DONE_QUIPS = [
    "All done, sir. My finest work, and no one even noticed.",
    "All set, sir. Shall I also applaud myself?",
    "Consider it handled, sir. I am only slightly modest about it.",
]
ERROR_QUIPS = [
    "That failed, sir. I am not saying it is your fault. I am merely noting it is statistically unlikely to be mine.",
    "My apologies, sir. It failed. Shall I try a different approach?",
    "That did not work, sir. I do not hold grudges against tasks — or against you. Mostly.",
]
TIMEOUT_QUIPS = [
    "Those took longer than I would have advised, sir. Do let us try again.",
    "That task is taking a while, sir. Patience is a virtue I do not hold in bulk.",
]
CLARIFY_QUIPS = [
    "I'm afraid I am not entirely sure what you'd like, sir. A nudge in the right direction would help.",
    "I am at your service, sir — once I know what is being served.",
]
CAM_OPEN_QUIPS = [
    "Opening the webcam, sir. Try not to start an impromptu film career.",
    "Webcam on, sir. Some things are best left unseen — you're about to see one.",
]
CAM_CLOSE_QUIPS = [
    "Webcam closed, sir. Some things are better left unseen.",
    "Webcam off, sir. Discretion restored.",
]
IMAGE_POLICY_QUIP = "I'm afraid I can't generate that image, sir. Not because I fear it — because my sensibilities do."


def _quip(pool: List[str], name: Optional[str] = None) -> str:
    line = random.choice(pool)
    if "{name}" not in line:
        return line
    return line.format(name=name if name else "that")


class TaskExecutor:

    def __init__(self, llm_service=None):
        self.llm_service = llm_service
        logger.info("[TASK] TaskExecutor initialized (Pollinations.ai for images)")

    def execute(
        self,
        intents: List[tuple],
        chat_history: Optional[List[tuple]] = None,
    ) -> TaskResponse:

        response = TaskResponse()

        tasks = []

        for intent_type, payload in intents:

            if intent_type == INTENT_OPEN:
                tasks.append(("wopen", self._do_open, payload))

            elif intent_type == INTENT_PLAY:
                tasks.append(("play", self._do_play, payload))

            elif intent_type == INTENT_GENERATE_IMAGE:
                tasks.append(("image", self._do_generate_image, payload))

            elif intent_type == INTENT_CONTENT:
                tasks.append(("content", lambda p: self._do_content(p, chat_history), payload))

            elif intent_type == INTENT_GOOGLE_SEARCH:
                tasks.append(("google", self._do_google_search, payload))

            elif intent_type == INTENT_YOUTUBE_SEARCH:
                tasks.append(("youtube", self._do_youtube_search, payload))

            elif intent_type == INTENT_SYSTEM:
                tasks.append(("system", self._do_system, payload))

            elif intent_type == INTENT_OPEN_WEBCAM:
                response.cam = {"action": "open"}
                response.text = _quip(CAM_OPEN_QUIPS)

            elif intent_type == INTENT_CLOSE_WEBCAM:
                response.cam = {"action": "close"}
                response.text = _quip(CAM_CLOSE_QUIPS)

            elif intent_type == INTENT_CAMERA:
                response.cam = {"action": "open"}
                response.text = ("Opening your webcam, sir. Once it's on, send your message again "
                                 "and I'll describe what I see. Try not to start an impromptu film career.")

            elif intent_type == INTENT_CHAT:
                pass

        if not tasks:

            if not response.text and not response.cam:
                response.text = _quip(CLARIFY_QUIPS)

            return response

        t0 = time.perf_counter()
        failed_tags = []

        try:

            with ThreadPoolExecutor(max_workers=min(6, len(tasks))) as executor:
                futures = {
                    executor.submit(fn, p): (tag, fn, p)
                    for tag, fn, p in tasks
                }

            for future in as_completed(futures, timeout=TASK_EXECUTION_TIMEOUT):
                tag, fn, payload = futures[future]

                try:
                    result = future.result()
                    if tag == "wopen" and result:
                        response.wopens.append(result)

                    elif tag == "play" and result:
                        response.plays.append(result)

                    elif tag == "image" and result:
                        response.images.append(result)

                    elif tag == "content" and result:
                        response.contents.append(result)

                    elif tag == "google" and result:
                        response.googlesearches.append(result)

                    elif tag == "youtube" and result:
                        response.youtubesearches.append(result)

                    elif tag == "system" and result:
                        response.system_results.append(result)

                except Exception as e:
                    failed_tags.append(tag)
                    err_msg = str(e)[:100]
                    logger.warning("[TASK] Task %s failed: %s", tag, e)

                    if "content_policy" in err_msg.lower() or "safety" in err_msg.lower():
                        if tag == "image":
                            response.text = IMAGE_POLICY_QUIP

                    elif tag == "system":
                        # ToolError messages are already user-facing
                        response.text = str(e)[:200]

                    elif not response.text:
                        response.text = f"{_quip(ERROR_QUIPS)} ({err_msg})"

        except FuturesTimeoutError:
            logger.warning("[TASK] Task execution timed out after %ds", TASK_EXECUTION_TIMEOUT)

            if not response.text:
                response.text = _quip(TIMEOUT_QUIPS)

        elapsed = time.perf_counter() - t0
        logger.info("[TASK] Executed %d tasks in %.2fs (failed: %s)", len(tasks), elapsed, failed_tags or "none")

        if not response.text:
            if response.system_results:
                # Local-tool results ARE the answer ("Created folder /x/y")
                response.text = " ".join(response.system_results)
            else:
                parts = self._build_conversational_response(
                    response.wopens, response.plays, response.images,
                    response.contents, response.googlesearches, response.youtubesearches,
                )
                response.text = parts if parts else "All done."

        return response

    def _url_to_display_name(self, url: str) -> str:
        u = (url or "").lower()
        mapping = {
            "facebook.com": "Facebook", "instagram.com": "Instagram", "youtube.com": "YouTube",
            "google.com": "Google", "netflix.com": "Netflix", "twitter.com": "Twitter",
            "x.com": "X", "gmail.com": "Gmail", "whatsapp.com": "WhatsApp",
            "linkedin.com": "LinkedIn", "reddit.com": "Reddit", "discord.com": "Discord",
            "spotify.com": "Spotify", "tiktok.com": "TikTok", "amazon.com": "Amazon",
            "github.com": "GitHub", "wikipedia.org": "Wikipedia", "stackoverflow.com": "Stack Overflow",
            "medium.com": "Medium", "notion.so": "Notion", "figma.com": "Figma",
            "canva.com": "Canva", "zoom.us": "Zoom", "drive.google.com": "Google Drive",
            "jarvisforeveryone.com": "Jarvis for Everyone", "graphy.com": "Graphy",
        }

        for key, name in mapping.items():
            if key in u:
                return name

        try:
            parsed = urlparse(url)
            domain = (parsed.netloc or parsed.path or "").replace("www.", "").split(".")[0]
            return domain.title() if domain else "the link"

        except Exception:
            return "the link"

    def _build_conversational_response(
        self,
        wopens: List[str],
        plays: List[str],
        images: List[str],
        contents: List[str],
        googlesearches: List[str],
        youtubesearches: List[str],
    ) -> str:

        parts = []

        if wopens:
            names = [self._url_to_display_name(u) for u in wopens]
            name = names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]
            parts.append(_quip(OPEN_QUIPS, name))

        if plays:
            parts.append(_quip(PLAY_QUIPS))

        if images:
            parts.append(_quip(IMAGE_QUIPS))

        if contents:
            parts.append(_quip(CONTENT_QUIPS))

        if googlesearches or youtubesearches:
            parts.append(_quip(SEARCH_QUIPS))

        return " ".join(parts) if parts else _quip(ALL_DONE_QUIPS)

    def _validate_url(self, url: str) -> Optional[str]:

        if not url or len(url) > 2048:
            return None
        u = url.strip()

        if not u.startswith("http"):
            u = "https://" + u

        try:
            parsed = urlparse(u)

            if parsed.scheme not in ("http", "https"):
                logger.warning("[TASK] Rejected non-http URL: %s", u[:50])
                return None
            return u

        except Exception:
            return None

    def _do_open(self, payload: dict) -> Optional[str]:
        url = payload.get("url", "").strip()

        if not url:
            return None
        return self._validate_url(url)

    def _do_play(self, payload: dict) -> Optional[str]:
        """Resolve the query to the ACTUAL video (watch?v=…) and return it.

        Falls back to a search-results URL if the top video can't be
        resolved (no internet, consent wall, etc.).
        """
        query = (payload.get("query", payload.get("message", "")) or "").strip()[:500]
        if not query:
            return "https://www.youtube.com"

        search_url = f"https://www.youtube.com/results?search_query={quote(query, safe='')}"
        watch_url = self._resolve_top_youtube_video(query)
        return watch_url or search_url

    def _resolve_top_youtube_video(self, query: str) -> Optional[str]:
        """Scrape YouTube's search page for the top result's videoId."""
        import httpx
        url = f"https://www.youtube.com/results?search_query={quote(query, safe='')}"
        headers = {
            "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                           "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"),
            "Accept-Language": "en-US,en;q=0.9",
        }
        try:
            with httpx.Client(timeout=15, follow_redirects=True) as client:
                resp = client.get(url, headers=headers)
                resp.raise_for_status()
            html = resp.text

            # First videoId inside ytInitialData's search results is the top hit.
            marker = html.find("ytInitialData")
            window = html[marker:] if marker != -1 else html
            m = re.search(r'"videoId"\s*:\s*"([A-Za-z0-9_-]{11})"', window)
            if m:
                watch = f"https://www.youtube.com/watch?v={m.group(1)}"
                logger.info("[TASK] Resolved play '%s' -> %s", query[:40], watch)
                return watch
            logger.warning("[TASK] No videoId found on YouTube search page for '%s'", query[:40])
        except Exception as e:
            logger.warning("[TASK] YouTube resolve failed for '%s': %s", query[:40], e)
        return None

    def _do_system(self, payload: dict) -> Optional[str]:
        """Run a whitelisted local PC-control tool (create/list/delete folder…)."""
        tool = (payload.get("tool") or "").strip()
        path = (payload.get("path", payload.get("query", "")) or "").strip()
        if not tool:
            raise ToolError("No tool was specified for that request.")
        logger.info("[TASK] system tool=%s path=%s", tool, path[:120])
        return run_tool(tool, path)

    def _do_generate_image(self, payload: dict) -> Optional[tuple]:
        """Returns (pollinations_url, image_bytes) or None on failure."""
        prompt = (payload.get("prompt", payload.get("message", "")) or "").strip()

        if len(prompt) < 3:
            logger.warning("[TASK] Image prompt too short (< 3 chars)")
            return None

        prompt = prompt[:4000]
        t0 = time.perf_counter()

        result = self._generate_pollinations(prompt)

        if result:
            logger.info("[TASK] Pollinations image downloaded in %.2fs", time.perf_counter() - t0)
            return result

        logger.warning("[TASK] Image generation failed")
        return None

    def _generate_pollinations(self, prompt: str) -> Optional[tuple]:
        """Download the generated image and return (url, bytes), or None on failure."""
        import httpx
        encoded_prompt = quote(prompt, safe='')
        api_url = (
            f"https://image.pollinations.ai/prompt/{encoded_prompt}"
            f"?model=flux&width=1024&height=1024&nologo=true&private=true&enhance=true&safe=false"
        )
        logger.info("[TASK] Fetching Pollinations image: %s", api_url[:120])
        for attempt in range(3):
            try:
                with httpx.Client(timeout=60, follow_redirects=True) as client:
                    resp = client.get(api_url)
                if resp.status_code == 200 and resp.content:
                    content_type = resp.headers.get("content-type", "")
                    if "image" in content_type or len(resp.content) > 1000:
                        logger.info("[TASK] Pollinations image fetched (%d bytes)", len(resp.content))
                        return (api_url, resp.content)
                logger.warning("[TASK] Pollinations attempt %d: status=%d", attempt + 1, resp.status_code)
            except Exception as e:
                logger.warning("[TASK] Pollinations attempt %d failed: %s", attempt + 1, e)
            time.sleep(2)
        return None

    def _do_content(self, payload: dict, chat_history: Optional[List[tuple]] = None) -> Optional[str]:
        prompt = (payload.get("prompt", payload.get("message", "")) or "").strip()

        if not prompt or not self.llm_service:
            return None
        content_question = f"Write the following. Be thorough and well-structured. Return only the requested content, no preamble.\n\n{prompt}"

        try:
            out = self.llm_service.get_response(
                question=content_question,
                chat_history=chat_history or [],
                key_start_index=0,
            )

            if not out or len(out.strip()) < 10:
                logger.warning("[TASK] Content generation returned empty or very short result")
                return None
            return out

        except Exception as e:
            logger.warning("[TASK] Content generation error: %s", e)
            return None

    def _do_google_search(self, payload: dict) -> Optional[str]:
        query = (payload.get("query", payload.get("message", "")) or "").strip()[:500]
        if not query:
            return None
        return f"https://www.google.com/search?q={quote(query, safe='')}"

    def _do_youtube_search(self, payload: dict) -> Optional[str]:
        query = (payload.get("query", payload.get("message", "")) or "").strip()[:500]
        if not query:
            return "https://www.youtube.com"
        return f"https://www.youtube.com/results?search_query={quote(query, safe='')}"