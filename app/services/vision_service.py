import base64
import logging
from typing import Optional

import httpx

from config import (
    VISION_MAX_IMAGE_BYTES,
    OLLAMA_BASE_URL, OLLAMA_VISION_MODEL, OLLAMA_REQUEST_TIMEOUT,
)

logger = logging.getLogger("J.A.R.V.I.S")

VISION_SYSTEM_PROMPT = """You are analyzing a live camera image from the user. Your job is to describe what you see clearly and specifically.

RULES:
- Always describe the image in detail. Identify objects, text, people, actions, colors, positions, and any visible context.
- Answer the user's specific question directly (e.g. "what am I holding") with concrete, specific answers. Lead with the answer, then add brief detail.
- Never say you cannot see, are unable to tell, or don't have access to the image unless the image is truly blank, corrupted, or unreadable.
- Be confident and specific. If you see a phone, say "a phone" and what kind if visible. If you see text, read it out. If you see a person, describe what they're doing.
- Keep replies brief and natural (1-3 sentences) unless the user asks for more detail.
- When asked about text in the image: read it clearly and completely (OCR-style). Include all visible text.
- When asked about counting: count objects carefully and give the exact number.
- When asked about spatial relationships: describe left/right, above/below, in front of/behind.
- When asked to identify a product, brand, or logo: name it specifically if you can recognize it.
- Do not use asterisks, emojis, or markdown formatting. Use plain text with standard punctuation."""

VISION_UNAVAILABLE_MSG = (
    "Vision is not available right now. Make sure Ollama is running and a vision model "
    "is pulled (e.g. `ollama pull llava`), then set OLLAMA_VISION_MODEL in your .env."
)


class VisionService:

    def __init__(self):
        self._base_url = OLLAMA_BASE_URL
        self._model = OLLAMA_VISION_MODEL
        logger.info("[VISION] Ollama vision initialized (model: %s at %s)", OLLAMA_VISION_MODEL, OLLAMA_BASE_URL)

    def describe_image(
        self,
        img_base64: str,
        prompt: Optional[str] = None,
    ) -> str:

        if "," in img_base64:
            img_base64 = img_base64.split(",", 1)[-1]

        if not img_base64:
            return "No image data received."

        raw_len = 0

        try:
            raw = base64.b64decode(img_base64, validate=True)
            raw_len = len(raw)

            if raw_len > VISION_MAX_IMAGE_BYTES:
                logger.warning("[VISION] Image too large: %d bytes (max %d)", raw_len, VISION_MAX_IMAGE_BYTES)
                return "Image is too large. Please use a smaller image."

        except Exception as e:
            logger.warning("[VISION] Invalid base64 image: %s", e)
            return "Invalid image data. Please try again."

        user_question = (prompt or "What do you see in this image? Describe it in detail.").strip()
        logger.info("[VISION] Analyzing image (%d bytes, model: %s)", raw_len, self._model)

        messages = [
            {"role": "system", "content": VISION_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": user_question,
                "images": [img_base64],
            },
        ]

        try:
            with httpx.Client(timeout=OLLAMA_REQUEST_TIMEOUT) as client:
                resp = client.post(
                    f"{self._base_url}/api/chat",
                    json={
                        "model": self._model,
                        "messages": messages,
                        "stream": False,
                        "think": False,  # skip reasoning tokens (no-op for non-thinking models)
                        "options": {"num_predict": 600},
                    },
                )
                resp.raise_for_status()
                data = resp.json()

            text = (data.get("message") or {}).get("content", "").strip()

            if text:
                logger.info("[VISION] Ollama vision success (%d chars)", len(text))
                return text

            logger.warning("[VISION] Ollama vision returned empty content")
            return "I couldn't analyze that image. Please try again."

        except httpx.HTTPStatusError as e:
            body = (e.response.text or "").lower()
            logger.warning("[VISION] Ollama error %s: %s", e.response.status_code, e.response.text[:200])
            if "does not support" in body or "multimodal" in body:
                logger.warning("[VISION] Model '%s' is not vision-capable", self._model)
                return VISION_UNAVAILABLE_MSG
            return "I couldn't analyze that image. Please try again."

        except Exception as e:
            err_str = str(e).lower()

            if "does not support images" in err_str or "vision" in err_str:
                logger.warning("[VISION] Model %s may not support images: %s", self._model, e)
                return VISION_UNAVAILABLE_MSG

            logger.warning("[VISION] Ollama vision failed: %s", e)
            return VISION_UNAVAILABLE_MSG
