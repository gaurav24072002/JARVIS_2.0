from typing import List, Optional, Iterator
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
import logging
import time
from app.services.vector_store import VectorStoreService
from app.utils.time_info import get_time_information
from app.utils.retry import with_retry
from config import (
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    OLLAMA_REQUEST_TIMEOUT,
    JARVIS_SYSTEM_PROMPT,
    GENERAL_CHAT_ADDENDUM,
)

logger = logging.getLogger("J.A.R.V.I.S")
LLM_REQUEST_TIMEOUT = OLLAMA_REQUEST_TIMEOUT

ALL_LLM_FAILED_MESSAGE = (
    "I'm unable to process your request at the moment. The local LLM is "
    "temporarily unavailable. Please check that Ollama is running and try again."
)


class LlmUnavailableError(Exception):
    pass


def escape_curly_braces(text: str) -> str:
    if not text:
        return text
    return text.replace("{", "{{").replace("}", "}}")


_REPEAT_WINDOW = 100
_REPEAT_THRESHOLD = 3
_REPEAT_CHECK_INTERVAL = 200


def _detect_repetition_loop(text: str) -> bool:

    if len(text) < _REPEAT_WINDOW * _REPEAT_THRESHOLD:
        return False

    phrase = text[-_REPEAT_WINDOW:]
    return text.count(phrase) >= _REPEAT_THRESHOLD


def _truncate_at_repetition(text: str) -> str:

    if len(text) < _REPEAT_WINDOW * _REPEAT_THRESHOLD:
        return text

    phrase = text[-_REPEAT_WINDOW:]
    if text.count(phrase) < _REPEAT_THRESHOLD:
        return text

    first = text.find(phrase)
    second = text.find(phrase, first + 1)

    if second > first:
        return text[:second].rstrip()
    return text


def _log_timing(label: str, elapsed: float, extra: str = ""):
    msg = f"[TIMING] {label}: {elapsed:.3f}s"

    if extra:
        msg += f" ({extra})"
    logger.info(msg)


class LlmService:

    def __init__(self, vector_store_service: VectorStoreService):

        try:
            self.llm = ChatOllama(
                model=OLLAMA_MODEL,
                base_url=OLLAMA_BASE_URL,
                temperature=0.5,
                num_predict=1024,
                reasoning=False,  # nemotron is a thinking model; disable for fast direct answers
                client_kwargs={"timeout": LLM_REQUEST_TIMEOUT},
            )
        except Exception as e:
            raise ValueError(
                f"Failed to initialize local Ollama client ({OLLAMA_BASE_URL}): {e}. "
                "Make sure Ollama is running (e.g. `ollama serve`)."
            ) from e

        self.vector_store_service = vector_store_service
        logger.info(
            "Initialized LlmService with local Ollama model '%s' at %s (no API keys)",
            OLLAMA_MODEL, OLLAMA_BASE_URL,
        )

    def _invoke_llm(
        self,
        prompt: ChatPromptTemplate,
        messages: list,
        question: str,
        key_start_index: int = 0,  # kept for API compatibility; single local model
    ) -> str:

        last_exc = None

        try:
            def _invoke():
                chain = prompt | self.llm
                return chain.invoke({"history": messages, "question": question})

            response = with_retry(
                _invoke,
                max_retries=2,
                initial_delay=0.5,
            )

            text = response.content

            if _detect_repetition_loop(text):
                logger.warning("[INVOKE] Repetition loop detected — truncating response (%d chars)", len(text))
                text = _truncate_at_repetition(text)
            return text

        except Exception as e:
            last_exc = e
            logger.warning("[INVOKE] Local LLM failed: %s", str(e)[:200])

        logger.error("Local LLM call failed.")
        raise LlmUnavailableError(ALL_LLM_FAILED_MESSAGE) from last_exc

    def _stream_llm(
        self,
        prompt: ChatPromptTemplate,
        messages: list,
        question: str,
        key_start_index: int = 0,  # kept for API compatibility; single local model
    ) -> Iterator[str]:

        try:
            chain = prompt | self.llm
            chunk_count = 0
            first_chunk_time = None
            stream_start = time.perf_counter()
            accumulated = ""
            last_check_len = 0
            repetition_stopped = False

            for chunk in chain.stream({"history": messages, "question": question}):
                content = ""

                if hasattr(chunk, "content"):
                    content = chunk.content or ""

                elif isinstance(chunk, dict) and "content" in chunk:
                    content = chunk.get("content", "") or ""

                if isinstance(content, str) and content:
                    if first_chunk_time is None:
                        first_chunk_time = time.perf_counter() - stream_start
                        _log_timing("first_chunk", first_chunk_time)

                    chunk_count += 1
                    accumulated += content

                    if len(accumulated) - last_check_len >= _REPEAT_CHECK_INTERVAL:
                        last_check_len = len(accumulated)

                        if _detect_repetition_loop(accumulated):
                            logger.warning("[STREAM] Repetition loop detected after %d chars — stopping", len(accumulated))
                            repetition_stopped = True
                            break

                    yield content

            total_stream = time.perf_counter() - stream_start
            _log_timing("ollama_stream_total", total_stream, f"chunks: {chunk_count}{', TRUNCATED-REPETITION' if repetition_stopped else ''}")
            return

        except Exception as e:
            logger.warning("[STREAM] Local LLM failed: %s", str(e)[:200])
            raise

    def _build_prompt_and_messages(
        self,
        question: str,
        chat_history: Optional[List[tuple]] = None,
        extra_system_parts: Optional[List[str]] = None,
        mode_addendum: str = "",
    ) -> tuple:

        context = ""
        context_sources = []
        t0 = time.perf_counter()

        try:
            retriever = self.vector_store_service.get_retriever(k=5)
            context_docs = retriever.invoke(question)

            if context_docs:
                context = "\n".join([doc.page_content for doc in context_docs])
                context_sources = [doc.metadata.get("source", "unknown") for doc in context_docs]
                logger.info("[CONTEXT] Retrieved %d chunks from sources: %s", len(context_docs), context_sources)
            else:
                logger.info("[CONTEXT] No relevant chunks found for query")

        except Exception as retrieval_err:
            logger.warning("Vector store retrieval failed, using empty context: %s", retrieval_err)

        finally:
            _log_timing("vector_db", time.perf_counter() - t0)

        time_info = get_time_information()
        system_message = JARVIS_SYSTEM_PROMPT
        system_message += f"\n\nCurrent time and date: {time_info}"

        if context:
            system_message += f"\n\nRelevant context from your learning data and past conversations:\n{escape_curly_braces(context)}"

        if extra_system_parts:
            system_message += "\n\n" + "\n\n".join(extra_system_parts)

        if mode_addendum:
            system_message += f"\n\n{mode_addendum}"

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_message),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}"),
        ])

        messages = []

        if chat_history:
            for human_msg, ai_msg in chat_history:
                messages.append(HumanMessage(content=human_msg))
                messages.append(AIMessage(content=ai_msg))

        logger.info("[PROMPT] System message length: %d chars | History pairs: %d | Question: %.100s",
                    len(system_message), len(chat_history) if chat_history else 0, question)

        return prompt, messages

    def get_response(
        self,
        question: str,
        chat_history: Optional[List[tuple]] = None,
        key_start_index: int = 0,
    ) -> str:

        try:
            prompt, messages = self._build_prompt_and_messages(
                question, chat_history, mode_addendum=GENERAL_CHAT_ADDENDUM,
            )
            t0 = time.perf_counter()
            result = self._invoke_llm(prompt, messages, question)
            _log_timing("ollama_api", time.perf_counter() - t0)
            logger.info("[RESPONSE] General chat | Length: %d chars | Preview: %.120s", len(result), result)
            return result

        except LlmUnavailableError:
            raise

        except Exception as e:
            raise Exception(f"Error getting response from local LLM: {str(e)}") from e

    def stream_response(
        self,
        question: str,
        chat_history: Optional[List[tuple]] = None,
        key_start_index: int = 0,
    ) -> Iterator:

        try:
            prompt, messages = self._build_prompt_and_messages(
                question, chat_history, mode_addendum=GENERAL_CHAT_ADDENDUM,
            )
            yield {"_activity": {"event": "context_retrieved", "message": "Retrieved relevant context from knowledge base"}}
            yield from self._stream_llm(prompt, messages, question)

        except LlmUnavailableError:
            raise

        except Exception as e:
            raise Exception(f"Error streaming response from local LLM: {str(e)}") from e
