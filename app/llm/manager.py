"""LLM Manager: wraps OpenAI / Groq calls with structured JSON output enforcement.

All outputs are forced to valid JSON via either:
  1. Native JSON mode (OpenAI)
  2. Prompt-level instruction + schema validation fallback

Includes a sliding-window rate limiter that caps outbound LLM requests
to ``settings.llm_rate_limit_rpm`` per 60-second window (default: 25 RPM).
When the limit is reached the caller sleeps until a slot opens rather than
firing requests that will be rejected with HTTP 429.
"""

import asyncio
import json
import logging
import re
import time
from collections import deque
from typing import Any, Deque, Dict, Optional, Type

from pydantic import BaseModel

from app.core.config import get_settings
from app.core.exceptions import LLMException, LLMRateLimitException

settings = get_settings()
logger = logging.getLogger(__name__)

_RATE_LIMIT_WINDOW_SECONDS = 60


def _build_json_prompt(user_prompt: str, schema_example: Optional[str] = None) -> str:
    """Wrap a prompt to enforce strict JSON output."""
    instruction = "You are a JSON-only AI. ALWAYS respond with valid JSON and nothing else. Do not add markdown code fences."
    if schema_example:
        instruction += f"\n\nExpected JSON schema/example:\n{schema_example}"
    return f"{instruction}\n\n{user_prompt}"


def _parse_json_response(text: str) -> Dict[str, Any]:
    """Extract JSON from LLM text response, stripping markdown fences if present."""
    # Strip markdown code fences
    text = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise LLMException(f"LLM returned invalid JSON: {exc}") from exc


class LLMRateLimiter:
    """Async-safe sliding-window rate limiter for outbound LLM requests.

    Tracks timestamps of recent calls in a ``deque`` and enforces a maximum
    of ``max_rpm`` requests per 60-second rolling window.
    """

    def __init__(self, max_rpm: int) -> None:
        self._max_rpm = max(1, max_rpm)
        self._timestamps: Deque[float] = deque()
        self._lock = asyncio.Lock()

    @property
    def max_rpm(self) -> int:
        return self._max_rpm

    def _prune(self, now: float) -> None:
        """Remove timestamps older than the 60-second window."""
        cutoff = now - _RATE_LIMIT_WINDOW_SECONDS
        while self._timestamps and self._timestamps[0] <= cutoff:
            self._timestamps.popleft()

    async def acquire(self) -> None:
        """Wait until a request slot is available, then record the call."""
        async with self._lock:
            now = time.monotonic()
            self._prune(now)

            if len(self._timestamps) >= self._max_rpm:
                # Wait until the oldest entry expires from the window.
                wait_seconds = self._timestamps[0] + _RATE_LIMIT_WINDOW_SECONDS - now
                if wait_seconds > 0:
                    logger.warning(
                        "LLM rate limit reached (%d/%d RPM). "
                        "Waiting %.2fs for a slot.",
                        len(self._timestamps),
                        self._max_rpm,
                        wait_seconds,
                    )
                    await asyncio.sleep(wait_seconds)
                    now = time.monotonic()
                    self._prune(now)

            self._timestamps.append(now)

    def remaining(self) -> int:
        """Return how many requests are still available in the current window."""
        self._prune(time.monotonic())
        return max(0, self._max_rpm - len(self._timestamps))


class LLMManager:
    """Manages LLM API calls with structured output and provider fallback.

    Includes an internal :class:`LLMRateLimiter` that enforces
    ``settings.llm_rate_limit_rpm`` requests per minute.

    Usage:
        manager = LLMManager()
        result = await manager.call(prompt, schema_example="{...}")
    """

    def __init__(self) -> None:
        self._provider = settings.llm_provider
        self._openai_client: Optional[Any] = None
        self._groq_client: Optional[Any] = None
        self._initialized = False
        self._rate_limiter = LLMRateLimiter(max_rpm=settings.llm_rate_limit_rpm)

    def _ensure_clients(self) -> None:
        if self._initialized:
            return
        if settings.openai_api_key:
            try:
                from openai import OpenAI  # type: ignore

                self._openai_client = OpenAI(api_key=settings.openai_api_key)
            except ImportError:
                pass
        if settings.groq_api_key:
            try:
                from groq import Groq  # type: ignore

                self._groq_client = Groq(api_key=settings.groq_api_key)
            except ImportError:
                pass
        self._initialized = True

    async def call(
        self,
        prompt: str,
        schema_example: Optional[str] = None,
        response_model: Optional[Type[BaseModel]] = None,
    ) -> Dict[str, Any]:
        """Call the configured LLM and return a parsed JSON dict.

        Waits for a rate-limiter slot before issuing the request.
        Falls back to the alternative provider if the primary fails.
        Raises :class:`LLMException` if no provider is available.
        """
        self._ensure_clients()
        full_prompt = _build_json_prompt(prompt, schema_example)

        # Try primary provider
        result = await self._try_primary(full_prompt)
        if result is not None:
            return result

        # Try fallback provider
        result = await self._try_fallback(full_prompt)
        if result is not None:
            return result

        # No LLM available — return empty structure so pipeline can continue
        raise LLMException("No LLM provider available or configured.")

    async def _try_primary(self, prompt: str) -> Optional[Dict[str, Any]]:
        if self._provider == "openai" and self._openai_client:
            return await self._call_openai(prompt)
        if self._provider == "groq" and self._groq_client:
            return await self._call_groq(prompt)
        return None

    async def _try_fallback(self, prompt: str) -> Optional[Dict[str, Any]]:
        if self._provider == "openai" and self._groq_client:
            return await self._call_groq(prompt)
        if self._provider == "groq" and self._openai_client:
            return await self._call_openai(prompt)
        return None

    async def _call_openai(self, prompt: str) -> Optional[Dict[str, Any]]:
        try:
            import asyncio

            await self._rate_limiter.acquire()
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self._openai_client.chat.completions.create(
                    model=settings.openai_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=settings.llm_temperature,
                    max_tokens=settings.llm_max_tokens,
                    response_format={"type": "json_object"},
                ),
            )
            text = response.choices[0].message.content or "{}"
            return _parse_json_response(text)
        except Exception:
            return None

    async def _call_groq(self, prompt: str) -> Optional[Dict[str, Any]]:
        try:
            import asyncio

            await self._rate_limiter.acquire()
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self._groq_client.chat.completions.create(
                    model=settings.groq_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=settings.llm_temperature,
                    max_tokens=settings.llm_max_tokens,
                ),
            )
            text = response.choices[0].message.content or "{}"
            return _parse_json_response(text)
        except Exception:
            return None


# Module-level singleton
_llm_manager: Optional[LLMManager] = None


def get_llm_manager() -> LLMManager:
    global _llm_manager
    if _llm_manager is None:
        _llm_manager = LLMManager()
    return _llm_manager
