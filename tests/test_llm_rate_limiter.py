"""Tests for the LLM rate limiter in app.llm.manager."""

import asyncio
import time
from unittest.mock import AsyncMock, patch

import pytest

from app.llm.manager import LLMManager, LLMRateLimiter


class TestLLMRateLimiter:
    """Unit tests for LLMRateLimiter."""

    async def test_allows_requests_under_limit(self):
        limiter = LLMRateLimiter(max_rpm=5)
        for _ in range(5):
            await limiter.acquire()
        assert limiter.remaining() == 0

    async def test_remaining_decreases(self):
        limiter = LLMRateLimiter(max_rpm=10)
        assert limiter.remaining() == 10
        await limiter.acquire()
        assert limiter.remaining() == 9
        await limiter.acquire()
        assert limiter.remaining() == 8

    async def test_max_rpm_property(self):
        limiter = LLMRateLimiter(max_rpm=25)
        assert limiter.max_rpm == 25

    async def test_min_max_rpm_clamped_to_one(self):
        limiter = LLMRateLimiter(max_rpm=0)
        assert limiter.max_rpm == 1
        limiter2 = LLMRateLimiter(max_rpm=-5)
        assert limiter2.max_rpm == 1

    async def test_waits_when_limit_reached(self):
        """When the window is full, acquire() should sleep until a slot opens."""
        limiter = LLMRateLimiter(max_rpm=2)

        # Fill the window
        await limiter.acquire()
        await limiter.acquire()
        assert limiter.remaining() == 0

        # The next acquire should wait — we patch asyncio.sleep to verify
        with patch("app.llm.manager.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            await limiter.acquire()
            mock_sleep.assert_awaited_once()
            # The sleep duration should be positive and <= 60 seconds
            wait_seconds = mock_sleep.call_args[0][0]
            assert 0 < wait_seconds <= 60

    async def test_prune_removes_old_timestamps(self):
        limiter = LLMRateLimiter(max_rpm=5)
        # Manually insert an old timestamp (older than 60s)
        old_time = time.monotonic() - 120
        limiter._timestamps.append(old_time)
        # After pruning the stale entry is gone, all 5 slots are free
        assert limiter.remaining() == 5
        assert len(limiter._timestamps) == 0

    async def test_default_config_is_25_rpm(self):
        """Verify the default LLM rate limit from settings is 25."""
        from app.core.config import get_settings

        settings = get_settings()
        assert settings.llm_rate_limit_rpm == 25


class TestLLMManagerRateLimiter:
    """Integration tests verifying LLMManager uses the rate limiter."""

    async def test_manager_has_rate_limiter(self):
        manager = LLMManager()
        assert hasattr(manager, "_rate_limiter")
        assert isinstance(manager._rate_limiter, LLMRateLimiter)
        assert manager._rate_limiter.max_rpm == 25

    async def test_call_acquires_rate_limit_slot(self):
        """Each actual LLM provider call should first acquire a rate-limiter slot."""
        manager = LLMManager()
        manager._rate_limiter = LLMRateLimiter(max_rpm=25)

        # No provider configured → acquire should NOT be called (no outbound request)
        with patch.object(
            manager._rate_limiter, "acquire", new_callable=AsyncMock
        ) as mock_acquire:
            with pytest.raises(Exception):
                await manager.call("test prompt")
            # Rate limiter is only invoked inside _call_openai / _call_groq
            # which are skipped when no client is configured.
            mock_acquire.assert_not_awaited()

    async def test_remaining_slots_decrease_after_calls(self):
        """When a provider is configured, rate-limit slots decrease after calls."""
        manager = LLMManager()
        manager._rate_limiter = LLMRateLimiter(max_rpm=3)

        # Simulate a configured client that returns a valid response
        manager._initialized = True
        manager._provider = "groq"
        mock_client = AsyncMock()
        mock_response = AsyncMock()
        mock_response.choices = [AsyncMock(message=AsyncMock(content='{"ok": true}'))]
        mock_client.chat.completions.create.return_value = mock_response
        manager._groq_client = mock_client

        initial = manager._rate_limiter.remaining()
        assert initial == 3

        # Patch run_in_executor to call the lambda directly
        with patch("asyncio.get_event_loop") as mock_loop:
            mock_loop.return_value.run_in_executor = AsyncMock(
                return_value=mock_response
            )
            await manager.call("test")
            await manager.call("test")

        assert manager._rate_limiter.remaining() == 1
