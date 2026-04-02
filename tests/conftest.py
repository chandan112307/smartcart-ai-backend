"""Shared test configuration and fixtures."""

import pytest

import app.agents.normalization as _normalization_mod


@pytest.fixture(autouse=True)
def _clear_normalization_cache():
    """Clear the in-memory normalization cache between tests."""
    _normalization_mod._normalization_cache.clear()
    yield
    _normalization_mod._normalization_cache.clear()
