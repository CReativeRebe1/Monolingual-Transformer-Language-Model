"""Shared pytest fixtures for the compliance test suite."""
import pytest


@pytest.fixture
def anyio_backend():
    """Restrict anyio-marked async tests to the asyncio backend (trio is not a project dependency)."""
    return "asyncio"
