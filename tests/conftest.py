"""Pytest configuration and global fixtures."""

import pytest
import pytest_asyncio
from app.db.session import async_engine


@pytest_asyncio.fixture(autouse=True)
async def cleanup_db_engine():
    """Dispose of engine connection pool after each test to prevent event loop reuse issues."""
    yield
    await async_engine.dispose()
