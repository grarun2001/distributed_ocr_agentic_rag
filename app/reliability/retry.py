"""Exponential backoff retry policy with jitter for synchronous and asynchronous tasks."""

import asyncio
import functools
import inspect
import random
import time
from typing import Any, Callable, Optional, Sequence, Tuple, Type, TypeVar

from app.core.logging import get_logger

logger = get_logger("retry_policy")

F = TypeVar("F", bound=Callable[..., Any])


def calculate_delay(
    attempt: int,
    base_delay: float,
    max_delay: float,
    exponential_factor: float,
    jitter: bool
) -> float:
    """Calculate exponential backoff delay with optional random jitter."""
    delay = base_delay * (exponential_factor ** (attempt - 1))
    delay = min(delay, max_delay)
    if jitter:
        # Full jitter between 50% and 150% of the computed delay
        delay = delay * random.uniform(0.5, 1.5)
    return max(0.01, delay)


def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 0.5,
    max_delay: float = 10.0,
    exponential_factor: float = 2.0,
    jitter: bool = True,
    retryable_exceptions: Tuple[Type[Exception], ...] = (Exception,)
) -> Callable[[F], F]:
    """
    Decorator for retrying transient network, database, or rate-limited API failures
    using exponential backoff and jitter.
    """

    def decorator(func: F) -> F:
        if inspect.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                last_exc: Optional[Exception] = None
                for attempt in range(1, max_retries + 1):
                    try:
                        return await func(*args, **kwargs)
                    except retryable_exceptions as exc:
                        last_exc = exc
                        if attempt == max_retries:
                            logger.error(
                                f"Function '{func.__name__}' failed after {max_retries} attempts: {exc}"
                            )
                            raise
                        delay = calculate_delay(attempt, base_delay, max_delay, exponential_factor, jitter)
                        logger.warning(
                            f"Function '{func.__name__}' failed (attempt {attempt}/{max_retries}): {exc}. "
                            f"Retrying in {delay:.2f}s..."
                        )
                        await asyncio.sleep(delay)
                if last_exc:
                    raise last_exc

            return async_wrapper  # type: ignore[return-value]
        else:
            @functools.wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                last_exc: Optional[Exception] = None
                for attempt in range(1, max_retries + 1):
                    try:
                        return func(*args, **kwargs)
                    except retryable_exceptions as exc:
                        last_exc = exc
                        if attempt == max_retries:
                            logger.error(
                                f"Function '{func.__name__}' failed after {max_retries} attempts: {exc}"
                            )
                            raise
                        delay = calculate_delay(attempt, base_delay, max_delay, exponential_factor, jitter)
                        logger.warning(
                            f"Function '{func.__name__}' failed (attempt {attempt}/{max_retries}): {exc}. "
                            f"Retrying in {delay:.2f}s..."
                        )
                        time.sleep(delay)
                if last_exc:
                    raise last_exc

            return sync_wrapper  # type: ignore[return-value]

    return decorator
