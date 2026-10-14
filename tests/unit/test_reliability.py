"""Unit tests for Circuit Breaker and Exponential Backoff Retry Policy."""

import asyncio
import time
import pytest

from app.core.exceptions import CircuitBreakerOpenError
from app.reliability.circuit_breaker import CircuitBreaker, CircuitState
from app.reliability.retry import calculate_delay, retry_with_backoff


def test_circuit_breaker_closed_to_open_transition():
    """Verify circuit breaker trips from CLOSED to OPEN after failure threshold."""
    cb = CircuitBreaker(name="test_service", failure_threshold=3, recovery_timeout=0.2)

    assert cb.state == CircuitState.CLOSED
    assert cb.failure_count == 0

    # Record 2 failures - should still be CLOSED
    cb.record_failure(ValueError("err 1"))
    cb.record_failure(ValueError("err 2"))
    assert cb.state == CircuitState.CLOSED
    assert cb.failure_count == 2

    # 3rd failure trips to OPEN
    cb.record_failure(ValueError("err 3"))
    assert cb.state == CircuitState.OPEN

    # Subsequent before_call must raise CircuitBreakerOpenError immediately
    with pytest.raises(CircuitBreakerOpenError):
        cb.before_call()


def test_circuit_breaker_half_open_recovery():
    """Verify circuit transitions to HALF_OPEN after timeout and resets to CLOSED after successes."""
    cb = CircuitBreaker(
        name="test_recovery",
        failure_threshold=2,
        recovery_timeout=0.1,
        half_open_success_threshold=2
    )

    cb.record_failure(RuntimeError("fail 1"))
    cb.record_failure(RuntimeError("fail 2"))
    assert cb.state == CircuitState.OPEN

    # Wait for recovery timeout
    time.sleep(0.15)
    # State evaluation should now show HALF_OPEN
    assert cb.state == CircuitState.HALF_OPEN

    # 1st success in HALF_OPEN
    cb.record_success()
    assert cb.state == CircuitState.HALF_OPEN
    assert cb.success_count == 1

    # 2nd success recovers to CLOSED
    cb.record_success()
    assert cb.state == CircuitState.CLOSED
    assert cb.failure_count == 0


def test_circuit_breaker_sync_decorator():
    """Verify circuit breaker decorator protects synchronous functions."""
    cb = CircuitBreaker(name="sync_test", failure_threshold=2, recovery_timeout=1.0)

    call_count = 0

    @cb
    def unstable_func(should_fail: bool):
        nonlocal call_count
        call_count += 1
        if should_fail:
            raise ConnectionError("Network failure")
        return "success"

    # Success call
    assert unstable_func(False) == "success"
    assert call_count == 1

    # 2 consecutive failures trip the breaker
    with pytest.raises(ConnectionError):
        unstable_func(True)
    with pytest.raises(ConnectionError):
        unstable_func(True)

    assert cb.state == CircuitState.OPEN

    # 4th call is blocked by circuit breaker before entering function
    with pytest.raises(CircuitBreakerOpenError):
        unstable_func(False)

    assert call_count == 3  # Function was not entered on 4th call


@pytest.mark.asyncio
async def test_circuit_breaker_async_decorator():
    """Verify circuit breaker decorator protects asynchronous functions."""
    cb = CircuitBreaker(name="async_test", failure_threshold=2, recovery_timeout=1.0)

    @cb
    async def async_fetch(fail: bool):
        await asyncio.sleep(0.01)
        if fail:
            raise TimeoutError("Async timeout")
        return "ok"

    res = await async_fetch(False)
    assert res == "ok"

    with pytest.raises(TimeoutError):
        await async_fetch(True)
    with pytest.raises(TimeoutError):
        await async_fetch(True)

    assert cb.state == CircuitState.OPEN

    with pytest.raises(CircuitBreakerOpenError):
        await async_fetch(False)


def test_retry_with_backoff_sync_succeeds_on_retry():
    """Verify sync retry succeeds on subsequent attempt."""
    attempts = 0

    @retry_with_backoff(max_retries=3, base_delay=0.01, max_delay=0.1, jitter=False)
    def flaky_service():
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise ConnectionResetError("Temporary reset")
        return "recovered"

    result = flaky_service()
    assert result == "recovered"
    assert attempts == 3


def test_retry_with_backoff_sync_exhaustion():
    """Verify sync retry re-raises error when attempts exhausted."""
    attempts = 0

    @retry_with_backoff(max_retries=3, base_delay=0.01, max_delay=0.05, jitter=False)
    def always_fails():
        nonlocal attempts
        attempts += 1
        raise ValueError("Permanent fatal error")

    with pytest.raises(ValueError):
        always_fails()

    assert attempts == 3


@pytest.mark.asyncio
async def test_retry_with_backoff_async():
    """Verify async retry succeeds on retry."""
    attempts = 0

    @retry_with_backoff(max_retries=3, base_delay=0.01, max_delay=0.05, jitter=False)
    async def async_flaky():
        nonlocal attempts
        attempts += 1
        await asyncio.sleep(0.01)
        if attempts < 2:
            raise IOError("Transient IO error")
        return "async_ok"

    result = await async_flaky()
    assert result == "async_ok"
    assert attempts == 2
