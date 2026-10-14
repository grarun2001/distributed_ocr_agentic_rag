"""Three-state Circuit Breaker implementation preventing cascading failures to external dependencies."""

import asyncio
from enum import Enum
import functools
import inspect
import threading
import time
from typing import Any, Callable, Dict, Optional, TypeVar

from app.core.exceptions import CircuitBreakerOpenError
from app.core.logging import get_logger

logger = get_logger("circuit_breaker")

F = TypeVar("F", bound=Callable[..., Any])


class CircuitState(str, Enum):
    """Lifecycle states of the circuit breaker."""
    CLOSED = "CLOSED"        # Normal execution; passes requests
    OPEN = "OPEN"            # Tripped; blocks requests immediately
    HALF_OPEN = "HALF_OPEN"  # Trial phase; tests if downstream service has recovered


class CircuitBreaker:
    """
    Guards external network calls (e.g. Groq LLM, DuckDuckGo web search, MinIO, Milvus)
    from cascading failures and resource starvation.
    """

    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        half_open_success_threshold: int = 2
    ) -> None:
        self.name = name
        self.failure_threshold = max(1, failure_threshold)
        self.recovery_timeout = max(0.1, recovery_timeout)
        self.half_open_success_threshold = max(1, half_open_success_threshold)

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_state_change = time.time()
        self._last_failure_time: Optional[float] = None
        self._lock = threading.Lock()

    @property
    def state(self) -> CircuitState:
        with self._lock:
            self._evaluate_state_transition()
            return self._state

    @property
    def failure_count(self) -> int:
        with self._lock:
            return self._failure_count

    @property
    def success_count(self) -> int:
        with self._lock:
            return self._success_count

    def _evaluate_state_transition(self) -> None:
        """Internal transition logic from OPEN to HALF_OPEN after recovery timeout."""
        if self._state == CircuitState.OPEN and self._last_failure_time is not None:
            elapsed = time.time() - self._last_failure_time
            if elapsed >= self.recovery_timeout:
                self._state = CircuitState.HALF_OPEN
                self._success_count = 0
                self._last_state_change = time.time()
                logger.info(
                    f"Circuit breaker '{self.name}' transitioned OPEN -> HALF_OPEN (elapsed: {elapsed:.1f}s)"
                )

    def before_call(self) -> None:
        """Validate if request can proceed or raise CircuitBreakerOpenError."""
        with self._lock:
            self._evaluate_state_transition()
            if self._state == CircuitState.OPEN:
                logger.warning(
                    f"Circuit breaker '{self.name}' is OPEN. Blocking request."
                )
                raise CircuitBreakerOpenError(self.name)

    def record_success(self) -> None:
        """Record successful execution and handle state transitions."""
        with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                logger.debug(
                    f"Circuit breaker '{self.name}' recorded HALF_OPEN success ({self._success_count}/{self.half_open_success_threshold})"
                )
                if self._success_count >= self.half_open_success_threshold:
                    self._state = CircuitState.CLOSED
                    self._failure_count = 0
                    self._success_count = 0
                    self._last_state_change = time.time()
                    logger.info(f"Circuit breaker '{self.name}' transitioned HALF_OPEN -> CLOSED (Service Healthy)")
            elif self._state == CircuitState.CLOSED:
                self._failure_count = 0

    def record_failure(self, exc: Exception) -> None:
        """Record failed execution and trip circuit if threshold exceeded."""
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()
            logger.warning(
                f"Circuit breaker '{self.name}' recorded failure #{self._failure_count}: {exc}"
            )

            if self._state == CircuitState.HALF_OPEN:
                # Any failure during half-open trips back to open immediately
                self._state = CircuitState.OPEN
                self._success_count = 0
                self._last_state_change = time.time()
                logger.error(f"Circuit breaker '{self.name}' transitioned HALF_OPEN -> OPEN (Probe Failed)")
            elif self._state == CircuitState.CLOSED:
                if self._failure_count >= self.failure_threshold:
                    self._state = CircuitState.OPEN
                    self._last_state_change = time.time()
                    logger.error(
                        f"Circuit breaker '{self.name}' threshold reached ({self._failure_count}/{self.failure_threshold}). Tripping to OPEN."
                    )

    def reset(self) -> None:
        """Manually force circuit breaker to CLOSED state."""
        with self._lock:
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._success_count = 0
            self._last_failure_time = None
            self._last_state_change = time.time()
            logger.info(f"Circuit breaker '{self.name}' manually reset to CLOSED.")

    def __call__(self, func: F) -> F:
        """Decorator for protecting synchronous or asynchronous functions."""
        if inspect.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                self.before_call()
                try:
                    result = await func(*args, **kwargs)
                    self.record_success()
                    return result
                except Exception as exc:
                    self.record_failure(exc)
                    raise
            return async_wrapper  # type: ignore[return-value]
        else:
            @functools.wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                self.before_call()
                try:
                    result = func(*args, **kwargs)
                    self.record_success()
                    return result
                except Exception as exc:
                    self.record_failure(exc)
                    raise
            return sync_wrapper  # type: ignore[return-value]


# Registry for named singleton circuit breakers
_registry: Dict[str, CircuitBreaker] = {}
_registry_lock = threading.Lock()


def get_circuit_breaker(
    name: str,
    failure_threshold: int = 5,
    recovery_timeout: float = 30.0,
    half_open_success_threshold: int = 2
) -> CircuitBreaker:
    """Retrieve or create a registered CircuitBreaker instance."""
    with _registry_lock:
        if name not in _registry:
            _registry[name] = CircuitBreaker(
                name=name,
                failure_threshold=failure_threshold,
                recovery_timeout=recovery_timeout,
                half_open_success_threshold=half_open_success_threshold
            )
        return _registry[name]
