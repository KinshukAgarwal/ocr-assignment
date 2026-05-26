import time
from dataclasses import dataclass


@dataclass
class CircuitBreaker:
    failure_threshold: int
    reset_seconds: int
    failure_count: int = 0
    opened_at: float | None = None

    def can_call(self) -> bool:
        if self.opened_at is None:
            return True
        return time.monotonic() - self.opened_at >= self.reset_seconds

    def record_success(self) -> None:
        self.failure_count = 0
        self.opened_at = None

    def record_failure(self) -> None:
        self.failure_count += 1
        if self.failure_count >= self.failure_threshold:
            self.opened_at = time.monotonic()
