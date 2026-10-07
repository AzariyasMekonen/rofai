import time
from collections import defaultdict, deque

from fastapi import HTTPException, Request

_PRUNE_THRESHOLD = 10_000


class RateLimiter:
    """In-memory sliding-window limiter per client IP. Single-process only."""

    def __init__(self, limit: int, window_seconds: float = 60.0):
        self.limit = limit
        self.window = window_seconds
        self._hits: dict[str, deque[float]] = defaultdict(deque)

    def __call__(self, request: Request) -> None:
        key = request.client.host if request.client else "unknown"
        now = time.monotonic()
        hits = self._hits[key]
        while hits and now - hits[0] > self.window:
            hits.popleft()
        if len(hits) >= self.limit:
            raise HTTPException(status_code=429, detail="Too many requests, please wait a moment.")
        hits.append(now)
        if len(self._hits) > _PRUNE_THRESHOLD:
            self._prune(now)

    def _prune(self, now: float) -> None:
        for key in [k for k, q in self._hits.items() if not q or now - q[-1] > self.window]:
            del self._hits[key]
