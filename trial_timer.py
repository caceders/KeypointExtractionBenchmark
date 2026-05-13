import time
import threading
import ctypes
import statistics
from collections import deque


class TrialTimeoutError(BaseException):
    pass


class TrialTimer:
    """
    Two-layer trial timeout:
      1. check(remaining) — called after each processed item; uses a rolling
         window average to project completion time and raises early if the trial
         is projected to exceed the budget. Requires min_samples items before
         projecting so a single slow outlier doesn't trigger a premature kill.
      2. Watchdog thread — hard backstop that fires after exactly `timeout`
         seconds regardless of item granularity, covering the case where a
         single item hangs indefinitely.

    Always call cancel() when the trial finishes (use try/finally).
    """

    def __init__(self, timeout: float, window: int = 10, min_samples: int = 5, min_fraction: float = 0.2):
        self._timeout = timeout
        self._start = time.monotonic()
        self._last = self._start
        self._window: deque[float] = deque(maxlen=window)
        self._min_samples = min_samples

        self._min_fraction = min_fraction
        self._cancel = threading.Event()
        self._target_tid = threading.current_thread().ident
        threading.Thread(target=self._watchdog, daemon=True).start()

    def _watchdog(self):
        if not self._cancel.wait(timeout=self._timeout):
            ctypes.pythonapi.PyThreadState_SetAsyncExc(
                ctypes.c_ulong(self._target_tid),
                ctypes.py_object(TrialTimeoutError),
            )

    def cancel(self):
        self._cancel.set()

    def check(self, remaining: int, total: int):
        now = time.monotonic()
        self._window.append(now - self._last)
        self._last = now
        elapsed = now - self._start

        if elapsed > self._timeout:
            raise TrialTimeoutError("Hard timeout exceeded")

        processed = total - remaining
        if processed >= self._min_samples and processed / total >= self._min_fraction:
            avg = statistics.median(self._window)
            if elapsed + avg * remaining > self._timeout:
                raise TrialTimeoutError("Projected to exceed timeout")
