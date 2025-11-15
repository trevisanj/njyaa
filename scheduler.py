# FILE: scheduler.py
import time, threading, heapq
from typing import Callable, Any, List, Tuple

class InternalScheduler:
    """
    Tiny recurring job scheduler.
    - Threaded (not async)
    - Supports recurring jobs with jitter-free cadence
    """
    def __init__(self, name="rv-scheduler"):
        self._name = name
        self._stop = threading.Event()
        self._cv = threading.Condition()
        self._heap: List[Tuple[float,int,Callable,tuple,dict,float]] = []
        self._seq = 0
        self._thr = None

    def start(self):
        if self._thr and self._thr.is_alive(): return
        self._thr = threading.Thread(target=self._run, name=self._name, daemon=True)
        self._thr.start()

    def stop(self, timeout: float = 2.0):
        self._stop.set()
        with self._cv: self._cv.notify_all()
        if self._thr: self._thr.join(timeout=timeout)

    def every(self, interval_sec: float, fn: Callable, *args: Any, first_delay: float | None = None, **kwargs: Any):
        """
        Schedule a recurring job.
        - interval_sec: cadence
        - first_delay: optional delay before first run (default = interval)
        """
        first_at = time.monotonic() + (interval_sec if first_delay is None else first_delay)
        with self._cv:
            heapq.heappush(self._heap, (first_at, self._seq, fn, args, kwargs, interval_sec))
            self._seq += 1
            self._cv.notify_all()

    def _run(self):
        while not self._stop.is_set():
            with self._cv:
                if not self._heap:
                    self._cv.wait(timeout=0.5); continue
                when, seq, fn, args, kwargs, interval = self._heap[0]
                now = time.monotonic()
                if now < when:
                    self._cv.wait(timeout=min(when-now, 0.5)); continue
                # pop due job
                heapq.heappop(self._heap)
            try:
                fn(*args, **kwargs)
            except Exception as e:
                # you can import your log here if you want, but keep scheduler standalone
                print(f"[scheduler] job error: {e}")
            # reinsert next occurrence at fixed cadence
            next_at = when + interval
            with self._cv:
                heapq.heappush(self._heap, (next_at, self._seq, fn, args, kwargs, interval))
                self._seq += 1
