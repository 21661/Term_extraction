
import asyncio
import random
import time
from collections import deque
from typing import Dict, List, Optional, Tuple, Set

try:
    # optional import; only for type hints
    from langchain_openai import ChatOpenAI
except Exception:  # pragma: no cover
    ChatOpenAI = object  # type: ignore


class LLMTokenPool:
    """
    A simple global pool that coordinates concurrency (max_concurrency)
    and RPM (requests-per-minute) across *all* workflows sharing the same LLMs.
    Keys are the actual llm objects (by id), so multiple wrappers resolving to
    the same object will be treated as one slot.

    Extra config per llm supports:
        {
            "max_concurrency": int (>=0),
            "rpm": int | None,   # None: no rpm cap; 0: disabled; >0: rpm cap
            "type": str | None,  # e.g. "mt"
        }
    """
    def __init__(self):
        # static config
        self._llms: Dict[int, object] = {}
        self._types: Dict[int, str] = {}
        self._capacity: Dict[int, int] = {}
        self._rpm: Dict[int, Optional[int]] = {}

        # dynamic state
        self._usage: Dict[int, int] = {}
        self._locks: Dict[int, asyncio.Lock] = {}
        self._windows: Dict[int, deque] = {}

        # global lock for registration
        self._register_lock = asyncio.Lock()

    def register_agent_llms(self, llm_configs: List[Tuple[object, dict]]) -> None:
        """
        Register multiple (llm, extra_config) pairs.
        Safe to call repeatedly; re-registration is idempotent.
        """
        import contextlib
        # simple non-async critical section; assume called during init time
        # (if used concurrently, last write wins but is idempotent)
        # Using no async lock to keep API sync.
        with contextlib.ExitStack():
            for llm, extra in llm_configs:
                lid = id(llm)
                if lid in self._llms:
                    # already registered; allow updates for rpm/capacity/type
                    pass
                self._llms[lid] = llm
                et = (extra or {})
                t = str(et.get("type", "") or "").lower()
                cap = int(et.get("max_concurrency", 1))
                rpm = et.get("rpm", None)

                self._types[lid] = t
                self._capacity[lid] = max(0, int(cap))
                self._rpm[lid] = rpm if rpm is None else int(rpm)

                self._usage.setdefault(lid, 0)
                self._locks.setdefault(lid, asyncio.Lock())
                self._windows.setdefault(lid, deque())

    def _eligible_ids(self, preferred_type: Optional[str], exclude_types: Optional[Set[str]]) -> List[int]:
        ids = [lid for lid in self._llms.keys() if self._capacity.get(lid, 0) > 0]
        if preferred_type:
            p = str(preferred_type).lower()
            ids = [lid for lid in ids if self._types.get(lid, "") == p]
        if exclude_types:
            ex = {str(x).lower() for x in exclude_types}
            ids = [lid for lid in ids if self._types.get(lid, "") not in ex]
        random.shuffle(ids)
        return ids

    async def acquire(self, preferred_type: Optional[str] = None, exclude_types: Optional[Set[str]] = None):
        """
        Reserve a concurrency slot and an rpm slot for one LLM and return that LLM.
        Will wait if nothing is available.
        """
        exclude_types = ({x.lower() for x in exclude_types} if exclude_types else None)
        while True:
            ids = self._eligible_ids(preferred_type, exclude_types)
            now = time.time()
            cutoff = now - 60.0

            for lid in ids:
                # concurrency check
                if self._usage.get(lid, 0) >= self._capacity.get(lid, 0):
                    continue

                # rpm check (with a per-llm lock to avoid races)
                lock = self._locks[lid]
                async with lock:
                    rpm = self._rpm.get(lid, None)
                    if rpm == 0:
                        # disabled
                        continue

                    window = self._windows[lid]
                    # evict old
                    while window and window[0] < cutoff:
                        window.popleft()

                    if rpm is None or len(window) < rpm:
                        # reserve: concurrency + rpm
                        self._usage[lid] = self._usage.get(lid, 0) + 1
                        if rpm is not None:
                            window.append(now)
                        return self._llms[lid]

            await asyncio.sleep(0.25)

    def release(self, llm: object) -> None:
        lid = id(llm)
        if lid in self._usage:
            self._usage[lid] = max(0, self._usage[lid] - 1)

    # Optional: small context manager for ergonomics
    async def __aenter__(self):
        raise RuntimeError("Use .acquire() explicitly or the 'session' helper.")

    async def __aexit__(self, exc_type, exc, tb):
        pass


# Global singleton
llm_pool = LLMTokenPool()