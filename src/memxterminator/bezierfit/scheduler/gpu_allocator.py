from __future__ import annotations

from collections import deque

from .spec import SchedulerPolicy


class GpuAllocator:
    """
    Deterministic GPU allocator for a fixed set of GPU IDs.

    This allocator does NOT query the system. The caller is responsible for
    passing the list of GPUs that should be considered available to the batch.
    """

    def __init__(self, gpus: list[int], *, policy: SchedulerPolicy) -> None:
        if not gpus:
            raise ValueError("GpuAllocator requires a non-empty GPU list")
        if len(set(gpus)) != len(gpus):
            raise ValueError(f"GpuAllocator GPU list contains duplicates: {gpus}")
        self._all = list(gpus)
        self._free = set(gpus)
        self._policy: SchedulerPolicy = policy
        self._rr = deque(gpus)

    @property
    def all_gpus(self) -> list[int]:
        return list(self._all)

    @property
    def free_gpus(self) -> list[int]:
        return sorted(self._free)

    def allocate(self, n: int) -> list[int] | None:
        if n <= 0:
            raise ValueError(f"Requested GPU count must be >= 1, got {n}")
        if len(self._free) < n:
            return None

        if self._policy == "fill_first":
            chosen = sorted(self._free)[:n]
            for g in chosen:
                self._free.remove(g)
            return chosen

        if self._policy == "round_robin":
            chosen: list[int] = []
            # Rotate over all GPUs, picking currently-free ones.
            while len(chosen) < n:
                g = self._rr[0]
                self._rr.rotate(-1)
                if g in self._free:
                    self._free.remove(g)
                    chosen.append(g)
            return chosen

        raise ValueError(f"Unsupported policy: {self._policy!r}")

    def release(self, gpu_ids: list[int]) -> None:
        for g in gpu_ids:
            if g not in self._all:
                raise ValueError(f"Cannot release unknown GPU id {g}; allocator GPUs={self._all}")
            if g in self._free:
                raise ValueError(f"GPU {g} already free (double release?)")
            self._free.add(g)

