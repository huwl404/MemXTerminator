from __future__ import annotations


def main() -> None:
    from memxterminator.bezierfit.scheduler.gpu_allocator import GpuAllocator

    # fill_first: deterministic lowest-first allocation.
    alloc = GpuAllocator([0, 1, 2], policy="fill_first")
    assert alloc.allocate(2) == [0, 1]
    assert alloc.allocate(1) == [2]
    assert alloc.allocate(1) is None
    alloc.release([0, 1])
    assert alloc.allocate(1) == [0]

    # round_robin: rotates through GPUs (among free ones).
    alloc2 = GpuAllocator([0, 1, 2], policy="round_robin")
    assert alloc2.allocate(1) == [0]
    assert alloc2.allocate(1) == [1]
    alloc2.release([0])
    # Next pick should be 2, since RR rotated past 0 and 1 already.
    assert alloc2.allocate(1) == [2]

    print(">>> OK: GPU allocator policies.")


if __name__ == "__main__":
    main()

