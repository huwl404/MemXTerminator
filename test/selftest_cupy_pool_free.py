from __future__ import annotations

import gc


def main() -> None:
    try:
        import cupy as cp  # noqa: F401
    except Exception as exc:
        raise SystemExit(f"CuPy import failed; run inside the mxt environment. Error: {exc}")

    pool = cp.get_default_memory_pool()

    # Start from a clean state (best-effort).
    pool.free_all_blocks()

    assert pool.total_bytes() == 0, f"Expected empty pool at start, got total_bytes={pool.total_bytes()}"

    # Allocate and then release to ensure the pool has cached blocks.
    x = cp.zeros((1024, 1024), dtype=cp.float32)
    assert pool.used_bytes() >= x.nbytes, "Expected pool.used_bytes() to account for the live allocation"
    assert pool.total_bytes() >= x.nbytes, "Expected pool.total_bytes() to include the live allocation"

    del x
    gc.collect()
    cp.cuda.Stream.null.synchronize()

    # After deletion, the allocation should be returned to the pool (used=0, total>0).
    assert pool.used_bytes() == 0, f"Expected used_bytes==0 after del, got {pool.used_bytes()}"
    assert pool.total_bytes() > 0, f"Expected cached blocks after del, got total_bytes={pool.total_bytes()}"

    pool.free_all_blocks()
    assert pool.total_bytes() == 0, f"Expected total_bytes==0 after free_all_blocks, got {pool.total_bytes()}"

    print(">>> OK: cp.get_default_memory_pool().free_all_blocks() releases cached blocks.")


if __name__ == "__main__":
    main()

