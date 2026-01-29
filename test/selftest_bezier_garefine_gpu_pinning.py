from __future__ import annotations

import multiprocessing
import os
import time


def _worker_get_device(_dummy: int) -> tuple[int, int]:
    import cupy as cp

    ident = getattr(multiprocessing.current_process(), "_identity", ())
    worker_rank = int(ident[0]) if ident else 1
    # Small sleep to ensure multiple workers have time to start and pick up work.
    time.sleep(0.05)
    return int(os.getpid()), worker_rank, int(cp.cuda.Device().id)


def main() -> None:
    try:
        import cupy as cp  # noqa: F401
    except Exception as exc:
        raise SystemExit(f"CuPy import failed; run inside the mxt environment. Error: {exc}")

    device_count = int(cp.cuda.runtime.getDeviceCount())
    if device_count <= 0:
        print(">>> SKIP: No CUDA devices visible.")
        return

    # Use spawn to avoid CUDA + fork hazards.
    multiprocessing.set_start_method("spawn", force=True)

    from memxterminator.bezierfit.lib.bezierfit import init_cuda  # noqa: WPS433 (test-only import)

    procs = 4
    with multiprocessing.Pool(processes=procs, initializer=init_cuda) as pool:
        # Use more tasks than workers to reduce the chance that only one worker
        # processes all tasks due to spawn startup latency.
        results = pool.map(_worker_get_device, range(procs * 4), chunksize=1)

    pids = {pid for pid, _rank, _dev in results}
    rank_to_device: dict[int, int] = {}
    for _pid, worker_rank, device_id in results:
        rank_to_device.setdefault(worker_rank, device_id)

    if procs > 1 and len(rank_to_device) < 2:
        raise AssertionError(
            f"Expected >=2 distinct worker ranks, got {len(rank_to_device)}. "
            f"(spawn startup latency?) pids={sorted(pids)} rank_to_device={rank_to_device}"
        )

    for worker_rank, device_id in sorted(rank_to_device.items()):
        expected = (worker_rank - 1) % device_count
        if device_count >= 2:
            assert device_id == expected, (
                f"Worker {worker_rank} pinned to device {device_id}, expected {expected} "
                f"(device_count={device_count})"
            )
        else:
            # Single-GPU fallback: all workers should report device 0.
            assert device_id == 0, f"Expected device 0 with device_count=1, got {device_id}"

    if device_count >= 2:
        used_devices = sorted(set(rank_to_device.values()))
        if len(used_devices) < 2:
            raise AssertionError(f"Expected >=2 GPUs used, got devices={used_devices} rank_to_device={rank_to_device}")

    print(
        f">>> OK: GA_Refine workers distributed across {device_count} device(s): "
        f"pids={sorted(pids)} rank_to_device={rank_to_device}"
    )


if __name__ == "__main__":
    main()
