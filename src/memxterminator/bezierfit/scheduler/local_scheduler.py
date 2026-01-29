from __future__ import annotations

import os
import signal
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from memxterminator.mxt_state import write_json_atomic

from .gpu_allocator import GpuAllocator
from .spec import BezierfitJob, JobResult, JobStatus, JobSpecFile, SchedulerSpec, build_job_argv, now_utc_iso


@dataclass
class _RunningJob:
    job: BezierfitJob
    popen: subprocess.Popen
    gpu_ids: list[int]
    stdout_fh: Any
    stderr_fh: Any
    stdout_path: str
    stderr_path: str
    started_utc: str


def _compute_repo_src_for_pythonpath() -> str | None:
    """
    Best-effort detect a repo checkout `src/` directory.

    - In a development checkout, this finds `<repo>/src`.
    - In an installed package, returns None (use installed import resolution).
    """
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "src" / "memxterminator"
        if candidate.is_dir():
            return str(parent / "src")
    return None


def _popen_kwargs_for_new_session() -> dict:
    if os.name == "posix":
        return {"start_new_session": True}
    return {}


def _terminate_pid(pid: int, *, sig: int) -> None:
    if pid <= 0:
        return
    try:
        if os.name == "posix":
            try:
                pgid = os.getpgid(pid)
                if pgid == pid:
                    os.killpg(pgid, sig)
                    return
            except Exception:
                pass
        os.kill(pid, sig)
    except OSError:
        return


def _ensure_job_output_root(job: BezierfitJob) -> None:
    Path(job.output_root).mkdir(parents=True, exist_ok=True)


def _job_state_record(job: BezierfitJob) -> dict[str, Any]:
    out_root = str(job.output_root)
    stdout_path = str(Path(out_root) / "scheduler_stdout.log")
    stderr_path = str(Path(out_root) / "scheduler_stderr.log")
    return {
        "job_id": job.job_id,
        "kind": job.kind,
        "enabled": bool(job.enabled),
        "output_root": out_root,
        "resources": asdict(job.resources),
        "status": "queued",
        "assigned_gpus": [],
        "pid": None,
        "returncode": None,
        "started_utc": None,
        "finished_utc": None,
        "stdout_path": stdout_path,
        "stderr_path": stderr_path,
    }


def _write_state(path: Path, state: dict[str, Any]) -> None:
    write_json_atomic(str(path), state)


def run_jobs(
    *,
    spec: SchedulerSpec,
    jobs: list[BezierfitJob],
    state_path: os.PathLike[str] | str,
    poll_interval_s: float = 0.2,
    terminate_timeout_s: float = 30.0,
) -> dict[str, JobResult]:
    """
    Run a batch of jobs with GPU scheduling and fail-fast semantics.
    """
    state_path_p = Path(os.fspath(state_path))
    state_path_p.parent.mkdir(parents=True, exist_ok=True)

    old_sigterm = None
    try:
        old_sigterm = signal.getsignal(signal.SIGTERM)

        def _handle_sigterm(_signum: int, _frame: object) -> None:  # pragma: no cover (signal delivery)
            # Reuse the same cancellation path as Ctrl+C.
            raise KeyboardInterrupt

        signal.signal(signal.SIGTERM, _handle_sigterm)
    except Exception:
        old_sigterm = None

    allocator = GpuAllocator(list(spec.gpus), policy=spec.policy)

    pending = [j for j in jobs if bool(j.enabled)]
    results: dict[str, JobResult] = {}
    running: dict[str, _RunningJob] = {}

    state: dict[str, Any] = {
        "schema_version": 1,
        "started_utc": now_utc_iso(),
        "finished_utc": None,
        "scheduler": {
            "gpus": list(spec.gpus),
            "policy": spec.policy,
            "max_running_jobs": int(spec.max_running_jobs),
            "fail_fast": True,
        },
        "jobs": [_job_state_record(j) for j in jobs],
    }
    job_state_by_id: dict[str, dict[str, Any]] = {rec["job_id"]: rec for rec in state["jobs"]}
    for j in jobs:
        if not bool(j.enabled):
            job_state_by_id[j.job_id].update(
                {
                    "status": "canceled",
                    "finished_utc": now_utc_iso(),
                    "reason": "disabled",
                }
            )
    _write_state(state_path_p, state)

    repo_src = _compute_repo_src_for_pythonpath()

    def start_job(job: BezierfitJob, *, allocated_gpus: list[int]) -> _RunningJob:
        _ensure_job_output_root(job)
        out_root = Path(job.output_root)
        stdout_path = str(out_root / "scheduler_stdout.log")
        stderr_path = str(out_root / "scheduler_stderr.log")

        stdout_fh = open(stdout_path, "w", encoding="utf-8")
        stderr_fh = open(stderr_path, "w", encoding="utf-8")

        # Ensure procs defaults are explicit for GPU-aware tools.
        args = dict(job.args)
        if job.kind in {"bezierfit_particle_pms", "bezierfit_micrograph_mms"} and "procs" not in args:
            args["procs"] = job.resources.procs if job.resources.procs is not None else int(job.resources.gpus)

        job_resolved = BezierfitJob(
            job_id=job.job_id,
            kind=job.kind,
            args=args,
            output_root=str(out_root),
            resources=job.resources,
            enabled=job.enabled,
        )
        argv = build_job_argv(job_resolved)

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(x) for x in allocated_gpus)
        if repo_src:
            existing = env.get("PYTHONPATH", "")
            env["PYTHONPATH"] = repo_src + (os.pathsep + existing if existing else "")

        # Write resolved spec for reproducibility/debugging.
        write_json_atomic(
            str(out_root / "job_spec_resolved.json"),
            {
                "job_id": job.job_id,
                "kind": job.kind,
                "output_root": str(out_root),
                "resources": asdict(job.resources),
                "assigned_gpus": allocated_gpus,
                "argv": argv,
                "args": args,
            },
        )

        started_utc = now_utc_iso()
        popen = subprocess.Popen(
            argv,
            cwd=str(out_root),
            env=env,
            stdout=stdout_fh,
            stderr=stderr_fh,
            **_popen_kwargs_for_new_session(),
        )

        return _RunningJob(
            job=job_resolved,
            popen=popen,
            gpu_ids=list(allocated_gpus),
            stdout_fh=stdout_fh,
            stderr_fh=stderr_fh,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            started_utc=started_utc,
        )

    def update_job_state(job_id: str, **updates: Any) -> None:
        rec = job_state_by_id[job_id]
        rec.update(updates)
        # Add counts to help UI.
        counts = {"queued": 0, "running": 0, "success": 0, "failed": 0, "canceled": 0}
        for r in state["jobs"]:
            counts[str(r["status"])] += 1
        state["counts"] = counts
        state["free_gpus"] = allocator.free_gpus
        _write_state(state_path_p, state)

    def finalize_job_result(job_id: str, result: JobResult) -> None:
        results[job_id] = result
        try:
            write_json_atomic(str(Path(result.output_root) / "job_result.json"), asdict(result))
        except Exception:
            pass

    def terminate_all(reason_status: JobStatus) -> None:
        # Send SIGTERM to all process groups first.
        for r in list(running.values()):
            pid = int(r.popen.pid)
            _terminate_pid(pid, sig=signal.SIGTERM)

        deadline = time.time() + float(terminate_timeout_s)
        while time.time() < deadline:
            alive = [r for r in running.values() if r.popen.poll() is None]
            if not alive:
                break
            time.sleep(0.2)

        # Escalate to SIGKILL for any remaining.
        for r in list(running.values()):
            if r.popen.poll() is None:
                pid = int(r.popen.pid)
                if os.name == "posix":
                    _terminate_pid(pid, sig=signal.SIGKILL)
                else:
                    try:
                        r.popen.kill()
                    except Exception:
                        pass

        # Collect and record canceled results.
        for job_id, r in list(running.items()):
            allocator.release(r.gpu_ids)
            update_job_state(
                job_id,
                status=reason_status,
                finished_utc=now_utc_iso(),
                returncode=r.popen.poll(),
            )
            try:
                r.stdout_fh.close()
            except Exception:
                pass
            try:
                r.stderr_fh.close()
            except Exception:
                pass
            finalize_job_result(
                job_id,
                JobResult(
                    job_id=job_id,
                    status=reason_status,
                    returncode=r.popen.poll(),
                    gpu_ids=list(r.gpu_ids),
                    output_root=r.job.output_root,
                    pid=int(r.popen.pid) if r.popen.pid else None,
                    started_utc=r.started_utc,
                    finished_utc=now_utc_iso(),
                    cmd_argv=build_job_argv(r.job),
                    stdout_path=r.stdout_path,
                    stderr_path=r.stderr_path,
                ),
            )
            running.pop(job_id, None)

    try:
        while pending or running:
            # Launch as many jobs as we can (subject to GPU + max-running constraints).
            while pending and len(running) < int(spec.max_running_jobs):
                job = pending[0]
                allocated = allocator.allocate(int(job.resources.gpus))
                if allocated is None:
                    break
                pending.pop(0)

                rjob = start_job(job, allocated_gpus=list(allocated))
                running[job.job_id] = rjob
                update_job_state(
                    job.job_id,
                    status="running",
                    assigned_gpus=list(allocated),
                    pid=int(rjob.popen.pid),
                    started_utc=rjob.started_utc,
                )
                print(f">>> START job_id={job.job_id} kind={job.kind} gpus={allocated} pid={rjob.popen.pid}", flush=True)

            # Poll running jobs for completion.
            any_failed = None
            for job_id, r in list(running.items()):
                rc = r.popen.poll()
                if rc is None:
                    continue

                status: JobStatus = "success" if int(rc) == 0 else "failed"
                finished_utc = now_utc_iso()
                allocator.release(r.gpu_ids)
                update_job_state(job_id, status=status, finished_utc=finished_utc, returncode=int(rc))
                print(f">>> DONE job_id={job_id} status={status} rc={rc}", flush=True)

                try:
                    r.stdout_fh.close()
                except Exception:
                    pass
                try:
                    r.stderr_fh.close()
                except Exception:
                    pass

                finalize_job_result(
                    job_id,
                    JobResult(
                        job_id=job_id,
                        status=status,
                        returncode=int(rc),
                        gpu_ids=list(r.gpu_ids),
                        output_root=r.job.output_root,
                        pid=int(r.popen.pid) if r.popen.pid else None,
                        started_utc=r.started_utc,
                        finished_utc=finished_utc,
                        cmd_argv=build_job_argv(r.job),
                        stdout_path=r.stdout_path,
                        stderr_path=r.stderr_path,
                    ),
                )

                running.pop(job_id, None)
                if status == "failed":
                    any_failed = job_id
                    break

            if any_failed and spec.fail_fast:
                print(f">>> FAIL_FAST triggered by job_id={any_failed}. Terminating remaining jobs...", flush=True)
                terminate_all("canceled")
                break

            time.sleep(float(poll_interval_s))
    except KeyboardInterrupt:
        print(">>> INTERRUPT received. Canceling batch...", flush=True)
        terminate_all("canceled")
    finally:
        state["finished_utc"] = now_utc_iso()
        _write_state(state_path_p, state)
        if old_sigterm is not None:
            try:
                signal.signal(signal.SIGTERM, old_sigterm)
            except Exception:
                pass

    return results
