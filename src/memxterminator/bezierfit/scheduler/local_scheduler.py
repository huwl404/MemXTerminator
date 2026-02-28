from __future__ import annotations

import os
import signal
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from memxterminator.mxt_state import write_json_atomic
from memxterminator.path_resolve import infer_input_base_dir

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
        "depends_on": list(job.depends_on),
        "status": "queued",
        "assigned_gpus": [],
        "pid": None,
        "returncode": None,
        "started_utc": None,
        "finished_utc": None,
        "reason": None,
        "stdout_path": stdout_path,
        "stderr_path": stderr_path,
    }


def _write_state(path: Path, state: dict[str, Any]) -> None:
    write_json_atomic(str(path), state)


def _infer_primary_input_path(kind: str, args: dict[str, Any]) -> str | None:
    """
    Best-effort pick a "primary" input file path for base-dir inference.
    """
    if kind == "bezierfit_particle_pms":
        p = args.get("particle")
        return None if p in (None, "") else str(p)
    if kind == "bezierfit_micrograph_mms":
        p = args.get("particle")
        return None if p in (None, "") else str(p)
    if kind == "bezierfit_mem_analyze":
        t = args.get("template")
        if t not in (None, ""):
            return str(t)
        p = args.get("particle")
        return None if p in (None, "") else str(p)
    return None


def _ensure_input_base_dir(kind: str, args: dict[str, Any]) -> dict[str, Any]:
    """
    Ensure `args["input_base_dir"]` is set when running under the batch scheduler.

    Rationale:
    - Scheduler runs each job with `cwd=<job_output_root>` for log isolation.
    - CryoSPARC `.cs` and STAR files often contain relative paths like `J220/extract/...`.
    - Setting input_base_dir makes job scripts cwd-independent.
    """
    existing = args.get("input_base_dir")
    if existing not in (None, ""):
        return args

    primary = _infer_primary_input_path(kind, args)
    if primary in (None, ""):
        return args

    args["input_base_dir"] = infer_input_base_dir(primary)
    return args


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

    job_by_id: dict[str, BezierfitJob] = {j.job_id: j for j in jobs}
    pending: list[BezierfitJob] = [j for j in jobs if bool(j.enabled)]
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

        args = _ensure_input_base_dir(job.kind, args)

        job_resolved = BezierfitJob(
            job_id=job.job_id,
            kind=job.kind,
            args=args,
            output_root=str(out_root),
            resources=job.resources,
            enabled=job.enabled,
            depends_on=job.depends_on,
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
                "depends_on": list(job.depends_on),
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

    def _job_status(job_id: str) -> str:
        rec = job_state_by_id.get(job_id)
        if rec is None:
            return "unknown"
        return str(rec.get("status", "unknown"))

    def _is_job_runnable(job: BezierfitJob) -> bool:
        return all(_job_status(dep_id) == "success" for dep_id in job.depends_on)

    def _has_transitive_non_success_dependency(job_id: str) -> bool:
        stack = list(job_by_id[job_id].depends_on)
        seen: set[str] = set()
        while stack:
            dep_id = stack.pop()
            if dep_id in seen:
                continue
            seen.add(dep_id)
            dep_status = _job_status(dep_id)
            if dep_status != "success":
                return True
            dep_job = job_by_id.get(dep_id)
            if dep_job is None:
                return True
            stack.extend(dep_job.depends_on)
        return False

    def _cancel_pending_jobs(*, reason_for_job: Any) -> None:
        nonlocal pending
        now = now_utc_iso()
        for job in list(pending):
            reason = str(reason_for_job(job))
            update_job_state(
                job.job_id,
                status="canceled",
                finished_utc=now,
                returncode=None,
                reason=reason,
            )
            stdout_path = str(Path(job.output_root) / "scheduler_stdout.log")
            stderr_path = str(Path(job.output_root) / "scheduler_stderr.log")
            finalize_job_result(
                job.job_id,
                JobResult(
                    job_id=job.job_id,
                    status="canceled",
                    returncode=None,
                    gpu_ids=[],
                    output_root=job.output_root,
                    pid=None,
                    started_utc=now,
                    finished_utc=now,
                    cmd_argv=build_job_argv(job),
                    stdout_path=stdout_path,
                    stderr_path=stderr_path,
                ),
            )
        pending = []

    def terminate_all(reason_status: JobStatus, *, reason: str) -> None:
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
                reason=reason,
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
            # Launch as many jobs as we can:
            # - keep pending spec order
            # - repeatedly scan left-to-right
            # - start first runnable job that fits available GPUs
            while pending and len(running) < int(spec.max_running_jobs):
                launched = False
                for idx, job in enumerate(pending):
                    if not _is_job_runnable(job):
                        continue
                    allocated = allocator.allocate(int(job.resources.gpus))
                    if allocated is None:
                        continue

                    pending.pop(idx)
                    rjob = start_job(job, allocated_gpus=list(allocated))
                    running[job.job_id] = rjob
                    update_job_state(
                        job.job_id,
                        status="running",
                        assigned_gpus=list(allocated),
                        pid=int(rjob.popen.pid),
                        started_utc=rjob.started_utc,
                        reason=None,
                    )
                    print(
                        f">>> START job_id={job.job_id} kind={job.kind} depends_on={list(job.depends_on)} "
                        f"gpus={allocated} pid={rjob.popen.pid}",
                        flush=True,
                    )
                    launched = True
                    break
                if not launched:
                    break

            # Poll running jobs for completion.
            any_failed = None
            for job_id, r in list(running.items()):
                rc = r.popen.poll()
                if rc is None:
                    continue

                status: JobStatus = "success" if int(rc) == 0 else "failed"
                finished_utc = now_utc_iso()
                allocator.release(r.gpu_ids)
                update_job_state(job_id, status=status, finished_utc=finished_utc, returncode=int(rc), reason=None)
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
                terminate_all("canceled", reason="canceled_by_fail_fast")
                _cancel_pending_jobs(
                    reason_for_job=lambda job: (
                        "upstream_failed" if _has_transitive_non_success_dependency(job.job_id) else "canceled_by_fail_fast"
                    )
                )
                break

            # Dependency deadlock / unsatisfiable pending jobs:
            # pending exists, nothing running, and none pending is runnable.
            if pending and not running:
                runnable_jobs = [job for job in pending if _is_job_runnable(job)]
                if not runnable_jobs:
                    print(
                        ">>> DEADLOCK no runnable pending jobs remain; dependencies are unsatisfied.",
                        flush=True,
                    )
                    for job in pending:
                        dep_states = [f"{dep_id}:{_job_status(dep_id)}" for dep_id in job.depends_on]
                        upstream_failed = any(_job_status(dep_id) in {"failed", "canceled"} for dep_id in job.depends_on)
                        reason = "blocked_upstream_failed" if upstream_failed else "blocked_upstream_not_finished"
                        print(
                            f">>> BLOCKED job_id={job.job_id} reason={reason} unmet_dependencies={dep_states}",
                            flush=True,
                        )
                    _cancel_pending_jobs(
                        reason_for_job=lambda job: (
                            "upstream_failed"
                            if any(_job_status(dep_id) in {"failed", "canceled"} for dep_id in job.depends_on)
                            else "deadlock_unsatisfied_dependencies"
                        )
                    )
                else:
                    # Runnable jobs may appear right after polling completes.
                    # If any runnable job can fit the configured GPU pool, allow
                    # the next scheduler iteration to launch it.
                    max_total_gpus = len(spec.gpus)
                    if any(int(job.resources.gpus) <= max_total_gpus for job in runnable_jobs):
                        time.sleep(float(poll_interval_s))
                        continue
                    print(
                        ">>> DEADLOCK runnable jobs exist but none can be allocated on configured GPUs.",
                        flush=True,
                    )
                    for job in runnable_jobs:
                        print(
                            f">>> BLOCKED job_id={job.job_id} reason=insufficient_gpus "
                            f"requested={int(job.resources.gpus)} free_now={allocator.free_gpus}",
                            flush=True,
                        )
                    _cancel_pending_jobs(reason_for_job=lambda _job: "deadlock_insufficient_gpus")
                break

            time.sleep(float(poll_interval_s))
    except KeyboardInterrupt:
        print(">>> INTERRUPT received. Canceling batch...", flush=True)
        terminate_all("canceled", reason="canceled_by_interrupt")
        _cancel_pending_jobs(reason_for_job=lambda _job: "canceled_by_interrupt")
    finally:
        state["finished_utc"] = now_utc_iso()
        _write_state(state_path_p, state)
        if old_sigterm is not None:
            try:
                signal.signal(signal.SIGTERM, old_sigterm)
            except Exception:
                pass

    return results
