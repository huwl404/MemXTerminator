from __future__ import annotations

import os
import signal
import sys
from pathlib import Path


def python_executable_for_subprocess() -> str:
    """
    Pick the interpreter GUI-launched subprocesses should use.

    HPC environment modules can prepend CUDA's `bin/` ahead of conda, making
    `sys.executable` point at `/software/cuda/.../bin/python`. Prefer the active
    conda env interpreter when available so `python -m memxterminator...` can
    import the package installed in that env.
    """
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        exe_name = "python.exe" if os.name == "nt" else "python"
        candidate = Path(conda_prefix) / ("Scripts" if os.name == "nt" else "bin") / exe_name
        if candidate.exists():
            return str(candidate)
    return sys.executable


def popen_kwargs_for_new_session() -> dict:
    """
    Return kwargs for subprocess.Popen to start a new session when possible.

    On POSIX, `start_new_session=True` makes the child process the leader of a
    new session and process group, which allows us to terminate the entire group
    (parent + multiprocessing workers) safely.
    """
    if os.name == "posix":
        return {"start_new_session": True}
    return {}


def terminate_pid(pid: int, *, sig: int = signal.SIGTERM) -> None:
    """
    Best-effort terminate a subprocess (and its worker children when possible).

    - If the process is the leader of its own process group (pgid == pid),
      terminate the whole group.
    - Otherwise, fall back to terminating only the PID.
    """
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

