from __future__ import annotations

"""
Bezierfit batch/job scheduler (single-node).

This package provides a small local scheduler that runs multiple Bezierfit jobs
in parallel as separate subprocesses, with per-job GPU visibility controlled via
CUDA_VISIBLE_DEVICES.

The scheduler is intentionally lightweight:
- No external dependencies beyond the MemXTerminator runtime requirements
- Fail-fast by default (stop the batch on first job failure)
"""

from .spec import BezierfitJob, JobResources, JobResult, JobSpecFile, JobStatus, SchedulerSpec

__all__ = [
    "BezierfitJob",
    "JobResources",
    "JobResult",
    "JobSpecFile",
    "JobStatus",
    "SchedulerSpec",
]

