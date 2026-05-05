"""Small distributed API placeholders for non-CSX execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class ClusterConfig:
    mgmt_address: Optional[str] = None
    credentials_path: Optional[str] = None
    num_csx: int = 1
    max_wgt_servers: int = 24
    max_act_per_csx: int = 1
    num_workers_per_csx: int = 1
    job_time_sec: Optional[int] = None
    mount_dirs: Optional[List[str]] = None
    python_paths: Optional[List[str]] = None
    job_labels: Optional[List[str]] = None


@dataclass
class WorkerState:
    worker_step: int = 0
    global_worker_id: int = 0


def get_worker_state():
    """Return None outside Cerebras worker state checkpoint callbacks."""

    return None
