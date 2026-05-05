"""Benchmark metrics containers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional


@dataclass
class BatchMetrics:
    global_step: int = 0
    epoch_step: int = 0
    sampling_time_ns: int = 0
    local_rate: float = 0.0
    global_rate: float = 0.0
    profile_activities: Optional[List[Any]] = None


@dataclass
class EpochMetrics:
    start_time_ns: int = 0
    end_time_ns: int = 0
    total_steps: int = 0
    iterator_creation: int = 0
    iteration_time: int = 0
    batch_metrics: List[BatchMetrics] = field(default_factory=list)

    @property
    def total_time(self) -> int:
        return max(0, self.end_time_ns - self.start_time_ns)


@dataclass
class Metrics:
    start_time_ns: int = 0
    end_time_ns: int = 0
    dataloader_build_time: int = 0
    total_steps: int = 0
    is_partial: bool = False
    batch_specs: Any = None
    epoch_metrics: List[EpochMetrics] = field(default_factory=list)

    @property
    def total_time(self) -> int:
        return max(0, self.end_time_ns - self.start_time_ns)

    @property
    def global_rate(self) -> float:
        return self.total_steps / (self.total_time / 1e9) if self.total_time else 0.0

    @property
    def global_sample_rate(self) -> float:
        return self.global_rate
