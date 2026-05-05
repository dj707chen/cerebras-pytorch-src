"""Simple dataloader benchmark helper."""

from __future__ import annotations

import time

from .utils.dataloader import BatchMetrics, EpochMetrics, Metrics


def benchmark_dataloader(dataloader, num_steps=None, *args, **kwargs):
    metrics = Metrics(start_time_ns=time.time_ns())
    epoch = EpochMetrics(start_time_ns=metrics.start_time_ns)
    for step, batch in enumerate(dataloader):
        if num_steps is not None and step >= num_steps:
            break
        epoch.batch_metrics.append(BatchMetrics(global_step=step, epoch_step=step))
        metrics.total_steps += 1
        metrics.batch_specs = type(batch).__name__
    epoch.end_time_ns = time.time_ns()
    epoch.total_steps = metrics.total_steps
    metrics.end_time_ns = epoch.end_time_ns
    metrics.epoch_metrics.append(epoch)
    return metrics
