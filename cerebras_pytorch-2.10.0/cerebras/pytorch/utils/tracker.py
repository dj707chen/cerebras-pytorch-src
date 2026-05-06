#
# This code is adapted from
# https://github.com/pytorch/xla/blob/8db3f891507424c91238cf5919c498e256398326/torch_xla/core/xla_model.py#L310-L351
#
# Copyright 2016-2023 Cerebras Systems
# SPDX-License-Identifier: BSD-3-Clause
#

"""Rate tracking utilities"""
import time


class RateTracker:
    """The rate tracker object"""

    def __init__(self, smooth_factor=None):
        self._smooth_factor = 0.4 if smooth_factor is None else smooth_factor
        self.reset()

    def reset(self):
        self.reset_time()
        self._partial_count = 0.0
        self._partial_rate = None
        self._count = 0.0

    def reset_time(self, offset: float = 0):
        """Reset the start time

        Args:
            offset: The number of seconds to subtract from the start time
        """
        self._start_time = time.perf_counter_ns() - offset * 1e9
        self._partial_time = self._start_time

    def _update(self, now, rate):
        self._partial_count += self._count
        self._count = 0.0
        self._partial_time = now
        self._partial_rate = rate

    def add(self, count):
        """Add number of samples processed"""
        self._count += count

    def _smooth(self, current_rate):
        if self._partial_rate is None:
            smoothed_rate = current_rate
        else:
            smoothed_rate = (
                1 - self._smooth_factor
            ) * current_rate + self._smooth_factor * self._partial_rate
        return smoothed_rate

    @property
    def total_count(self):
        """Total number of samples processed since the last reset"""
        return self._partial_count + self._count

    def rate(self):
        """Smoothed samples/second of all the samples added since last queried"""
        now = time.perf_counter_ns()
        delta = (now - self._partial_time) / 1e9
        report_rate = 0.0
        if delta > 0:
            report_rate = self._smooth(self._count / delta)
            self._update(now, report_rate)
        return report_rate

    def global_rate(self):
        """
        Non-smoothed samples/second since the beginning of when the rate tracker
        as initialized
        """
        delta = self.elapsed_seconds()
        return (self.total_count / delta) if delta > 0 else 0.0

    def elapsed_seconds(self):
        """Time (seconds) elapsed since the last reset"""
        return (time.perf_counter_ns() - self._start_time) / 1e9
