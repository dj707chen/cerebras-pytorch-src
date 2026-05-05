"""Metric classes matching the documented Cerebras PyTorch surface."""

from __future__ import annotations

import math
from typing import Dict, Optional

from ._compat import optional_torch


class Metric:
    registry: Dict[str, "Metric"] = {}

    def __init__(self, name: Optional[str] = None):
        self.name = name or type(self).__name__
        self._num_updates = 0
        self._states = {}
        Metric.registry[self.name] = self

    @property
    def num_updates(self) -> int:
        return self._num_updates

    def reset(self):
        raise NotImplementedError

    def update(self, *args, **kwargs):
        raise NotImplementedError

    def compute(self):
        raise NotImplementedError

    def register_state(self, name, tensor, persistent=False):
        setattr(self, name, tensor)
        self._states[name] = {"value": tensor, "persistent": persistent}

    def forward(self, *args, **kwargs):
        self.update(*args, **kwargs)
        self._num_updates += 1
        return self.compute()

    __call__ = forward


def get_all_metrics():
    return dict(Metric.registry)


def _as_float(value):
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "item") and getattr(value, "numel", lambda: 1)() == 1:
        return float(value.item())
    return float(value)


class AccuracyMetric(Metric):
    def __init__(self, name: Optional[str] = None):
        super().__init__(name)
        self.reset()

    def reset(self):
        self.correct = 0.0
        self.total = 0.0
        self._num_updates = 0

    def update(self, labels, predictions, weights=None, dtype=None):
        torch = optional_torch()
        if torch is not None and hasattr(predictions, "shape"):
            preds = predictions.argmax(dim=-1) if predictions.ndim > labels.ndim else predictions
            matches = (preds == labels).to(dtype or torch.float32)
            if weights is not None:
                matches = matches * weights
                self.total += _as_float(weights.sum())
            else:
                self.total += float(matches.numel())
            self.correct += _as_float(matches.sum())
        else:
            self.correct += float(predictions == labels)
            self.total += 1.0
        self._num_updates += 1

    def compute(self):
        return self.correct / self.total if self.total else 0.0


class PerplexityMetric(Metric):
    def __init__(self, name: Optional[str] = None):
        super().__init__(name)
        self.reset()

    def reset(self):
        self.total_loss = 0.0
        self.total_weight = 0.0
        self._num_updates = 0

    def update(self, labels, loss, weights=None, dtype=None):
        torch = optional_torch()
        if weights is not None and torch is not None and hasattr(weights, "sum"):
            weight = _as_float(weights.sum())
        elif torch is not None and hasattr(labels, "numel"):
            weight = float(labels.numel())
        else:
            weight = 1.0
        self.total_loss += _as_float(loss) * weight
        self.total_weight += weight
        self._num_updates += 1

    def compute(self):
        return math.exp(self.total_loss / self.total_weight) if self.total_weight else 0.0


class _ConfusionMetric(Metric):
    def __init__(self, num_classes: int, name: Optional[str] = None):
        super().__init__(name)
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        self.tp = [0.0] * self.num_classes
        self.fp = [0.0] * self.num_classes
        self.fn = [0.0] * self.num_classes
        self._num_updates = 0

    def update(self, labels, predictions, weights=None, dtype=None):
        torch = optional_torch()
        if torch is not None and hasattr(predictions, "detach"):
            preds = predictions.argmax(dim=-1) if predictions.ndim > labels.ndim else predictions
            pairs = zip(labels.detach().cpu().reshape(-1).tolist(), preds.detach().cpu().reshape(-1).tolist())
        else:
            pairs = zip(labels, predictions)
        for label, pred in pairs:
            label = int(label)
            pred = int(pred)
            if label == pred:
                self.tp[label] += 1
            else:
                self.fn[label] += 1
                self.fp[pred] += 1
        self._num_updates += 1


class DiceCoefficientMetric(_ConfusionMetric):
    def compute(self):
        vals = []
        for tp, fp, fn in zip(self.tp, self.fp, self.fn):
            denom = 2 * tp + fp + fn
            vals.append((2 * tp / denom) if denom else 0.0)
        return sum(vals) / len(vals) if vals else 0.0


class MeanIOUMetric(_ConfusionMetric):
    def compute(self):
        vals = []
        for tp, fp, fn in zip(self.tp, self.fp, self.fn):
            denom = tp + fp + fn
            vals.append((tp / denom) if denom else 0.0)
        return sum(vals) / len(vals) if vals else 0.0


class FBetaScoreMetric(_ConfusionMetric):
    def __init__(self, num_classes: int, beta: float = 1.0, average: str = "macro", ignore_labels=None, name: Optional[str] = None):
        self.beta = beta
        self.average = average
        self.ignore_labels = set(ignore_labels or [])
        super().__init__(num_classes, name)

    def update(self, labels, predictions, dtype=None):
        return super().update(labels, predictions, dtype=dtype)

    def compute(self):
        beta2 = self.beta * self.beta
        vals = []
        for idx, (tp, fp, fn) in enumerate(zip(self.tp, self.fp, self.fn)):
            if idx in self.ignore_labels:
                continue
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            denom = beta2 * precision + recall
            vals.append(((1 + beta2) * precision * recall / denom) if denom else 0.0)
        return sum(vals) / len(vals) if vals else 0.0
