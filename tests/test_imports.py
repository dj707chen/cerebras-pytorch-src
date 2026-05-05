def test_public_imports_without_torch():
    import cerebras.pytorch as cstorch

    assert cstorch.backend("CPU").backend_type == "CPU"
    assert cstorch.use_cs() is False
    assert cstorch.amp.set_half_dtype("bfloat16") is not None


def test_metric_registry_and_accuracy_numbers():
    from cerebras.pytorch.metrics import AccuracyMetric, get_all_metrics

    metric = AccuracyMetric("acc")
    metric.update(1, 1)
    metric.update(0, 1)
    assert metric.compute() == 0.5
    assert get_all_metrics()["acc"] is metric


def test_schedules_are_importable():
    from cerebras.pytorch.optim import ConstantLR, LinearWD
    from cerebras.pytorch.sparse import Static
    from cerebras.pytorch.sparse.utils import Linear

    assert ConstantLR(initial_val=0.1).get() == 0.1
    assert round(LinearWD(initial_val=0.1, end_val=0.0, total_iters=10).step(), 6) == 0.09
    assert Static(0.5).state_dict()["sparsity"] == 0.5
    assert Linear(init=0.0, end=1.0, steps=2).update() == 0.5
