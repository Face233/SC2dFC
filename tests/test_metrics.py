import numpy as np

from scdfc.evaluation import dynamic_state_metrics, retrieval_metrics, sequence_metrics, subject_bootstrap_difference


def test_metrics_are_best_for_exact_prediction():
    rng = np.random.default_rng(5)
    target = rng.normal(size=(20, 15))
    template = np.zeros_like(target)
    metrics = sequence_metrics(target, target, template, 5)
    assert metrics["mse"] == 0
    assert metrics["long_residual_pearson"] > 0.999
    assert metrics["fcd_pearson"] > 0.999


def test_retrieval_and_subject_bootstrap():
    rng = np.random.default_rng(6)
    targets = rng.normal(size=(8, 20, 15))
    result = retrieval_metrics(targets, targets, np.zeros((20, 15)), 5)
    assert result["retrieval_top1"] == 1
    boot = subject_bootstrap_difference(np.ones(8), np.zeros(8), [str(i // 2) for i in range(8)], replicates=100, seed=1)
    assert boot["passes"]


def test_retrieval_accepts_other_run_from_same_subject():
    rng = np.random.default_rng(12)
    targets = rng.normal(size=(4, 10, 6))
    predictions = targets[[1, 0, 3, 2]]
    subjects = ["A", "A", "B", "B"]
    result = retrieval_metrics(predictions, targets, np.zeros((10, 6)), 2, subjects)
    assert result["retrieval_top1"] == 1


def test_state_metrics_exact():
    labels = np.array([0, 0, 1, 1, 0, 2, 2])
    metrics = dynamic_state_metrics(labels, labels, 3)
    assert all(value == 0 for value in metrics.values())
