"""
Tests for ECE computation and temperature scaling.

All tests use synthetic data — no checkpoint or LSWMD.pkl required.
"""
import numpy as np
import pytest
import torch

from wafer.calibrate import compute_ece


def test_ece_perfect_calibration():
    """A model that is always correct with confidence 1.0 must have ECE = 0."""
    n = 300
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 3, n)
    probs = np.zeros((n, 3), dtype=float)
    for i, c in enumerate(labels):
        probs[i, c] = 1.0

    ece = compute_ece(probs, labels, n_bins=10)
    assert ece < 1e-6, f"Expected ECE ≈ 0 for perfect calibration, got {ece:.6f}"


def test_ece_worst_calibration():
    """A model that is always wrong with confidence 1.0 should have ECE close to 1."""
    n = 300
    labels = np.zeros(n, dtype=int)          # all true class = 0
    probs = np.zeros((n, 3), dtype=float)
    probs[:, 1] = 1.0                        # always predicts class 1 with full confidence

    ece = compute_ece(probs, labels, n_bins=10)
    # conf = 1.0, acc = 0.0 → |conf - acc| = 1.0 for every sample
    assert ece > 0.9, f"Expected ECE ≈ 1.0 for worst calibration, got {ece:.4f}"


def test_temperature_scaling_lowers_confidence():
    """Dividing logits by T > 1 must strictly reduce the max-class softmax probability."""
    logits = torch.tensor([[4.0, 1.0, -1.0]])
    T = 2.0
    probs_raw  = torch.softmax(logits,      dim=1).max().item()
    probs_temp = torch.softmax(logits / T,  dim=1).max().item()
    assert probs_temp < probs_raw, (
        f"T={T}: expected max prob to decrease, got {probs_raw:.4f} → {probs_temp:.4f}"
    )


def test_temperature_scaling_preserves_argmax():
    """Temperature scaling must not change which class has the highest probability."""
    logits = torch.tensor([[3.0, 5.0, 1.0, 2.0]])
    for T in [0.5, 1.0, 2.0, 5.0]:
        pred_raw  = logits.argmax(dim=1).item()
        pred_temp = (logits / T).argmax(dim=1).item()
        assert pred_raw == pred_temp, (
            f"T={T}: argmax changed from {pred_raw} to {pred_temp}"
        )


def test_temperature_above_one_reduces_confidence():
    """T > 1 must reduce confidence; T < 1 must increase it (squeeze or sharpen)."""
    logits = torch.tensor([[3.0, 0.5, -0.5]])
    conf_base   = torch.softmax(logits,        dim=1).max().item()
    conf_hot    = torch.softmax(logits / 0.5,  dim=1).max().item()
    conf_cold   = torch.softmax(logits / 2.0,  dim=1).max().item()
    assert conf_hot  > conf_base, "T<1 should increase max confidence"
    assert conf_cold < conf_base, "T>1 should decrease max confidence"
