"""
Tests for model construction, forward pass shape, and TTA correctness.

All tests run on CPU with randomly-initialised weights — no checkpoint required.
"""
import os

import numpy as np
import pytest
import torch

os.environ["WAFER_DEVICE"] = "cpu"   # force CPU before any wafer imports

from wafer.config import WaferConfig
from wafer.model import build_model
from wafer.evaluate import tta_predict


def _cpu_cfg(arch: str = "resnet18") -> WaferConfig:
    cfg = WaferConfig(device="cpu", arch=arch)
    return cfg


class TestBuildModel:
    def test_resnet18_output_shape(self):
        """ResNet-18 must produce (B, 9) logits for a (B, 3, 224, 224) input."""
        cfg = _cpu_cfg("resnet18")
        model = build_model(cfg, num_classes=9)
        x = torch.zeros(2, 3, 224, 224)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, 9), f"Expected (2,9), got {out.shape}"

    def test_resnet50_output_shape(self):
        """ResNet-50 variant must also produce (B, 9) logits."""
        cfg = _cpu_cfg("resnet50")
        model = build_model(cfg, num_classes=9)
        x = torch.zeros(1, 3, 224, 224)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 9), f"Expected (1,9), got {out.shape}"

    def test_custom_num_classes(self):
        """num_classes parameter must control output dimension."""
        cfg = _cpu_cfg()
        model = build_model(cfg, num_classes=5)
        x = torch.zeros(1, 3, 224, 224)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 5)

    def test_accepts_small_input(self):
        """Model should handle smaller spatial inputs without crashing."""
        cfg = _cpu_cfg()
        model = build_model(cfg, num_classes=9)
        x = torch.zeros(1, 3, 64, 64)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 9)


class TestTTA:
    def test_tta_output_shape(self):
        """tta_predict must return (B, num_classes) probability array."""
        cfg = _cpu_cfg()
        model = build_model(cfg, num_classes=9)
        x = torch.zeros(3, 3, 224, 224)
        probs = tta_predict(model, x, "cpu", temperature=1.0)
        assert probs.shape == (3, 9), f"Expected (3,9), got {probs.shape}"

    def test_tta_probabilities_sum_to_one(self):
        """TTA-averaged softmax probabilities must sum to 1 per sample."""
        cfg = _cpu_cfg()
        model = build_model(cfg, num_classes=9)
        x = torch.zeros(4, 3, 224, 224)
        probs = tta_predict(model, x, "cpu", temperature=1.0)
        row_sums = probs.sum(axis=1)
        np.testing.assert_allclose(row_sums, np.ones(4), atol=1e-5)

    def test_tta_d4_invariance_on_uniform_input(self):
        """
        For a rotationally symmetric input (all channels uniform), every element
        of the D4 group produces the same tensor, so TTA and single-pass probabilities
        must be identical.
        """
        cfg = _cpu_cfg()
        model = build_model(cfg, num_classes=9)
        model.eval()

        # Constant "all-passing" map: one-hot channel 1 = 1.0 everywhere
        x = torch.zeros(1, 3, 224, 224)
        x[:, 1, :, :] = 1.0

        probs_tta = tta_predict(model, x, "cpu", temperature=1.0)
        with torch.no_grad():
            logits = model(x)
        probs_single = torch.softmax(logits, dim=1).cpu().numpy()

        np.testing.assert_allclose(probs_tta, probs_single, atol=1e-5)

    def test_tta_temperature_divides_logits(self):
        """Higher temperature must produce softer (lower max-confidence) TTA output."""
        cfg = _cpu_cfg()
        model = build_model(cfg, num_classes=9)
        x = torch.zeros(1, 3, 224, 224)

        probs_T1 = tta_predict(model, x, "cpu", temperature=1.0)
        probs_T5 = tta_predict(model, x, "cpu", temperature=5.0)

        max_T1 = probs_T1.max()
        max_T5 = probs_T5.max()
        assert max_T5 < max_T1, (
            f"Higher temperature should lower max confidence: {max_T1:.4f} vs {max_T5:.4f}"
        )
