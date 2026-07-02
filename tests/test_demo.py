"""Tests for demo input validation — no GPU or LSWMD.pkl required."""
import numpy as np

from wafer.demo import _lut_mismatch_fraction, _png_to_tensor


def _lut_image(h: int = 64, w: int = 64) -> np.ndarray:
    """RGB image drawn from the project LUT {40, 160, 255}."""
    rng = np.random.default_rng(0)
    gray = rng.choice([40, 160, 255], size=(h, w)).astype(np.uint8)
    return np.stack([gray] * 3, axis=-1)


def test_lut_image_scores_zero_mismatch():
    assert _lut_mismatch_fraction(_lut_image()) == 0.0


def test_arbitrary_photo_scores_high_mismatch():
    rng = np.random.default_rng(1)
    photo = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
    # Uniform gray values: only ~3 tol-25 bands around {40,160,255} land "near"
    assert _lut_mismatch_fraction(photo) > 0.3


def test_grayscale_input_supported():
    gray = np.full((32, 32), 160, dtype=np.uint8)
    assert _lut_mismatch_fraction(gray) == 0.0


def test_lut_image_decodes_to_valid_one_hot():
    tensor = _png_to_tensor(_lut_image(), input_size=64)
    assert tensor.shape == (3, 64, 64)
    assert np.allclose(tensor.sum(dim=0).numpy(), 1.0)
