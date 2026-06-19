"""
Tests for data encoding and the demo PNG round-trip.

All tests use synthetic tensors — LSWMD.pkl is NOT required.
"""
import numpy as np
import pandas as pd
import pytest
import torch

from wafer.data import encode_map, CLASS_NAMES, WaferDataset


def _synthetic_wafer(size: int = 20) -> np.ndarray:
    """
    Synthetic wafer map (H, W) with values {0, 1, 2}:
      outer ring = 0 (outside wafer boundary)
      interior   = 1 (passing die)
      center pixel = 2 (failing die)
    """
    m = np.ones((size, size), dtype=np.int64)
    m[0, :] = m[-1, :] = m[:, 0] = m[:, -1] = 0
    m[size // 2, size // 2] = 2
    return m


def test_encode_roundtrip():
    """encode_map → argmax along channel dim must recover the original {0,1,2} map."""
    wmap = _synthetic_wafer()
    tensor = encode_map(wmap)                        # (3, H, W) one-hot float
    assert tensor.shape == (3, wmap.shape[0], wmap.shape[1])
    assert tensor.dtype == torch.float32
    decoded = tensor.argmax(dim=0).numpy().astype(np.int64)
    np.testing.assert_array_equal(decoded, wmap)


def test_encode_clips_out_of_range():
    """Values outside [0, 2] must be clipped rather than raising."""
    wmap = np.array([[0, 1, 2, 3, -1]], dtype=np.int64)   # 3 and -1 are out of range
    tensor = encode_map(wmap)
    decoded = tensor.argmax(dim=0).numpy()
    # clip(0,2): 3→2, -1→0
    expected = np.array([[0, 1, 2, 2, 0]])
    np.testing.assert_array_equal(decoded, expected)


def test_png_roundtrip():
    """
    LUT render (wmap → gray pixels) → _png_to_tensor → argmax must recover the map.
    Verifies that the demo's reverse-LUT decode is the exact inverse of the LUT render.
    """
    from wafer.demo import _png_to_tensor
    wmap = _synthetic_wafer()
    lut = np.array([40, 160, 255], dtype=np.uint8)
    gray = lut[wmap]                                    # (H, W) uint8
    rgb = np.stack([gray, gray, gray], axis=-1)         # (H, W, 3) RGB
    tensor = _png_to_tensor(rgb, input_size=wmap.shape[0])
    decoded = tensor.argmax(dim=0).numpy().astype(np.int64)
    np.testing.assert_array_equal(decoded, wmap)


def test_augmentation_preserves_shape():
    """WaferDataset with augment=True must return (3, input_size, input_size) tensors."""
    wmap = _synthetic_wafer(30)
    df = pd.DataFrame({"waferMap": [wmap], "label": [CLASS_NAMES[0]]})
    class_to_idx = {name: i for i, name in enumerate(CLASS_NAMES)}
    ds = WaferDataset(df, class_to_idx, input_size=64, augment=True)
    tensor, label = ds[0]
    assert tensor.shape == (3, 64, 64), f"Expected (3,64,64), got {tensor.shape}"
    assert label == 0


def test_augmentation_values_remain_binary():
    """After augmentation, one-hot tensor values must still be exactly 0.0 or 1.0."""
    wmap = _synthetic_wafer(30)
    df = pd.DataFrame({"waferMap": [wmap], "label": [CLASS_NAMES[0]]})
    class_to_idx = {name: i for i, name in enumerate(CLASS_NAMES)}
    ds = WaferDataset(df, class_to_idx, input_size=32, augment=True)
    for _ in range(8):                      # check multiple random augmentations
        tensor, _ = ds[0]
        unique_vals = torch.unique(tensor).tolist()
        assert set(unique_vals).issubset({0.0, 1.0}), (
            f"Non-binary values after augmentation: {unique_vals}"
        )
