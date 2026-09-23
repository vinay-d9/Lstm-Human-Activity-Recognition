"""Small checks for the sequence-shaping and preprocessing utilities."""

from __future__ import annotations

import numpy as np

from src.data import fit_scaler, transform_windows


def test_scaling_preserves_window_shape() -> None:
    windows = np.arange(3 * 128 * 9, dtype=np.float32).reshape(3, 128, 9)
    scaler = fit_scaler(windows)
    normalized = transform_windows(windows, scaler)
    assert normalized.shape == (3, 128, 9)
    assert np.allclose(normalized.mean(axis=(0, 1)), 0, atol=1e-6)
