"""Load and normalize sequential signals from the UCI HAR Dataset."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler


CLASS_NAMES = [
    "WALKING",
    "WALKING_UPSTAIRS",
    "WALKING_DOWNSTAIRS",
    "SITTING",
    "STANDING",
    "LAYING",
]

# This fixed order defines the nine feature columns passed to the LSTM and app.
SENSOR_FILES = [
    "body_acc_x",
    "body_acc_y",
    "body_acc_z",
    "body_gyro_x",
    "body_gyro_y",
    "body_gyro_z",
    "total_acc_x",
    "total_acc_y",
    "total_acc_z",
]


def load_split(dataset_dir: str | Path, split: str) -> tuple[np.ndarray, np.ndarray]:
    """Return raw UCI HAR windows and zero-based activity labels for one split.

    The returned signal tensor has shape ``(samples, 128, 9)``. Each 128-row
    window is a 2.56-second recording at 50 Hz.
    """
    if split not in {"train", "test"}:
        raise ValueError("split must be 'train' or 'test'")

    root = Path(dataset_dir).expanduser()
    inertial_dir = root / split / "Inertial Signals"
    labels_path = root / split / f"y_{split}.txt"
    if not inertial_dir.is_dir() or not labels_path.is_file():
        raise FileNotFoundError(
            f"Could not find the UCI HAR {split!r} split under {root}. "
            "Pass the directory named 'UCI HAR Dataset', not its parent."
        )

    channels: list[np.ndarray] = []
    for sensor_name in SENSOR_FILES:
        signal_path = inertial_dir / f"{sensor_name}_{split}.txt"
        if not signal_path.is_file():
            raise FileNotFoundError(f"Missing sensor file: {signal_path}")
        channel = np.loadtxt(signal_path, dtype=np.float32)
        if channel.ndim != 2 or channel.shape[1] != 128:
            raise ValueError(
                f"Expected 128 readings per row in {signal_path.name}; "
                f"got shape {channel.shape}."
            )
        channels.append(channel)

    sample_counts = {channel.shape[0] for channel in channels}
    if len(sample_counts) != 1:
        raise ValueError("Sensor files do not contain the same number of windows.")

    # (samples, 128, 9): LSTM time axis is second, sensor-channel axis last.
    features = np.stack(channels, axis=-1)
    labels = np.loadtxt(labels_path, dtype=np.int64).reshape(-1) - 1
    if len(labels) != features.shape[0]:
        raise ValueError("The number of activity labels does not match the signal windows.")
    if not np.isin(labels, np.arange(len(CLASS_NAMES))).all():
        raise ValueError("Encountered an activity label outside the expected range 1-6.")
    return features, labels


def fit_scaler(windows: np.ndarray) -> StandardScaler:
    """Fit one per-channel scaler on every time step of training windows only."""
    _validate_windows(windows)
    return StandardScaler().fit(windows.reshape(-1, windows.shape[-1]))


def transform_windows(windows: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    """Apply a fitted per-channel scaler while preserving sequence structure."""
    _validate_windows(windows)
    original_shape = windows.shape
    transformed = scaler.transform(windows.reshape(-1, original_shape[-1]))
    return transformed.reshape(original_shape).astype(np.float32)


def _validate_windows(windows: np.ndarray) -> None:
    if windows.ndim != 3 or windows.shape[1:] != (128, len(SENSOR_FILES)):
        raise ValueError(
            "Expected sensor windows shaped (samples, 128, 9); "
            f"got {windows.shape}."
        )
