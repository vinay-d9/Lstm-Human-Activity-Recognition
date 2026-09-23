"""Train, evaluate, and save a compact LSTM model for UCI HAR signals."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from src.data import CLASS_NAMES, SENSOR_FILES, fit_scaler, load_split, transform_windows


def build_model(input_shape: tuple[int, int], num_classes: int) -> tf.keras.Model:
    """Build a deliberately small LSTM classifier suitable for a laptop CPU."""
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=input_shape),
            tf.keras.layers.LSTM(64, dropout=0.20),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dropout(0.20),
            tf.keras.layers.Dense(num_classes, activation="softmax"),
        ],
        name="har_lstm",
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("data/UCI HAR Dataset"),
        help="Directory named 'UCI HAR Dataset'.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts"),
        help="Directory for the saved model, scaler, and evaluation artifacts.",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--validation-size", type=float, default=0.20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.patience < 1:
        raise ValueError("epochs, batch-size, and patience must each be at least 1.")
    if not 0 < args.validation_size < 1:
        raise ValueError("validation-size must be between 0 and 1.")

    set_seed(args.seed)
    output_dir = args.output_dir.expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading raw 128-step inertial-signal windows...")
    x_train_raw, y_train = load_split(args.dataset_dir, "train")
    x_test_raw, y_test = load_split(args.dataset_dir, "test")
    train_indices, validation_indices = train_test_split(
        np.arange(len(y_train)),
        test_size=args.validation_size,
        random_state=args.seed,
        stratify=y_train,
    )

    # Fit only on the fitting partition to keep validation and test evaluation honest.
    scaler = fit_scaler(x_train_raw[train_indices])
    x_train = transform_windows(x_train_raw[train_indices], scaler)
    x_validation = transform_windows(x_train_raw[validation_indices], scaler)
    x_test = transform_windows(x_test_raw, scaler)
    y_fit = y_train[train_indices]
    y_validation = y_train[validation_indices]

    model = build_model(input_shape=x_train.shape[1:], num_classes=len(CLASS_NAMES))
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=args.patience, mode="max", restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            output_dir / "best_model.keras", monitor="val_accuracy", mode="max", save_best_only=True
        ),
    ]
    history = model.fit(
        x_train,
        y_fit,
        validation_data=(x_validation, y_validation),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=2,
    )

    # Save the restored best epoch under the stable filename consumed by Streamlit.
    model.save(output_dir / "har_lstm.keras")
    joblib.dump(scaler, output_dir / "scaler.joblib")
    (output_dir / "class_names.json").write_text(json.dumps(CLASS_NAMES, indent=2) + "\n")
    (output_dir / "sensor_channels.json").write_text(json.dumps(SENSOR_FILES, indent=2) + "\n")

    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    probabilities = model.predict(x_test, verbose=0)
    predicted_labels = probabilities.argmax(axis=1)
    metrics = {
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),
        "test_samples": int(len(y_test)),
        "best_validation_accuracy": float(max(history.history["val_accuracy"])),
        "epochs_completed": int(len(history.history["loss"])),
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    (output_dir / "classification_report.txt").write_text(
        classification_report(y_test, predicted_labels, target_names=CLASS_NAMES, digits=4, zero_division=0)
    )
    save_confusion_matrix(y_test, predicted_labels, output_dir / "confusion_matrix.png")
    save_training_curves(history.history, output_dir / "training_curves.png")

    # The app can use this raw (unscaled) held-out example immediately after training.
    np.save(output_dir / "demo_window.npy", x_test_raw[0])
    (output_dir / "demo_window_label.json").write_text(
        json.dumps({"true_label": CLASS_NAMES[int(y_test[0])]}, indent=2) + "\n"
    )

    print(f"\nTest accuracy: {test_accuracy:.2%}")
    print(f"Saved model and artifacts to: {output_dir.resolve()}")


def save_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, destination: Path) -> None:
    matrix = confusion_matrix(y_true, y_pred, labels=np.arange(len(CLASS_NAMES)))
    figure, axis = plt.subplots(figsize=(8, 6))
    image = axis.imshow(matrix, cmap="Blues")
    figure.colorbar(image, ax=axis, label="Windows")
    axis.set(
        xticks=np.arange(len(CLASS_NAMES)),
        yticks=np.arange(len(CLASS_NAMES)),
        xticklabels=[name.replace("_", "\n") for name in CLASS_NAMES],
        yticklabels=[name.replace("_", "\n") for name in CLASS_NAMES],
        xlabel="Predicted activity",
        ylabel="True activity",
        title="LSTM confusion matrix",
    )
    plt.setp(axis.get_xticklabels(), rotation=25, ha="right")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(column, row, str(matrix[row, column]), ha="center", va="center")
    figure.tight_layout()
    figure.savefig(destination, dpi=160)
    plt.close(figure)


def save_training_curves(history: dict[str, list[float]], destination: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(history["loss"], label="Train")
    axes[0].plot(history["val_loss"], label="Validation")
    axes[0].set(title="Loss", xlabel="Epoch", ylabel="Cross-entropy")
    axes[1].plot(history["accuracy"], label="Train")
    axes[1].plot(history["val_accuracy"], label="Validation")
    axes[1].set(title="Accuracy", xlabel="Epoch", ylabel="Accuracy")
    for axis in axes:
        axis.legend()
        axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(destination, dpi=160)
    plt.close(figure)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


if __name__ == "__main__":
    main()
