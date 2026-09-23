# Human Activity Recognition with an LSTM

This is a small sequential-classification project for the supplied **UCI Human Activity Recognition Using Smartphones** data. It classifies six activities from raw accelerometer and gyroscope windows:

- walking
- walking upstairs
- walking downstairs
- sitting
- standing
- laying

Unlike the dataset's 561 engineered feature table, this project trains on the raw sequential inertial signals. Each example has **128 time steps × 9 sensor channels**, corresponding to 2.56 seconds sampled at 50 Hz.

## Project files

| File | Purpose |
| --- | --- |
| `src/data.py` | Reads the 9 raw signal files, builds `(samples, 128, 9)` tensors, and normalizes channels. |
| `train.py` | Splits training data into fitting/validation partitions, trains an LSTM, evaluates the held-out test set, and saves artifacts. |
| `app.py` | Streamlit UI for a saved demo window or a user-uploaded `.npy`/CSV sensor window. |

## Setup

TensorFlow currently has the smoothest installation path here with Python 3.11 or 3.12. From this project directory:

```bash
/Library/Frameworks/Python.framework/Versions/3.11/bin/python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install --timeout 300 -r requirements.txt
```

## Train and evaluate

Point `--dataset-dir` at the **directory named `UCI HAR Dataset`** (not its parent). For the supplied download:

```bash
python train.py \
  --dataset-dir "/Users/vinay/Downloads/human+activity+recognition+using+smartphones/UCI HAR Dataset" \
  --epochs 20
```

The script uses 80% of the original training split to fit and 20% for validation, stratified by activity. It fits the scaler only on the fitting partition, uses early stopping, and reports final performance only against the untouched UCI test split.

Saved in `artifacts/`:

- `har_lstm.keras` — the trained model
- `best_model.keras` — the best validation checkpoint
- `scaler.joblib`, `class_names.json`, and `sensor_channels.json` — preprocessing and metadata required by the app
- `metrics.json`, `classification_report.txt`, `confusion_matrix.png`, `training_curves.png` — evaluation outputs
- `demo_window.npy` — a raw held-out example for the Streamlit interface

## Run the prediction interface

After training:

```bash
streamlit run app.py
```

The app starts with the held-out demo example. It also accepts an `.npy` array or headerless CSV containing one `128 × 9` sensor window. Rows are time steps; columns must be in this order:

```text
body_acc_x, body_acc_y, body_acc_z,
body_gyro_x, body_gyro_y, body_gyro_z,
total_acc_x, total_acc_y, total_acc_z
```

The app applies the exact scaler saved during training before requesting a prediction.

## Model

`Input(128, 9) → LSTM(64) → Dense(32, ReLU) → Dropout(0.2) → Softmax(6)`

It is intentionally modest so it remains understandable and trains comfortably on a laptop. Increase `--epochs` or layer sizes only after reviewing validation and test results.
