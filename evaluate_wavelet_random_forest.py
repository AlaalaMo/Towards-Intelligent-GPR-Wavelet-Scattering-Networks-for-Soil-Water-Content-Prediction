"""Evaluate the frozen config 41 bundle on a two-header-row raw CSV."""

import argparse
import csv
import hashlib
import io
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from wavelet_rf_features import (
    DEPTHS_CM,
    SCAN_COLUMNS,
    SCHEMA_VERSION,
    TARGET_COLUMNS,
    WAVELET_SETTINGS,
    extract_features,
)
from wavelet_rf_split import ROW_ID


PREDICTION_COLUMNS = [f"pred_swc_{depth}cm" for depth in DEPTHS_CM]


def read_scans(path):
    payload = path.read_bytes()
    reader = csv.reader(io.StringIO(payload.decode("utf-8-sig"), newline=""))
    headers = list(zip(next(reader, ()), next(reader, ())))
    if not headers or len(set(headers)) != len(headers):
        raise ValueError("Expected unique columns in a two-level scan header.")
    scans = pd.read_csv(io.BytesIO(payload), header=[0, 1], dtype=str, keep_default_na=False)
    if ROW_ID not in scans or scans[ROW_ID].eq("").any() or not scans[ROW_ID].is_unique:
        raise ValueError("Supply a unique (meta, row_id) value on every scan.")
    return scans, hashlib.sha256(payload).hexdigest()


def predict_scans(scans, artifact_dir):
    manifest = json.loads((artifact_dir / "manifest.json").read_text(encoding="utf-8"))
    expected = {
        "artifact_format_version": 1,
        "schema_version": SCHEMA_VERSION,
        "config_index": 41,
        "domain": "freq_mean_sweep",
        "wavelet_transform": "height_wavelet_transform",
        "wavelet_settings": WAVELET_SETTINGS,
        "scan_columns": [list(column) for column in SCAN_COLUMNS],
        "depths_cm": list(DEPTHS_CM),
        "target_columns": [list(column) for column in TARGET_COLUMNS],
    }
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise ValueError("The artifact manifest is incompatible with config 41.")
    features = extract_features(scans)
    if manifest.get("feature_columns") != [list(column) for column in features.columns]:
        raise ValueError("The bundle's feature ordering does not match extracted features.")
    objects = []
    for name in ("standard_scaler.pkl", "random_forest.pkl"):
        payload = (artifact_dir / name).read_bytes()
        if hashlib.sha256(payload).hexdigest() != manifest["sha256"].get(name):
            raise ValueError(f"Bundle checksum mismatch for {name}.")
        objects.append(pickle.loads(payload))
    scaler, model = objects
    predictions = model.predict(scaler.transform(features.to_numpy()))
    if predictions.shape != (len(scans), len(DEPTHS_CM)) or not np.isfinite(predictions).all():
        raise ValueError("The model returned invalid SWC predictions.")
    return predictions, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-dataset", required=True, type=Path)
    parser.add_argument("--artifact-dir", default=Path("artifacts"), type=Path)
    parser.add_argument("--output-dir", default=Path("eval"), type=Path)
    args = parser.parse_args()
    scans, dataset_hash = read_scans(args.eval_dataset)
    targets = scans.loc[:, list(TARGET_COLUMNS)].to_numpy(dtype=np.float64)
    if not np.isfinite(targets).all() or (targets == 0).any():
        raise ValueError("Finite nonzero SWC targets are required for research percent MAPE.")
    predicted, manifest = predict_scans(scans, args.artifact_dir)
    errors = targets - predicted
    metrics = {
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "mape": float(np.mean(np.abs(errors / targets) * 100)),
        "mae": float(np.mean(np.abs(errors))),
        "evaluated_row_count": len(scans),
        "run_id": manifest["run_id"],
        "evaluation_dataset_sha256": dataset_hash,
        "matches_exported_held_out_dataset": dataset_hash == manifest["held_out_dataset_sha256"],
    }
    for idx, name in enumerate(PREDICTION_COLUMNS):
        if name in scans.columns.get_level_values(0):
            raise ValueError(f"Input already contains a {name} column.")
        scans[(name, "prediction")] = predicted[:, idx]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    scans.to_csv(args.output_dir / "predictions.csv", index=False)
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
