"""Fit and export a config 41 release bundle from private run provenance."""

import argparse
import hashlib
import json
import pickle
import platform
import subprocess
from importlib.metadata import version
from pathlib import Path
from uuid import uuid4

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

from wavelet_rf_features import (
    DEPTHS_CM,
    SCAN_COLUMNS,
    SCHEMA_VERSION,
    TARGET_COLUMNS,
    WAVELET_SETTINGS,
    extract_features,
)
from wavelet_rf_split import ROW_ID, export_evaluation_scans, load_private_scans, verified_partitions


FOREST_PARAMETERS = {
    "ccp_alpha": 0.13319440636297358,
    "max_depth": 3,
    "max_features": 0.2,
    "max_leaf_nodes": 10,
    "min_impurity_decrease": 0.00010000000000000009,
    "min_samples_leaf": 8,
    "min_samples_split": 10,
    "n_estimators": 20,
}


def fit_bundle(split, estimator_seed):
    """Fit a scaler on the full eligible feature matrix and a forest on train rows."""
    eligible = split.eligible.set_index(ROW_ID, drop=False)
    features = extract_features(eligible)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features.to_numpy())
    train_positions = eligible.index.get_indexer(split.train[ROW_ID])
    if (train_positions < 0).any():
        raise ValueError("Verified training IDs must occur in the eligible feature rows.")
    targets = split.train.loc[:, list(TARGET_COLUMNS)].to_numpy(dtype=np.float64)
    if not np.isfinite(targets).all():
        raise ValueError("The training targets contain nonfinite values.")
    model = RandomForestRegressor(**FOREST_PARAMETERS)
    # The paper's estimator uses the global NumPy generator. Seed a new
    # fixed-parameter run for reproducibility without advancing caller state.
    prior_state = np.random.get_state()
    try:
        np.random.seed(estimator_seed)
        model.fit(scaled[train_positions], targets)
    finally:
        np.random.set_state(prior_state)
    return scaler, model, features.columns


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private-run", required=True, type=Path)
    parser.add_argument("--artifact-dir", default=Path("artifacts"), type=Path)
    parser.add_argument("--eval-output", default=Path("eval/test_dataset.csv"), type=Path)
    args = parser.parse_args()
    record = json.loads(args.private_run.read_text(encoding="utf-8"))
    estimator_seed = record["estimator_seed"]
    if type(estimator_seed) is not int or not 0 <= estimator_seed < 2**32:
        parser.error("The private run record must specify a numeric estimator_seed.")
    sources = [
        (str(args.private_run.parent / source["path"]), source["original_location"])
        for source in record["sources"]
    ]
    options = record["split"].copy()
    options["stratification_columns"] = [
        tuple(column) for column in options["stratification_columns"]
    ]
    split = verified_partitions(load_private_scans(sources), **options)
    scaler, model, feature_columns = fit_bundle(split, estimator_seed)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    source_root = Path(__file__).resolve().parent
    source_files = (
        "fit_wavelet_random_forest.py",
        "wavelet_rf_features.py",
        "wavelet_rf_split.py",
        "Data_process.py",
        "akela_shared/splitting.py",
        "akela_shared/csv_loader.py",
        "preprocessing_data_loader.py",
    )
    manifest = {
        "artifact_format_version": 1,
        "schema_version": SCHEMA_VERSION,
        "run_id": str(uuid4()),
        "config_index": 41,
        "domain": "freq_mean_sweep",
        "wavelet_transform": "height_wavelet_transform",
        "wavelet_settings": WAVELET_SETTINGS,
        "standard_scaler_fit_scope": "all_eligible_rows_before_split",
        "scan_columns": SCAN_COLUMNS,
        "feature_columns": list(feature_columns),
        "depths_cm": DEPTHS_CM,
        "target_columns": TARGET_COLUMNS,
        "estimator": "sklearn.ensemble.RandomForestRegressor",
        "estimator_parameters": model.get_params(),
        "release_numpy_seed": estimator_seed,
        "code_revision": revision,
        "source_sha256": {
            name: hashlib.sha256((source_root / name).read_bytes()).hexdigest()
            for name in source_files
        },
        "dependencies": {
            "python": platform.python_version(),
            **{name: version(name) for name in ("numpy", "pandas", "scipy", "kymatio", "scikit-learn")},
        },
        "sha256": {},
    }
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    for name, obj in (("standard_scaler.pkl", scaler), ("random_forest.pkl", model)):
        payload = pickle.dumps(obj)
        (args.artifact_dir / name).write_bytes(payload)
        manifest["sha256"][name] = hashlib.sha256(payload).hexdigest()
    export_evaluation_scans(split.test, args.eval_output)
    manifest["held_out_dataset_sha256"] = hashlib.sha256(args.eval_output.read_bytes()).hexdigest()
    (args.artifact_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
