"""Parity and schema checks for config 41 without private scans."""

import csv
import json
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from Data_process import height_wavelet_transform, transform_location_wise_height_group
from evaluate_wavelet_random_forest import PREDICTION_COLUMNS, read_scans
from fit_wavelet_random_forest import FOREST_PARAMETERS, fit_bundle
from loading_in_config import get_subset_columns
from wavelet_rf_features import SCAN_COLUMNS, TARGET_COLUMNS, extract_features
from wavelet_rf_split import LOCATION, ROW_ID, partition_scans


def test_config_41_matches_research_fitter_and_is_row_local():
    rng = np.random.default_rng(41)
    signals = rng.normal(size=(2, len(SCAN_COLUMNS))) + 1j * rng.normal(
        size=(2, len(SCAN_COLUMNS))
    )
    scans = pd.DataFrame(signals, columns=pd.MultiIndex.from_tuples(SCAN_COLUMNS))
    research_input = scans.copy()
    research_input[("meta", "location")] = ["site", "site"]
    research_input[("y", "avgdepth3")] = [20, 30]
    transformed = transform_location_wise_height_group(
        research_input, [height_wavelet_transform]
    )
    research = get_subset_columns(transformed, "height_2", "all")["height_2"]
    release = extract_features(scans)
    pd.testing.assert_index_equal(release["height_2"].columns, research.columns)
    np.testing.assert_allclose(release["height_2"], research, rtol=1e-12, atol=1e-14)
    pd.testing.assert_frame_equal(
        release.iloc[[1]], extract_features(scans.iloc[[1]]), rtol=1e-12, atol=1e-14
    )


def test_config_41_rejects_invalid_inputs():
    scans = pd.DataFrame(
        np.zeros((1, len(SCAN_COLUMNS)), dtype=complex),
        columns=pd.MultiIndex.from_tuples(SCAN_COLUMNS),
    )
    with pytest.raises(ValueError, match="frequency order"):
        extract_features(scans.iloc[:, ::-1])
    with pytest.raises(ValueError, match="frequency order"):
        extract_features(scans.iloc[:, :-1])
    with pytest.raises(ValueError, match="zero-range"):
        extract_features(scans)
    scans.iloc[0, 0] = complex(np.nan)
    with pytest.raises(ValueError, match="nonfinite"):
        extract_features(scans)


def test_split_and_fitted_bundle_preserve_row_order():
    rng = np.random.default_rng(42)
    samples = rng.normal(size=(5, len(SCAN_COLUMNS), 2))
    scans = pd.DataFrame(
        samples[:, :, 0] + 1j * samples[:, :, 1],
        columns=pd.MultiIndex.from_tuples(SCAN_COLUMNS),
    )
    scans[LOCATION] = ["b", "a", "a", "b", "a"]
    scans[ROW_ID] = ["b0", "a0", "a1", "b1", "a2"]
    for depth, target_column in enumerate(TARGET_COLUMNS):
        scans[target_column] = [10 + depth, 20 + depth, 50, 30 + depth, 40 + depth]
    scans[("height_1", "other")] = 0
    provenance = {
        "location": "all",
        "split_function": "test_train_80_20",
        "test_size": 0.5,
        "random_seed": 42,
        "stratification_columns": [],
    }
    split = partition_scans(scans, **provenance)
    assert split.eligible[ROW_ID].tolist() == ["a0", "a2", "b0", "b1"]
    assert split.train[ROW_ID].tolist() == ["a0", "a2"]
    assert split.test[ROW_ID].tolist() == ["b0", "b1"]
    assert list(split.test.columns) == list(scans.columns[:-1])

    scaler, model, columns = fit_bundle(split, estimator_seed=42)
    training_features = extract_features(split.train).to_numpy()
    assert scaler.n_samples_seen_ == len(split.train)
    np.testing.assert_allclose(scaler.mean_, training_features.mean(axis=0))
    np.testing.assert_allclose(scaler.var_, training_features.var(axis=0))
    assert all(model.get_params()[key] == value for key, value in FOREST_PARAMETERS.items())
    assert model.random_state is None
    test_features = extract_features(split.test)
    pd.testing.assert_index_equal(test_features.columns, columns)
    assert test_features.shape[0] == 2
    scaled = scaler.transform(test_features.to_numpy())
    predictions = model.predict(scaled)
    assert predictions.shape == (2, 4)
    loaded_scaler, loaded_model = pickle.loads(pickle.dumps((scaler, model)))
    single_features = extract_features(split.test.iloc[[1]])
    single_scaled = loaded_scaler.transform(single_features.to_numpy())
    np.testing.assert_allclose(single_scaled, scaled[[1]], rtol=1e-12, atol=1e-14)
    np.testing.assert_array_equal(loaded_model.predict(single_scaled), predictions[[1]])


def test_release_commands_in_fresh_processes(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    raw = tmp_path / "Final_data" / "freq_mean_sweep" / "unprocessed" / "raw.csv"
    raw.parent.mkdir(parents=True)
    bundle = tmp_path / "artifacts"
    evaluation = tmp_path / "evaluation"
    held_out = tmp_path / "eval" / "test_dataset.csv"
    unlabeled = tmp_path / "unlabeled.csv"
    estimates = tmp_path / "estimates.csv"
    sample_count = len(SCAN_COLUMNS)
    rng = np.random.default_rng(41)
    signals = rng.normal(size=(10, 3 * sample_count, 2))
    with raw.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["x"] * (3 * sample_count) + ["y"] * 4)
        writer.writerow(list(range(sample_count)) * 3 + [col[1] for col in TARGET_COLUMNS])
        for i, scan in enumerate(signals):
            signal = scan[:, 0] + 1j * scan[:, 1]
            writer.writerow([*signal, *(10 + i + depth for depth in range(4))])

    def run(script, *args, cwd=tmp_path):
        result = subprocess.run(
            [sys.executable, "-B", str(repo / script), *map(str, args)],
            cwd=cwd, capture_output=True, text=True,
        )
        assert result.returncode == 0, result.stderr

    run("fit_wavelet_random_forest.py")
    raw.unlink()
    run("evaluate_wavelet_random_forest.py", "--artifact-dir", bundle,
        "--eval-dataset", held_out, "--output-dir", evaluation)
    scans, dataset_hash = read_scans(held_out)
    scans.drop(columns=list(TARGET_COLUMNS)).to_csv(unlabeled, index=False)
    run("predict_wavelet_random_forest.py", "--artifact-dir", bundle, "--scans", unlabeled,
        "--output", estimates)

    def csv_rows(path):
        with path.open(newline="", encoding="utf-8") as stream:
            return list(csv.reader(stream))

    prediction_rows = csv_rows(evaluation / "predictions.csv")
    estimate_rows = csv_rows(estimates)
    assert [row[:-4] for row in prediction_rows] == csv_rows(held_out)
    assert prediction_rows[0][-4:] == PREDICTION_COLUMNS
    assert estimate_rows[0] == ["row_id", *PREDICTION_COLUMNS]
    _, expected_test = train_test_split(
        list(range(10)), test_size=0.2, random_state=42, stratify=["fixture"] * 10
    )
    assert [int(row[0].rsplit("#", 1)[1]) for row in estimate_rows[1:]] == expected_test
    assert [row[-4:] for row in prediction_rows[2:]] == [row[1:] for row in estimate_rows[1:]]
    metrics = json.loads((evaluation / "metrics.json").read_text(encoding="utf-8"))
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    assert metrics["run_id"] == manifest["run_id"]
    assert manifest["standard_scaler_fit_scope"] == "training_rows_only"
    assert manifest["split_strategy"]["split_function"] == "stratified_train_test_split_80_20"
    scaler = pickle.loads((bundle / "standard_scaler.pkl").read_bytes())
    assert scaler.n_samples_seen_ == 8
    assert metrics["evaluation_dataset_sha256"] == dataset_hash
    assert metrics["matches_exported_held_out_dataset"]
    assert metrics["evaluated_row_count"] == 2
    expected = scans.loc[:, list(TARGET_COLUMNS)].to_numpy(dtype=float)
    predicted = np.array([row[1:] for row in estimate_rows[1:]], dtype=float)
    errors = expected - predicted
    np.testing.assert_allclose(
        [metrics["rmse"], metrics["mape"], metrics["mae"]],
        [np.sqrt(np.mean(errors**2)), np.mean(100 * np.abs(errors / expected)),
         np.mean(np.abs(errors))],
    )
