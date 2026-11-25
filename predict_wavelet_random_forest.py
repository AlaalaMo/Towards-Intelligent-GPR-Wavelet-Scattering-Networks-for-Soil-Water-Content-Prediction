"""Predict SWC by depth for labeled or unlabeled config 41 scans."""

import argparse
from pathlib import Path

import pandas as pd

from evaluate_wavelet_random_forest import PREDICTION_COLUMNS, predict_scans, read_scans
from wavelet_rf_split import ROW_ID


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scans", required=True, type=Path)
    parser.add_argument("--artifact-dir", default=Path("artifacts"), type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    scans, _ = read_scans(args.scans)
    predicted, _ = predict_scans(scans, args.artifact_dir)
    result = pd.DataFrame(predicted, columns=PREDICTION_COLUMNS)
    result.insert(0, "row_id", scans[ROW_ID].to_numpy())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
