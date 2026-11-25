"""Verify config 41 split membership before selecting release scan columns."""

import csv
from argparse import Namespace
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import NamedTuple

import numpy as np
import pandas as pd

from akela_shared import splitting
from akela_shared.csv_loader import load_and_process_csv
from preprocessing_data_loader import check_and_concatenate_dataframes


HEIGHT = "height_2"
ROW_ID = ("meta", "row_id")
LOCATION = ("meta", "location")
ELIGIBILITY_TARGET = ("y", "avgdepth3")


class VerifiedSplit(NamedTuple):
    eligible: pd.DataFrame
    train: pd.DataFrame
    test: pd.DataFrame


def load_private_scans(sources):
    """Load original input files in recorded order.

    Each source pairs a current input path with the original research
    location label. Row IDs combine that label with the zero-based data
    row number within the source before any filtering.
    """
    args = Namespace(domain="freq_mean_sweep")
    frames = []
    seen_locations = set()
    for path, location in sources:
        if not isinstance(location, str) or not location or location in seen_locations:
            raise ValueError("Each input file needs a unique original location label.")
        seen_locations.add(location)
        frame = load_and_process_csv(path, args)
        frame[LOCATION] = location
        frame[ROW_ID] = [f"{location}#{position}" for position in range(len(frame))]
        frames.append(frame)
    scans = check_and_concatenate_dataframes(frames)
    if scans is None:
        raise ValueError("Original input CSVs are missing or have incompatible schemas.")
    return scans


def verified_partitions(
    scans,
    *,
    location,
    split_function,
    test_size,
    random_seed,
    stratification_columns,
    expected_train_ids,
    expected_test_ids,
):
    """Select raw partitions and require an exact ordered match to run evidence.

    Scans must already have research input column names and stable row IDs.
    Supply the original run settings and the ordered IDs from its private
    split evidence.
    """
    if not isinstance(scans.columns, pd.MultiIndex) or scans.columns.nlevels != 2:
        raise ValueError("Expected a two-level scan header.")
    if scans.columns.has_duplicates:
        raise ValueError("Scan header contains duplicate columns.")
    required = {ROW_ID, LOCATION, ELIGIBILITY_TARGET}
    if not required.issubset(scans.columns):
        raise ValueError("Scans must include row_id, location and avgdepth3.")
    row_ids = scans[ROW_ID]
    if row_ids.isna().any() or row_ids.duplicated().any() or row_ids.eq("").any():
        raise ValueError("Scan row IDs must be present and unique.")

    # Research preprocessing groups by location with sorted keys and preserves
    # the input order within each location before main_fitting applies filters.
    ordered = scans.sort_values(LOCATION, kind="stable").dropna(subset=[LOCATION])
    eligible = ordered.loc[ordered[ELIGIBILITY_TARGET] <= 48]
    if location.lower() != "all":
        eligible = eligible.loc[eligible[LOCATION].str.contains(location)]

    ids = eligible[ROW_ID]
    if split_function == "stratified_train_test_split_80_20":
        if not stratification_columns:
            raise ValueError("Supply the original ordered stratification columns.")
        metadata = eligible.loc[:, list(stratification_columns)]
        train, _, _, test, _, _ = splitting.stratified_train_test_split_80_20(
            ids, ids, metadata, test_size=test_size, random_seed=random_seed
        )
    elif split_function == "test_train_80_20":
        if stratification_columns:
            raise ValueError("The sequential splitter does not stratify.")
        train, _, _, test, _, _ = splitting.test_train_80_20(
            ids, ids, test_size=test_size, random_seed=random_seed
        )
    else:
        raise ValueError(f"Unsupported research split function: {split_function}")

    for name, actual, expected in (
        ("training", train, expected_train_ids),
        ("held-out", test, expected_test_ids),
    ):
        if actual.tolist() != list(expected):
            raise ValueError(f"The {name} row IDs or ordering differ from the original run.")

    levels = eligible.columns.get_level_values(0)
    if HEIGHT not in levels:
        raise ValueError(f"Missing required {HEIGHT} scan columns.")
    keep = ~levels.str.startswith("height") | (levels == HEIGHT)
    selected = eligible.loc[:, keep].set_index(ROW_ID, drop=False)
    return VerifiedSplit(
        selected.reset_index(drop=True),
        selected.loc[train.tolist()].reset_index(drop=True),
        selected.loc[test.tolist()].reset_index(drop=True),
    )


def export_evaluation_scans(scans, destination):
    """Stage the held-out CSV and check headers, identity and complex values."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    height_columns = [
        index for index, (group, _) in enumerate(scans.columns) if group == HEIGHT
    ]
    expected_signals = scans.iloc[:, height_columns].to_numpy(dtype=np.complex128)
    if not height_columns or not np.isfinite(expected_signals).all():
        raise ValueError(f"Evaluation scans must include finite {HEIGHT} values.")
    id_column = scans.columns.get_loc(ROW_ID)
    temp_path = None
    try:
        with NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="", dir=destination.parent, delete=False
        ) as stream:
            temp_path = Path(stream.name)
            scans.to_csv(stream, index=False)
        with temp_path.open(encoding="utf-8", newline="") as stream:
            reader = csv.reader(stream)
            headers = list(zip(next(reader), next(reader)))
            if headers != list(scans.columns):
                raise ValueError("Evaluation CSV did not preserve both header levels.")
            rows = list(reader)
        ids = [row[id_column] for row in rows]
        signals = np.array(
            [[complex(row[index]) for index in height_columns] for row in rows],
            dtype=np.complex128,
        )
        if ids != scans[ROW_ID].astype(str).tolist() or not np.array_equal(
            signals, expected_signals
        ):
            raise ValueError("Evaluation CSV did not preserve row IDs and complex scans.")
        temp_path.replace(destination)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
