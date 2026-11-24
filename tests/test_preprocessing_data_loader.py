"""Test input loading with temporary two-header-row CSV fixtures."""

import os
from argparse import Namespace
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

import numpy as np
import pandas as pd

import preprocessing_data_loader as loader


class PreprocessingDataLoaderTests(TestCase):
    """Check discovery order, schema checks, and input data conversion."""

    def test_load_original_csv_schema_and_preserve_missing_values(self) -> None:
        args = Namespace(domain="freq_mean_sweep")
        with TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "b.csv"
            second = root / "a.csv"
            first.write_text(
                "x,y,y,y,y\n0,shed_y,avgdepth5,avgdepth6,target\n"
                "1 + 2j,1,10,20,30\n,0,11,21,31\n",
                encoding="utf-8",
            )
            second.write_text(
                "x,y,y\n0,target,shed\n3-4j,40,0\n", encoding="utf-8"
            )
            paths = [str(first), str(second)]
            output = StringIO()
            with patch.object(loader.glob, "glob", return_value=paths) as discover:
                with redirect_stdout(output):
                    result = loader.load_initial_dataset(args)
            discover.assert_called_once_with(
                str(Path("Final_data", "freq_mean_sweep", "unprocessed", "*.csv"))
            )
            self.assertIsNotNone(result)
            self.assertEqual(list(result.columns),
                             [("height_1", "freq_mean_sweep_0"), ("y", "shed"),
                              ("y", "target"), ("meta", "location")])
            signal = result[("height_1", "freq_mean_sweep_0")]
            self.assertEqual(signal.dtype, np.dtype("complex128"))
            self.assertEqual(signal.iloc[0], 1 + 2j)
            self.assertTrue(np.isnan(signal.iloc[1]))
            self.assertEqual(signal.iloc[2], 3 - 4j)
            self.assertEqual(result[("y", "target")].tolist(), [30, 31, 40])
            self.assertEqual(result[("meta", "location")].tolist(),
                             [str(first), str(first), str(second)])
            self.assertEqual(list(result.index), [0, 1, 0])
            self.assertIn("contains 1 NaN values", output.getvalue())

    def test_empty_inputs_and_schema_mismatches_return_none(self) -> None:
        columns = pd.MultiIndex.from_tuples([("height_1", "x")])
        reference = pd.DataFrame([[1.]], columns=columns)
        wrong_dtype = reference.astype("complex128")
        wrong_columns = reference.rename(columns={"x": "other"}, level=1)
        cases = (([], "No DataFrames"), ([reference, wrong_dtype], "Type mismatch"),
                 ([reference, wrong_columns], "Column mismatch"))
        for frames, message in cases:
            with self.subTest(message=message):
                output = StringIO()
                with redirect_stdout(output):
                    self.assertIsNone(loader.check_and_concatenate_dataframes(frames))
                self.assertIn(message, output.getvalue())

    def test_reject_invalid_complex_input(self) -> None:
        columns = pd.MultiIndex.from_tuples([("x", "0")])
        with self.assertRaises(ValueError):
            loader.convert_x_columns_to_complex(pd.DataFrame([["invalid"]], columns=columns))
        with self.assertRaises(TypeError):
            loader.convert_x_columns_to_complex(pd.DataFrame([[1.]], columns=columns))

    def test_loads_only_csvs_in_the_requested_unprocessed_domain(self) -> None:
        """Check discovery, schema, values, and source paths together."""
        domain = "freq_mean_sweep"
        with TemporaryDirectory() as directory:
            original_directory = os.getcwd()
            os.chdir(directory)
            try:
                base = Path("Final_data")
                raw = base / domain / "unprocessed"
                expected_rows = {
                    str(raw / "shed_b.csv"): [2 + 3j, 20],
                    str(raw / "shed_a.csv"): [1 + 4j, 10],
                }
                for filename, (signal, target) in expected_rows.items():
                    path = Path(filename)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(
                        f"x,y,y\nsample,shed_y,avgdepth5\n{signal},{target},99\n",
                        encoding="utf-8",
                    )
                for path in (base / "other" / "unprocessed" / "other.csv",
                             base / domain / "processed" / "processed.csv",
                             raw / "notes.txt"):
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text("unused", encoding="utf-8")

                discovered = loader.get_csv_files(domain=domain)
                self.assertCountEqual(discovered, expected_rows)
                expected = pd.DataFrame(
                    [expected_rows[path] + [path] for path in discovered],
                    columns=pd.MultiIndex.from_tuples([
                        ("height_1", f"{domain}_sample"),
                        ("y", "shed"),
                        ("meta", "location"),
                    ]),
                    index=[0] * len(discovered),
                )
                result = loader.load_initial_dataset(Namespace(domain=domain))
                pd.testing.assert_frame_equal(result, expected)
                self.assertIsNone(loader.load_initial_dataset(Namespace(domain="missing")))
            finally:
                os.chdir(original_directory)
