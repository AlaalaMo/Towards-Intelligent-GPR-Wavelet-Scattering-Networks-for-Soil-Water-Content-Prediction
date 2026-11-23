"""Protect configuration order, transform resolution, and output paths."""

from argparse import Namespace
from contextlib import redirect_stdout
from io import StringIO
import os
from unittest import TestCase
from unittest.mock import patch

from preprocessing_config import generate_configurations


LOCATION = "location_wise_transforms"
HEIGHT = "height_wise_transforms"
ROW = "location_wise_height_wise_row_wise_transforms"
CENTER = "column_center_by_loc"
POWER = "p_o_wer_conversion"
LOG = "column_log_transform"
ZSCORE = "column_z_mag_transform"
BOTH = "numerical_first_second_1_h_1_k"
FIRST = "numerical_first_derivative_1_h_1_k"
SECOND = "numerical_second_derivative_1_h_1_k"
DIFF_2 = "column_difference_2"
DIFF_16 = "column_difference_16"


class PreprocessingConfigTests(TestCase):
    """Characterize the current experimental configuration contract."""

    def test_complete_configurations_and_manifest_in_order(self) -> None:
        centered_suffix = [CENTER, LOG, ZSCORE]
        uncentered_suffix = [LOG, ZSCORE]
        initial_options = [
            False, [CENTER], [CENTER, LOG], centered_suffix, [CENTER, ZSCORE],
            [POWER, CENTER],
            [POWER, BOTH, *centered_suffix], [POWER, FIRST, *centered_suffix],
            [POWER, SECOND, *centered_suffix],
            [BOTH, *centered_suffix], [FIRST, *centered_suffix], [SECOND, *centered_suffix],
            [POWER, DIFF_2, *centered_suffix], [POWER, DIFF_16, *centered_suffix],
        ]
        uncentered_options = [
            [], [LOG], uncentered_suffix, [ZSCORE], [POWER],
            [POWER, BOTH, *uncentered_suffix], [POWER, FIRST, *uncentered_suffix],
            [POWER, SECOND, *uncentered_suffix],
            [BOTH, *uncentered_suffix], [FIRST, *uncentered_suffix], [SECOND, *uncentered_suffix],
            [POWER, DIFF_2, *uncentered_suffix], [POWER, DIFF_16, *uncentered_suffix],
        ]
        expected = [
            {"domain": "freq_mean_sweep", "save_location": "up_to_may_2023",
             LOCATION: location, HEIGHT: height, ROW: False}
            for batch in (initial_options, uncentered_options)
            for height in (False, ["height_wavelet_transform"])
            for location in batch
        ]
        tasks, manifest = generate_configurations()
        self.assertEqual(len(tasks), 54)
        for records in ([vars(task) for task in tasks], manifest):
            # Item lists also check the field order used in output naming.
            self.assertEqual([list(record.items()) for record in records],
                             [list(record.items()) for record in expected])

    def test_every_configured_name_resolves_to_its_data_process_function(self) -> None:
        import Data_process as dp
        import preprocessing_main as runner

        tasks, _ = generate_configurations()
        for task in tasks:
            for field in (LOCATION, HEIGHT, ROW):
                for name in getattr(task, field) or ():
                    with self.subTest(name=name):
                        self.assertIs(getattr(runner, name), getattr(dp, name))
        with self.assertRaisesRegex(KeyError, "missing_transform"):
            runner.process_configuration(Namespace(**{LOCATION: ["missing_transform"]}))

    def test_writer_paths_agree_with_manifest_reader_abbreviations(self) -> None:
        import pandas as pd

        from loading_in_config import abbreviate_transform_name as reader_abbreviation
        import preprocessing_main as runner

        tasks, manifest = generate_configurations()
        columns = pd.MultiIndex.from_tuples(
            [(f"height_{height}", "x") for height in (1, 2, 3)]
        )
        frame = pd.DataFrame([[1., 2., 3.]], columns=columns)
        with patch.object(runner, "concat_df", frame), \
             patch.object(runner, "transform_location_wise_height_group",
                          side_effect=lambda data, callbacks: data), \
             patch.object(runner.os.path, "exists", return_value=False), \
             patch.object(runner.os, "makedirs"), \
             patch.object(pd.DataFrame, "to_csv") as save, \
             redirect_stdout(StringIO()):
            for task, record in zip(tasks, manifest):
                runner.process_configuration(task)
                abbreviation = reader_abbreviation(record)
                expected = os.path.join("Final_data", record["domain"],
                                        record["save_location"], abbreviation, "transformed.csv")
                path, = save.call_args[0]
                self.assertEqual(os.path.normpath(path), expected)
                self.assertEqual(save.call_args[1], {"index": False})
