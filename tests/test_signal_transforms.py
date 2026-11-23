"""Regression tests for signal arithmetic and DataFrame transform contracts."""

from unittest import TestCase

import numpy as np
import pandas as pd

import Data_process as dp


def height_frame(values: np.ndarray) -> pd.DataFrame:
    """Build one height block with numbered sample labels."""
    columns = pd.MultiIndex.from_product([["height_1"], range(values.shape[1])])
    return pd.DataFrame(values, columns=columns)


class SignalTransformTests(TestCase):
    """Check signal values, boundary handling, and output labels."""

    def test_center_and_normalize_real_arrays(self) -> None:
        data = np.array([[1., 4.], [3., 8.]])
        np.testing.assert_array_equal(dp.center_array_columns(data), [[-1, -2], [1, 2]])
        # The current column routine returns mean divided by standard deviation.
        np.testing.assert_array_equal(dp.z_score_normalize_column(data), [[2, 3], [2, 3]])
        np.testing.assert_array_equal(dp.min_max_normalize_rows(data), [[0, 1], [0, 1]])
        with np.errstate(invalid="ignore"):
            self.assertTrue(np.isnan(dp.min_max_normalize_rows(np.ones((1, 2)))).all())

    def test_complex_magnitude_log_and_normalization(self) -> None:
        data = np.array([1j, -3])
        np.testing.assert_array_equal(dp.power_convsersion(data), [1, 3])
        np.testing.assert_allclose(dp.log_transform_complex(data),
                                   [1j * np.log(2), -np.log(4)], atol=1e-14)
        np.testing.assert_allclose(dp.z_score_normalize_complex(data), [-1j, -1],
                                   atol=1e-14)
        np.testing.assert_array_equal(dp.z_score_normalize_complex(np.array([1j, -1j])),
                                      [-1 + 1j, -1 - 1j])

    def test_column_wrappers(self) -> None:
        frame = height_frame(np.array([[1j], [-3]]))
        cases = (
            (dp.column_center_by_loc, "centered_0", [1.5 + .5j, -1.5 - .5j]),
            (dp.p_o_wer_conversion, "mag_0", [1., 3.]),
            (dp.column_log_transform, "log_i_0", [1j * np.log(2), -np.log(4)]),
            (dp.column_z_mag_transform, "z_mag0", [-1j, -1]),
        )
        for transform, label, expected in cases:
            with self.subTest(transform=transform.__name__):
                result = transform(frame, ["height_1"])
                self.assertEqual(list(result.columns), [("height_1", label)])
                np.testing.assert_allclose(result.iloc[:, 0], expected, atol=1e-14)
                self.assertEqual(result.dtypes.iloc[0], np.asarray(expected).dtype)

    def test_difference_strides_and_validation(self) -> None:
        data = np.array([[0, 1, 4, 9, 16]])
        np.testing.assert_array_equal(dp.compute_column_difference(data), [[4, 8, 12]])
        np.testing.assert_array_equal(dp.compute_kth_block_column_difference(data), [[4, 12]])
        for transform in (dp.compute_column_difference, dp.compute_kth_block_column_difference):
            with self.subTest(transform=transform.__name__):
                with self.assertRaises(TypeError):
                    transform(data.tolist())
                with self.assertRaises(ValueError):
                    transform(data, k=5)

    def test_difference_wrappers(self) -> None:
        frame = height_frame(np.arange(35, dtype=float)[None, :])
        cases = (
            (dp.column_difference_2, "2diff_", 2, 33),
            (dp.column_difference_16, "16diff_", 16, 19),
            (dp.column_difference_block_2, "b16diff_", 2, 17),
            (dp.column_difference_block_16, "b16diff_", 16, 2),
        )
        for transform, prefix, step, width in cases:
            with self.subTest(transform=transform.__name__):
                result = transform(frame, ["height_1"])
                self.assertEqual(list(result.columns),
                                 [("height_1", f"{prefix}{i}") for i in range(width)])
                np.testing.assert_array_equal(result.to_numpy(), np.full((1, width), step))

    def test_derivative_boundaries(self) -> None:
        quadratic = np.array([[0., 1., 4., 9., 16.]])
        first = dp.compute_first_derivative_central_with_boundaries_subset
        second = dp.compute_second_derivative_central_with_boundaries_subset
        np.testing.assert_array_equal(first(quadratic, h=1), [[1., 2., 4., 6., 7.]])
        np.testing.assert_array_equal(second(quadratic, h=1), [[2., 2., 2., 2., 2.]])
        for derivative in (first, second):
            with self.subTest(derivative=derivative.__name__):
                # The current default h is a float used as a slice step.
                with self.assertRaises(TypeError):
                    derivative(quadratic)
                with self.assertRaises(ValueError):
                    derivative(quadratic, h=1, k=5)

    def test_derivative_wrappers_preserve_signal_and_feature_order(self) -> None:
        data = np.arange(52, dtype=float)[None, :]
        frame = height_frame(data)
        cases = (
            (dp.numerical_first_second_1_h_1_k, dp.numerical_first_derivative_1_h_1_k,
             dp.numerical_second_derivative_1_h_1_k, 1, 52),
            (dp.numerical_first_second_4_h_4_k, dp.numerical_first_derivative_4_h_4_k,
             dp.numerical_second_derivative_4_h_4_k, 4, 13),
        )
        for combined, first, second, stride, width in cases:
            with self.subTest(stride=stride):
                result = combined(frame, ["height_1"])
                pd.testing.assert_frame_equal(result.iloc[:, :52], frame)
                for transform, prefix, start in (
                    (first, "fir", 52), (second, "sec", 52 + width)
                ):
                    individual = transform(frame, ["height_1"])
                    pd.testing.assert_frame_equal(individual.iloc[:, :52], frame)
                    features = individual.iloc[:, 52:]
                    self.assertEqual(list(features.columns),
                                     [("height_1", f"{prefix}_{stride}_{i}")
                                      for i in range(width)])
                    pd.testing.assert_frame_equal(features, result.iloc[:, start:start + width])
                if stride == 4:
                    np.testing.assert_array_equal(result.iloc[:, 52:65], np.full((1, 13), 4))
                    np.testing.assert_array_equal(result.iloc[:, 65:], np.zeros((1, 13)))
                else:
                    # The five-point mirror filter leaves the interior ramp linear.
                    np.testing.assert_allclose(result.iloc[:, 55:101], 1., atol=1e-12)
                    np.testing.assert_allclose(result.iloc[:, 107:153], 0., atol=1e-12)

    def test_savgol_mirror_padding_and_complex_channels(self) -> None:
        ramp = np.array([[0., 1., 2., 3., 4.]])
        expected = np.array([[2/3, 1., 2., 3., 10/3]])
        np.testing.assert_allclose(dp.savgol_moving_average_per_row(ramp, 3, 1), expected)
        np.testing.assert_allclose(dp.savgol_moving_average_per_row(ramp * (1 + 2j), 3, 1),
                                   expected * (1 + 2j))
        with self.assertRaises(ValueError):
            dp.savgol_moving_average_per_row(ramp, window_length=4)

    def test_binned_impulse_response_and_power_delay(self) -> None:
        row = np.array([1j, 3j, 1., 5., 99.])
        np.testing.assert_array_equal(dp.process_row_h(row, 2, 2), [2j, 3])
        np.testing.assert_array_equal(dp.process_row_pdp([2j, 3]), [4, 9])
        frame = height_frame(row[None, :])
        frame.index = [10]
        settings = dict(bandwidth=2, receiption_time=1, N=5)
        impulse = dp.h_t_transform(frame, ["height_1"], **settings)
        power = dp.p_d_p_transform(frame, ["height_1"], **settings)
        self.assertEqual(list(impulse.columns), [("height_1", "h(t)_1"), ("height_1", "h(t)_2")])
        self.assertEqual(list(power.columns), [("height_1", "pdp_0"), ("height_1", "pdp_1")])
        self.assertEqual(list(impulse.index), [10])
        self.assertEqual(list(power.index), [0])
        np.testing.assert_array_equal(impulse.to_numpy(), [[2j, 3]])
        np.testing.assert_array_equal(power.to_numpy(), [[4., 9.]])
