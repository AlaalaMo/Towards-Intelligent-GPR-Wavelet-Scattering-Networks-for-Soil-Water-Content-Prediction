"""Regression tests for frequency packing and feature extraction."""

from unittest import TestCase
from unittest.mock import patch

import numpy as np
import pandas as pd

import Data_process as dp


class FeatureTransformTests(TestCase):
    """Check feature values, trimming, and MultiIndex labels."""

    def test_frequency_packing_dimensions_padding_and_transpose(self) -> None:
        signal = np.arange(1021) + 1j
        self.assertEqual(dp.adjust_rows_and_columns(signal), (16, 64))
        self.assertEqual(dp.adjust_rows_and_columns(np.empty(512)), (17, 31))
        self.assertEqual(dp.adjust_rows_and_columns(np.empty(4097)), (46, 90))
        packed = dp.reshape_carrier_frequency_vector(signal)
        self.assertEqual(packed.shape, (64, 16))
        self.assertEqual(packed.dtype, signal.dtype)
        np.testing.assert_array_equal(packed.T.ravel(), np.r_[0, signal, 0, 0])

    def test_singular_values_report_rank_one_variance(self) -> None:
        matrix = [[1, 3], [3, 1]]
        np.testing.assert_allclose(dp.sub_bandwidth_compute_singular_values(matrix),
                                   [1., 0.], atol=1e-14)

    def test_complex_covariance_returns_upper_triangle_in_row_order(self) -> None:
        matrix = np.array([[1, 3], [1j, 3j], [-1, -3]])
        a, b, c = 1 - 1j/3, 2j/3, -1 - 1j/3
        expected = [abs(a)**2, a*b.conjugate(), a*c.conjugate(),
                    abs(b)**2, b*c.conjugate(), abs(c)**2]
        result = dp.compute_lower_triangular_covariance(matrix)
        np.testing.assert_allclose(result, expected, atol=1e-14)
        self.assertEqual(result.dtype, np.dtype("complex128"))

    def test_row_feature_wrappers_pass_full_or_trimmed_matrix(self) -> None:
        matrix = np.arange(8 * 5).reshape(8, 5)
        row = pd.Series([1., 2.], index=pd.MultiIndex.from_product([["height_1"], [0, 1]]))
        cases = (
            (dp.row_height_full_singular_values, "sub_bandwidth_compute_singular_values",
             "var_explained", matrix),
            (dp.row_height_trimmed_singular_values, "sub_bandwidth_compute_singular_values",
             "var_explained", matrix[1:7]),
            (dp.row_height_full_covariance, "compute_lower_triangular_covariance",
             "cov_elem", matrix),
            (dp.row_height_trimmed_covariance, "compute_lower_triangular_covariance",
             "cov_elem", matrix[1:7]),
        )
        for transform, helper, prefix, expected_matrix in cases:
            with self.subTest(transform=transform.__name__):
                with patch.object(dp, "reshape_carrier_frequency_vector",
                                  return_value=matrix) as pack:
                    with patch.object(dp, helper, return_value=np.array([.75, .25])) as compute:
                        result, = transform(row, ["height_1"])
                pack.assert_called_once()
                pd.testing.assert_series_equal(pack.call_args[0][0], row["height_1"])
                compute.assert_called_once()
                np.testing.assert_array_equal(compute.call_args[0][0], expected_matrix)
                self.assertEqual(list(result.index),
                                 [("height_1", f"{prefix}_1"), ("height_1", f"{prefix}_2")])
                np.testing.assert_array_equal(result, [.75, .25])

    def test_remove_leading_variance_and_renormalize(self) -> None:
        index = pd.MultiIndex.from_product(
            [["height_1"], ["var_explained_1", "var_explained_2", "var_explained_3"]]
        )
        for values, expected in (([.6, .3, .1], [.75, .25]), ([1., 0., 0.], [0., 0.])):
            with self.subTest(values=values):
                result, = dp.row_height_drop_var_explained_1_and_renormalize(
                    pd.Series(values, index=index), ["height_1"]
                )
                self.assertEqual(list(result.index),
                                 [("height_1", "1_dropped_var_explained_2"),
                                  ("height_1", "1_dropped_var_explained_3")])
                np.testing.assert_allclose(result, expected)

    def test_pca_threshold_and_exclusion(self) -> None:
        columns = pd.MultiIndex.from_product([["height_1"], [0, 1, 2]])
        frame = pd.DataFrame(np.diag([5., 2., 1.]), columns=columns, index=[6, 7, 8])
        cases = ((dp.project_PCA_8_var, "8_pc_1", [5., 0., 0.]),
                 (dp.project_PCA_8_var_exclude_first, "8_pc_ex_1_1", [0., 2., 0.]))
        for transform, label, expected in cases:
            with self.subTest(transform=transform.__name__):
                result = transform(frame, ["height_1"])
                self.assertEqual(list(result.columns), [("height_1", label)])
                pd.testing.assert_index_equal(result.index, frame.index)
                np.testing.assert_allclose(np.abs(result.iloc[:, 0]), expected, atol=1e-14)

    def test_wavelet_second_order_coefficients_for_real_and_complex_signals(self) -> None:
        samples = np.arange(64, dtype=float)
        real = np.vstack([np.sin(samples / 3), np.cos(samples / 7)])
        imag = np.vstack([np.cos(samples / 5), np.sin(samples / 9)])
        columns = pd.MultiIndex.from_product([["height_1"], range(samples.size)])
        settings = dict(J=4, Q=4, max_order=2)
        scattering = dp.Scattering1D(shape=samples.size, **settings)
        second_order = scattering.meta()["order"] == 2
        self.assertTrue(second_order.any())

        def reference(values: np.ndarray) -> np.ndarray:
            """Compute the library reference for the normalized signals."""
            minimum = values.min(axis=1, keepdims=True)
            span = np.ptp(values, axis=1, keepdims=True)
            return scattering((values - minimum) / span)[:, second_order].reshape(2, -1)

        real_reference = reference(real)
        imag_reference = reference(imag)
        for values, channels in ((real, ("",)), (real + 1j * imag, ("real_", "imag_"))):
            with self.subTest(channels=channels):
                result = dp.height_wavelet_transform(
                    pd.DataFrame(values, columns=columns), ["height_1"], **settings
                )
                expected = real_reference
                if len(channels) == 2:
                    expected = np.hstack([real_reference, imag_reference])
                self.assertEqual(list(result.columns),
                                 [("height_1", f"scattering_order2_{channel}coeff_{i}")
                                  for channel in channels
                                  for i in range(real_reference.shape[1])])
                self.assertEqual(result.columns.names, ["height_column", "scattering_coeff"])
                self.assertEqual(result.to_numpy().dtype, expected.dtype)
                np.testing.assert_allclose(result, expected, rtol=1e-7, atol=1e-12)
