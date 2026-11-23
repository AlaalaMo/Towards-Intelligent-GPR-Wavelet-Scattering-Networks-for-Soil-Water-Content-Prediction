import math

import pandas as pd
import numpy as np
from kymatio import Scattering1D
from scipy.signal import savgol_filter


def z_score_normalize_column(data):
    """Divide each column mean by the corresponding column standard deviation.

    The function subtracts the centered array from the original, yielding the
    column means up to floating-point rounding. Zero standard deviations follow
    NumPy division behavior.

    Args:
        data (np.ndarray): Numeric data with observations along axis 0.

    Returns:
        np.ndarray: Ratios broadcast to the shape of data.
    """
    mean_cols = center_array_columns(data)
    std_cols = data.std(axis=0)
    return (data - mean_cols) / std_cols

def center_array_columns(data):
   """Subtract the mean along axis 0 from each observation.

   Args:
       data (np.ndarray): Numeric data with observations along axis 0.

   Returns:
       np.ndarray: Mean-centered data with the same shape as data.
   """
   return data - data.mean(axis=0)

def power_convsersion(A):
    """Compute elementwise magnitudes.

    Args:
        A (array-like): Real or complex numeric samples.

    Returns:
        np.ndarray or scalar: Absolute values with the input shape.
    """
    return np.abs(A)
def min_max_normalize_rows(arr):
    """Scale each row by its own minimum and range.

    Constant rows have zero range and produce NaNs under default NumPy
    settings.

    Args:
        arr (np.ndarray): Two-dimensional real array shaped (rows, samples).

    Returns:
        np.ndarray: Same-shaped array with nonconstant finite rows scaled to
            [0, 1].
    """
    row_min = arr.min(axis=1, keepdims=True)
    row_max = arr.max(axis=1, keepdims=True)

    normalized = (arr - row_min) / (row_max - row_min)

    return normalized
def log_transform_complex(complex_array):
    """Compress complex magnitudes with log1p.

    Args:
        complex_array (np.ndarray): Complex samples of any shape.

    Returns:
        np.ndarray: Same-shaped complex array with log(1 + magnitude) and
            the original phase.
    """
    amplitudes = np.abs(complex_array)

    log_amplitudes = np.log1p(amplitudes)

    phase = np.angle(complex_array)
    log_transformed_complex_array = log_amplitudes * np.exp(1j * phase)

    return log_transformed_complex_array


def z_score_normalize_complex(complex_array):
    """Standardize magnitudes and recombine them with the original phasors.

    Magnitude statistics use all elements. Negative standardized magnitudes
    reverse the phasor direction. If the magnitude standard deviation is
    exactly zero, the mean magnitude is subtracted directly from each input
    sample.

    Args:
        complex_array (np.ndarray): Complex samples of any shape.

    Returns:
        np.ndarray: Same-shaped array of transformed samples.
    """
    amplitudes = np.abs(complex_array)

    mean_amplitude = np.mean(amplitudes)
    std_amplitude = np.std(amplitudes)

    if std_amplitude == 0:
        return complex_array - mean_amplitude  # Constant magnitudes make the z-score divisor zero.

    z_normalized_amplitudes = (amplitudes - mean_amplitude) / std_amplitude

    phase = np.angle(complex_array)
    z_normalized_complex_array = z_normalized_amplitudes * np.exp(1j * phase)

    return z_normalized_complex_array

def process_row_h(row, rounded_Num_of_Bins, rounded_Samples_per_Bin):
    """Average consecutive bins of impulse response samples.

    The row must contain at least the product of the two bin counts. Trailing
    samples beyond that product are discarded.

    Args:
        row (array-like): One-dimensional numeric samples in time order.
        rounded_Num_of_Bins (int): Number of bins to form.
        rounded_Samples_per_Bin (int): Number of consecutive samples per
            bin.

    Returns:
        np.ndarray: One mean sample per bin.
    """
    row = np.asarray(row)  # Ensure input is a NumPy array
    reshaped = row[:rounded_Num_of_Bins * rounded_Samples_per_Bin].reshape(rounded_Num_of_Bins, rounded_Samples_per_Bin)
    return np.mean(reshaped, axis=1)

def process_row_pdp(h_row):
    """Compute a power delay profile from impulse response samples.

    Args:
        h_row (array-like): One-dimensional real or complex impulse response
            samples.

    Returns:
        np.ndarray: Squared magnitudes in the original sample order.
    """
    h_row = np.asarray(h_row)  # Ensure input is a NumPy array
    return np.abs(h_row) ** 2

def compute_kth_block_column_difference(A, k = 2):
    """Difference adjacent columns after selecting every k-th column.

    Args:
        A (np.ndarray): Numeric array shaped (m, n).
        k (int): Stride for selecting columns, starting at column zero.

    Returns:
        np.ndarray: Later columns minus earlier columns, shaped (m, (n - 1)
            // k).

    Raises:
        TypeError: If A is not a NumPy array.
        ValueError: If A is not two-dimensional or k is not an integer in
            [1, n).
    """
    if not isinstance(A, np.ndarray):
        raise TypeError("Input A must be a NumPy array.")

    if A.ndim != 2:
        raise ValueError("Input A must be a 2D NumPy array.")

    m, n = A.shape

    if not isinstance(k, int) or k <= 0:
        raise ValueError("k must be a positive integer.")

    if k >= n:
        raise ValueError("k must be smaller than the number of columns in A.")

    selected_base = A[:, ::k]

    selected_shifted = A[:, k::k]

    min_cols = min(selected_base.shape[1], selected_shifted.shape[1])
    selected_base = selected_base[:, :min_cols]
    selected_shifted = selected_shifted[:, :min_cols]

    difference = selected_shifted - selected_base

    return difference


def compute_column_difference(A, k=2):
    """Difference all pairs of columns separated by k positions.

    Args:
        A (np.ndarray): Numeric array shaped (m, n).
        k (int): Column offset between paired samples.

    Returns:
        np.ndarray: Later columns minus earlier columns, shaped (m, n - k).

    Raises:
        TypeError: If A is not a NumPy array.
        ValueError: If A is not two-dimensional or k is not an integer in
            [1, n).
    """
    if not isinstance(A, np.ndarray):
        raise TypeError("Input A must be a NumPy array.")

    if A.ndim != 2:
        raise ValueError("Input A must be a 2D NumPy array.")

    m, n = A.shape

    if not isinstance(k, int) or k <= 0:
        raise ValueError("k must be a positive integer.")

    if k >= n:
        raise ValueError("k must be smaller than the number of columns in A.")

    return A[:, k:] - A[:, :-k]


def compute_first_derivative_central_with_boundaries_subset(A, h=1.0, k=1):
    """Estimate each row's first derivative after striding through its columns.

    Use centered differences divided by 2 * k in the interior and one-sided
    differences divided by k at the ends. At least 2 * k selected columns are
    needed for full boundary stencils. The scale is in selected-column units.
    Integer arrays can truncate fractional results.

    Args:
        A (np.ndarray): Numeric array shaped (rows, samples).
        h (int): Nonzero slice step for selecting columns. Supply an integer
            explicitly because the default 1.0 is not a valid slice step.
        k (int): Positive stencil offset in the selected columns.

    Returns:
        np.ndarray: Derivatives with the shape and dtype of A[:, ::h].

    Raises:
        TypeError: If A is not a NumPy array.
        ValueError: If A is not two-dimensional, k is not a positive
            integer, or k is at least the number of selected columns.
    """
    if not isinstance(A, np.ndarray):
        raise TypeError("Input A must be a NumPy array.")

    if A.ndim != 2:
        raise ValueError("Input A must be a 2D NumPy array.")

    m, n = A.shape

    if not isinstance(k, int) or k <= 0:
        raise ValueError("k must be a positive integer.")

    A_subset = A[:, ::h]
    m_subset, n_subset = A_subset.shape

    if k >= n_subset:
        raise ValueError("k must be smaller than the number of columns in the subsetted A.")

    derivative_full = np.zeros_like(A_subset)

    if 2 * k < n_subset:
        derivative_full[:, k:n_subset - k] = (A_subset[:, 2 * k:n_subset] - A_subset[:, 0:n_subset - 2 * k]) / (2 * k)

    derivative_full[:, :k] = (A_subset[:, k:2 * k] - A_subset[:, :k]) / (k)

    derivative_full[:, n_subset - k:] = (A_subset[:, n_subset - k:] - A_subset[:, n_subset - 2 * k:n_subset - k]) / (k)

    return derivative_full

def compute_second_derivative_central_with_boundaries_subset(A, h=1.0, k=1):
    """Estimate each row's second derivative after striding through its columns.

    Use centered second differences in the interior and one-sided second
    differences at the ends, dividing by k squared. At least 3 * k selected
    columns are needed for full boundary stencils. The scale is in
    selected-column units. Integer arrays can truncate fractional results.

    Args:
        A (np.ndarray): Numeric array shaped (rows, samples).
        h (int): Nonzero slice step for selecting columns. Supply an integer
            explicitly because the default 1.0 is not a valid slice step.
        k (int): Positive stencil offset in the selected columns.

    Returns:
        np.ndarray: Derivatives with the shape and dtype of A[:, ::h].

    Raises:
        TypeError: If A is not a NumPy array.
        ValueError: If A is not two-dimensional, k is not a positive
            integer, or 2 * k is at least the number of selected columns.
    """
    if not isinstance(A, np.ndarray):
        raise TypeError("Input A must be a NumPy array.")

    if A.ndim != 2:
        raise ValueError("Input A must be a 2D NumPy array.")

    m, n = A.shape

    if not isinstance(k, int) or k <= 0:
        raise ValueError("k must be a positive integer.")

    A_subset = A[:, ::h]
    m_subset, n_subset = A_subset.shape

    if 2 * k >= n_subset:
        raise ValueError("k is too large for the number of columns in the subsetted A.")

    second_derivative_full = np.zeros_like(A_subset)

    second_derivative_full[:, k:n_subset - k] = (A_subset[:, 2 * k:n_subset] - 2 * A_subset[:, k:n_subset - k] + A_subset[:, 0:n_subset - 2 * k]) / (k ** 2)

    second_derivative_full[:, :k] = (A_subset[:, 2 * k:k + 2 * k] - 2 * A_subset[:, k:2 * k] + A_subset[:, :k]) / (k ** 2)

    second_derivative_full[:, n_subset - k:] = (A_subset[:, n_subset - k:] - 2 * A_subset[:, n_subset - 2 * k:n_subset - k] + A_subset[:, n_subset - 3 * k:n_subset - 2 * k]) / (k ** 2)

    return second_derivative_full
def savgol_moving_average_per_row(A, window_length=5, polyorder=2):
    """Smooth each row with a Savitzky–Golay filter.

    The filter mirrors samples at the boundaries and processes real and
    imaginary parts separately for complex inputs.

    Args:
        A (np.ndarray): Real or complex array shaped (rows, samples).
        window_length (int): Positive odd filter window length.
        polyorder (int): Nonnegative polynomial degree smaller than the
            window length.

    Returns:
        np.ndarray: Smoothed samples with the same shape as A.

    Raises:
        TypeError: If A is not a NumPy array.
        ValueError: If A is not two-dimensional or the window length or
            polynomial degree fails the stated constraints.
    """
    if not isinstance(A, np.ndarray):
        raise TypeError("Input A must be a NumPy array.")

    if A.ndim != 2:
        raise ValueError("Input A must be a 2D NumPy array.")

    if not isinstance(window_length, int) or window_length < 1 or window_length % 2 == 0:
        raise ValueError("window_length must be a positive odd integer.")

    if not isinstance(polyorder, int) or polyorder < 0:
        raise ValueError("polyorder must be a non-negative integer.")

    if window_length <= polyorder:
        raise ValueError("window_length must be greater than polyorder.")

    if np.iscomplexobj(A):
        real_part = np.apply_along_axis(
            lambda m: savgol_filter(m, window_length=window_length, polyorder=polyorder, mode='mirror'),
            axis=1,
            arr=A.real
        )
        imag_part = np.apply_along_axis(
            lambda m: savgol_filter(m, window_length=window_length, polyorder=polyorder, mode='mirror'),
            axis=1,
            arr=A.imag
        )
        smoothed_A = real_part + 1j * imag_part
    else:
        smoothed_A = np.apply_along_axis(
            lambda m: savgol_filter(m, window_length=window_length, polyorder=polyorder, mode='mirror'),
            axis=1,
            arr=A
        )

    return smoothed_A


def adjust_rows_and_columns(vector, initial_num_cols=64):
    """Choose dimensions for packing a carrier-frequency vector.

    The row count is the ceiling of the vector length divided by the width. The
    width is resized by factors of sqrt(2), with integer truncation, to target
    16–64 rows.

    Args:
        vector (array-like): One-dimensional carrier-frequency samples. Use
            a length and initial width that allow a positive width and a row
            count in the target range.
        initial_num_cols (int): Positive starting width before the
            adjustments.

    Returns:
        tuple[int, int]: Row count and column count before transposition.
    """
    num_cols = initial_num_cols
    num_rows = np.ceil(len(vector) / num_cols).astype(int)

    while num_rows < 16:
        num_cols = int(num_cols / math.sqrt(2))  # Reduce columns
        num_rows = np.ceil(len(vector) / num_cols).astype(int)  # Recalculate rows

    while num_rows > 64:
        num_cols = int(num_cols * math.sqrt(2))  # Increase columns
        num_rows = np.ceil(len(vector) / num_cols).astype(int)  # Recalculate rows

    return num_rows, num_cols
def reshape_carrier_frequency_vector(vector):
    """Pack carrier-frequency samples into a matrix for sub-bandwidth features.

    The function zero-pads both ends to fill the matrix, placing any extra
    padding sample on the right.

    Args:
        vector (array-like): One-dimensional carrier-frequency samples
            compatible with the dimension selection starting at width 64.

    Returns:
        np.ndarray: Transposed matrix shaped (num_cols, num_rows) for the
            chosen packing dimensions.
    """
    num_rows, num_cols = adjust_rows_and_columns(vector)


    required_length = num_rows * num_cols

    total_padding = required_length - len(vector)
    pad_left = total_padding // 2
    pad_right = total_padding - pad_left

    padded_vector = np.pad(vector, (pad_left, pad_right), mode='constant')

    reshaped_matrix = padded_vector.reshape(num_rows, num_cols).T

    return reshaped_matrix


def sub_bandwidth_compute_singular_values(matrix):
    """Compute the variance fraction associated with each singular value.

    Convert to complex128, standardize each row with z_score_normalize_complex,
    and center each column before SVD. Divide each squared singular value by
    their sum. A zero sum produces undefined fractions.

    Args:
        matrix (array-like): Two-dimensional real or complex sub-bandwidth
            matrix.

    Returns:
        np.ndarray: One fraction per singular value, in descending
            singular-value order.
    """
    # Convert to complex if not already (this is a no-op if already complex)
    matrix = np.asarray(matrix, dtype=np.complex128)
    normalzied_z_score_length_complex_matrix = np.apply_along_axis(z_score_normalize_complex, axis = 1, arr = matrix)
    matrix_centered = center_array_columns(normalzied_z_score_length_complex_matrix)
    num_rows = matrix_centered.shape[0]
    singular_values = np.linalg.svd(matrix_centered, compute_uv=False)[:num_rows]


    total_variance = np.sum(singular_values ** 2)
    prop_variance_explained = (singular_values ** 2) / total_variance

    return prop_variance_explained


def compute_lower_triangular_covariance(matrix):
    """Return the upper triangle of the row covariance calculation, including its
    diagonal.

    Convert to complex128, standardize the rows, and center the columns.
    Multiply the centered matrix by its conjugate transpose and divide by the
    number of rows minus one.

    Args:
        matrix (array-like): Two-dimensional real or complex matrix with at
            least two rows for a nonzero divisor.

    Returns:
        np.ndarray: Flat array of upper-triangle entries in NumPy
            triangular-index order.
    """
    matrix = np.asarray(matrix, dtype=np.complex128)

    normalized_matrix = np.apply_along_axis(z_score_normalize_complex, axis=1, arr=matrix)

    matrix_centered = center_array_columns(normalized_matrix)

    matrix_hermitian = np.conjugate(matrix_centered.T)

    covariance_matrix = np.dot(matrix_centered, matrix_hermitian) / (matrix_centered.shape[0] - 1)

    return covariance_matrix[np.triu_indices_from(covariance_matrix)]
def row_height_full_singular_values(row, height_columns):
    """Extract singular-value variance fractions from all rows of each reshaped
    height signal.

    Each height signal is zero-padded and reshaped using
    reshape_carrier_frequency_vector.

    Args:
        row (pd.Series): One observation with each height selecting a
            one-dimensional carrier-frequency signal.
        height_columns (iterable of str): First-level height labels to
            process, in output order.

    Returns:
        list[pd.Series]: One series per height, indexed by (height,
            var_explained_N) with one-based feature numbering.
    """
    height_transformed_data = []
    for col in height_columns:
        vector = row[col]
        matrix = reshape_carrier_frequency_vector(vector)
        prop_variance = sub_bandwidth_compute_singular_values(matrix)

        new_data = pd.Series(prop_variance, index=pd.MultiIndex.from_product(
            [[col], [f'var_explained_{i + 1}' for i in range(len(prop_variance))]]))
        height_transformed_data.append(new_data)

    return height_transformed_data


def row_height_trimmed_singular_values(row, height_columns):
    """Extract singular-value variance fractions from the middle rows of each
    reshaped height signal.

    The retained rows start at int(0.125 * row_count) and stop before int(0.875
    * row_count) in the transposed signal matrix.

    Args:
        row (pd.Series): One observation with each height selecting a
            one-dimensional carrier-frequency signal.
        height_columns (iterable of str): First-level height labels to
            process, in output order.

    Returns:
        list[pd.Series]: One series per height, indexed by (height,
            var_explained_N) with one-based feature numbering.
    """
    height_transformed_data = []

    for col in height_columns:
        vector = row[col]
        matrix = reshape_carrier_frequency_vector(vector)

        lower_bound = int(0.125 * matrix.shape[0])  # Start at 12.5% of rows
        upper_bound = int(0.875 * matrix.shape[0])  # End at 87.5% of rows
        middle_matrix = matrix[lower_bound:upper_bound, :]

        prop_variance = sub_bandwidth_compute_singular_values(middle_matrix)

        new_data = pd.Series(prop_variance, index=pd.MultiIndex.from_product(
            [[col], [f'var_explained_{i + 1}' for i in range(len(prop_variance))]]))

        height_transformed_data.append(new_data)

    return height_transformed_data


def row_height_full_covariance(row, height_columns):
    """Extract upper-triangle covariance features from all rows of each reshaped
    height signal.

    Each height signal is zero-padded and reshaped using
    reshape_carrier_frequency_vector.

    Args:
        row (pd.Series): One observation with each height selecting a
            one-dimensional carrier-frequency signal.
        height_columns (iterable of str): First-level height labels to
            process, in output order.

    Returns:
        list[pd.Series]: One series per height, indexed by (height,
            cov_elem_N) with one-based feature numbering.
    """
    height_transformed_data = []

    for col in height_columns:
        vector = row[col]
        matrix = reshape_carrier_frequency_vector(vector)

        # Compute the lower triangular covariance matrix
        lower_triangular_covariance = compute_lower_triangular_covariance(matrix)

        new_data = pd.Series(lower_triangular_covariance, index=pd.MultiIndex.from_product(
            [[col], [f'cov_elem_{i + 1}' for i in range(len(lower_triangular_covariance))]]))

        height_transformed_data.append(new_data)

    return height_transformed_data
def row_height_trimmed_covariance(row, height_columns):
    """Extract upper-triangle covariance features from the middle rows of each
    reshaped height signal.

    The retained rows start at int(0.125 * row_count) and stop before int(0.875
    * row_count) in the transposed signal matrix.

    Args:
        row (pd.Series): One observation with each height selecting a
            one-dimensional carrier-frequency signal.
        height_columns (iterable of str): First-level height labels to
            process, in output order.

    Returns:
        list[pd.Series]: One series per height, indexed by (height,
            cov_elem_N) with one-based feature numbering.
    """
    height_transformed_data = []

    for col in height_columns:
        vector = row[col]
        n = 64
        matrix = reshape_carrier_frequency_vector(vector)

        # Trim to the middle 75% of rows
        lower_bound = int(0.125 * matrix.shape[0])  # Start at 12.5% of rows
        upper_bound = int(0.875 * matrix.shape[0])  # End at 87.5% of rows
        middle_matrix = matrix[lower_bound:upper_bound, :]

        # Compute the lower triangular covariance matrix for the middle rows
        lower_triangular_covariance = compute_lower_triangular_covariance(middle_matrix)

        new_data = pd.Series(lower_triangular_covariance, index=pd.MultiIndex.from_product(
            [[col], [f'cov_elem_{i + 1}' for i in range(len(lower_triangular_covariance))]]))

        height_transformed_data.append(new_data)

    return height_transformed_data


def column_center_by_loc(group, height_columns, prefix="centered_"):
    """Center each height sample across observations in the supplied group.

    Args:
        group (pd.DataFrame): Numeric signals with two-level columns
            (height, sample) and one signal per row.
        height_columns (iterable of str): First-level height labels to
            process, in output order.
        prefix (str): Text prepended to each second-level sample label.

    Returns:
        pd.DataFrame: Centered height columns with a fresh default row
            index.
    """
    transformed_data_list = []

    for col in height_columns:
        col_columns = group.loc[:, pd.IndexSlice[col, :]]
        height_data = np.vstack(col_columns.values)  # Shape: (num_rows, signal_length)
        height_data = height_data - height_data.mean(axis=0)

        center_height = pd.DataFrame(height_data, columns=col_columns.columns)
        new_columns = pd.MultiIndex.from_tuples(
            [(lvl1, prefix + str(lvl2)) for lvl1, lvl2 in center_height.columns]
        )
        center_height.columns = new_columns
        transformed_data_list.append(center_height)
    transformed_data = pd.concat(transformed_data_list, axis=1)

    return transformed_data
def column_difference_2(group, height_columns, prefix="2diff_"):
    """Difference all sample pairs 2 positions apart at each height.

    Each selected height must have more than 2 samples. Output labels use the
    leading original sample labels, truncated to the difference width.

    Args:
        group (pd.DataFrame): Numeric signals with two-level columns
            (height, sample) and one signal per row.
        height_columns (iterable of str): First-level height labels to
            process, in output order.
        prefix (str): Text prepended to each retained second-level sample
            label.

    Returns:
        pd.DataFrame: Difference features with a fresh default row index.
    """
    transformed_data_list = []

    for col in height_columns:
        col_columns = group.loc[:, pd.IndexSlice[col, :]]
        height_data = np.vstack(col_columns.values)  # Shape: (num_rows, signal_length)
        k=2
        height_data = compute_column_difference(height_data,k=k)

        df_transformed = pd.DataFrame(height_data, columns=col_columns.columns[:height_data.shape[1]])
        new_columns = pd.MultiIndex.from_tuples(
            [(lvl1, prefix + str(lvl2)) for lvl1, lvl2 in df_transformed.columns]
        )
        df_transformed.columns = new_columns
        transformed_data_list.append(df_transformed)
    transformed_data = pd.concat(transformed_data_list, axis=1)

    return transformed_data
def column_difference_16(group, height_columns, prefix="16diff_"):
    """Difference all sample pairs 16 positions apart at each height.

    Each selected height must have more than 16 samples. Output labels use the
    leading original sample labels, truncated to the difference width.

    Args:
        group (pd.DataFrame): Numeric signals with two-level columns
            (height, sample) and one signal per row.
        height_columns (iterable of str): First-level height labels to
            process, in output order.
        prefix (str): Text prepended to each retained second-level sample
            label.

    Returns:
        pd.DataFrame: Difference features with a fresh default row index.
    """
    transformed_data_list = []

    for col in height_columns:
        col_columns = group.loc[:, pd.IndexSlice[col, :]]
        height_data = np.vstack(col_columns.values)  # Shape: (num_rows, signal_length)
        k=16
        height_data = compute_column_difference(height_data,k=k)

        df_transformed = pd.DataFrame(height_data, columns=col_columns.columns[:height_data.shape[1]])
        new_columns = pd.MultiIndex.from_tuples(
            [(lvl1, prefix + str(lvl2)) for lvl1, lvl2 in df_transformed.columns]
        )
        df_transformed.columns = new_columns
        transformed_data_list.append(df_transformed)
    transformed_data = pd.concat(transformed_data_list, axis=1)

    return transformed_data
def column_difference_block_2(group, height_columns, prefix="b16diff_"):
    """Difference adjacent values after selecting every second sample at each
    height.

    Each selected height must have more than 2 samples. Output labels use the
    leading original sample labels, truncated to the difference width.

    Args:
        group (pd.DataFrame): Numeric signals with two-level columns
            (height, sample) and one signal per row.
        height_columns (iterable of str): First-level height labels to
            process, in output order.
        prefix (str): Text prepended to each retained second-level sample
            label.

    Returns:
        pd.DataFrame: Difference features with a fresh default row index.
    """
    transformed_data_list = []

    for col in height_columns:
        col_columns = group.loc[:, pd.IndexSlice[col, :]]
        height_data = np.vstack(col_columns.values)  # Shape: (num_rows, signal_length)
        k=2
        height_data = compute_kth_block_column_difference(height_data,k=k)

        df_transformed = pd.DataFrame(height_data, columns=col_columns.columns[:height_data.shape[1]])
        new_columns = pd.MultiIndex.from_tuples(
            [(lvl1, prefix + str(lvl2)) for lvl1, lvl2 in df_transformed.columns]
        )
        df_transformed.columns = new_columns
        transformed_data_list.append(df_transformed)
    transformed_data = pd.concat(transformed_data_list, axis=1)

    return transformed_data

def column_difference_block_16(group, height_columns, prefix="b16diff_"):
    """Difference adjacent values after selecting every sixteenth sample at each
    height.

    Each selected height must have more than 16 samples. Output labels use the
    leading original sample labels, truncated to the difference width.

    Args:
        group (pd.DataFrame): Numeric signals with two-level columns
            (height, sample) and one signal per row.
        height_columns (iterable of str): First-level height labels to
            process, in output order.
        prefix (str): Text prepended to each retained second-level sample
            label.

    Returns:
        pd.DataFrame: Difference features with a fresh default row index.
    """
    transformed_data_list = []

    for col in height_columns:
        col_columns = group.loc[:, pd.IndexSlice[col, :]]
        height_data = np.vstack(col_columns.values)  # Shape: (num_rows, signal_length)
        k=16
        height_data = compute_kth_block_column_difference(height_data,k=k)

        df_transformed = pd.DataFrame(height_data, columns=col_columns.columns[:height_data.shape[1]])
        new_columns = pd.MultiIndex.from_tuples(
            [(lvl1, prefix + str(lvl2)) for lvl1, lvl2 in df_transformed.columns]
        )
        df_transformed.columns = new_columns
        transformed_data_list.append(df_transformed)
    transformed_data = pd.concat(transformed_data_list, axis=1)

    return transformed_data

def numerical_first_second_1_h_1_k(group,height_columns):
    """Append first and second derivatives to each height's original samples.

    Smooth each row with a five-sample, degree-two Savitzky–Golay filter before
    differencing at unit stride and offset. Each height requires at least 3
    samples for the stencil. Use a default row index on group for positional
    alignment.

    Args:
        group (pd.DataFrame): Numeric signals with two-level columns
            (height, sample) and one signal per row.
        height_columns (iterable of str): First-level height labels to
            process, in output order.

    Returns:
        pd.DataFrame: Original and derivative columns combined by index.
            Derivative columns have a default row index and use fir_1_ or sec_1_
            prefixes.
    """
    transformed_data_list = []
    prefix = {'fir_1_':compute_first_derivative_central_with_boundaries_subset,'sec_1_':compute_second_derivative_central_with_boundaries_subset}
    for col in  height_columns:
        col_columns = group.loc[:, pd.IndexSlice[col, :]]
        transformed_data_list.append(col_columns)
        height_data = np.vstack(col_columns.values)  # Shape: (num_rows, signal_length)
        height_data = savgol_moving_average_per_row(height_data, window_length=5, polyorder=2)
        for pre, derivitive_func in prefix.items():
            height_der = derivitive_func(height_data, h=1, k=1)
            df_transformed = pd.DataFrame(height_der, columns=col_columns.columns[:height_der.shape[1]])
            new_columns = pd.MultiIndex.from_tuples(
                [(lvl1, pre + str(lvl2)) for lvl1, lvl2 in df_transformed.columns]
            )
            df_transformed.columns = new_columns
            transformed_data_list.append(df_transformed)
    transformed_data = pd.concat(transformed_data_list, axis=1)
    return transformed_data
def numerical_first_second_4_h_4_k(group,height_columns):
    """Append first and second derivatives to each height's original samples.

    Use every fourth sample with a finite-difference offset of four in the
    subset. The derivative labels use the leading source labels. Full boundary
    stencils require at least 12 selected samples per height. Use a default row
    index on group for positional alignment.

    Args:
        group (pd.DataFrame): Numeric signals with two-level columns
            (height, sample) and one signal per row.
        height_columns (iterable of str): First-level height labels to
            process, in output order.

    Returns:
        pd.DataFrame: Original and derivative columns combined by index.
            Derivative columns have a default row index and use fir_4_ or sec_4_
            prefixes.
    """
    transformed_data_list = []
    prefix = {'fir_4_':compute_first_derivative_central_with_boundaries_subset,'sec_4_':compute_second_derivative_central_with_boundaries_subset}
    for col in  height_columns:
        col_columns = group.loc[:, pd.IndexSlice[col, :]]
        transformed_data_list.append(col_columns)
        height_data = np.vstack(col_columns.values)  # Shape: (num_rows, signal_length)
        for pre, derivitive_func in prefix.items():
            height_der = derivitive_func(height_data, h=4, k=4)
            df_transformed = pd.DataFrame(height_der, columns=col_columns.columns[:height_der.shape[1]])
            new_columns = pd.MultiIndex.from_tuples(
                [(lvl1, pre + str(lvl2)) for lvl1, lvl2 in df_transformed.columns]
            )
            df_transformed.columns = new_columns
            transformed_data_list.append(df_transformed)
    transformed_data = pd.concat(transformed_data_list, axis=1)
    return transformed_data


def numerical_first_derivative_1_h_1_k(group, height_columns):
    """Append first derivatives to each height's original samples.

    Smooth each row with a five-sample, degree-two Savitzky–Golay filter before
    differencing at unit stride and offset. Each height requires at least 2
    samples for the stencil. Use a default row index on group for positional
    alignment.

    Args:
        group (pd.DataFrame): Numeric signals with two-level columns
            (height, sample) and one signal per row.
        height_columns (iterable of str): First-level height labels to
            process, in output order.

    Returns:
        pd.DataFrame: Original and derivative columns combined by index.
            Derivative columns have a default row index and use fir_1_ prefixes.
    """
    transformed_data_list = []
    for col in height_columns:
        col_columns = group.loc[:, pd.IndexSlice[col, :]]
        transformed_data_list.append(col_columns)
        height_data = np.vstack(col_columns.values)  # Shape: (num_rows, signal_length)
        height_data = savgol_moving_average_per_row(height_data, window_length=5, polyorder=2)

        # Compute first derivative
        height_der = compute_first_derivative_central_with_boundaries_subset(height_data, h=1, k=1)
        df_transformed = pd.DataFrame(height_der, columns=col_columns.columns[:height_der.shape[1]])
        new_columns = pd.MultiIndex.from_tuples(
            [(lvl1, 'fir_1_' + str(lvl2)) for lvl1, lvl2 in df_transformed.columns]
        )
        df_transformed.columns = new_columns
        transformed_data_list.append(df_transformed)

    transformed_data = pd.concat(transformed_data_list, axis=1)
    return transformed_data


def numerical_second_derivative_1_h_1_k(group, height_columns):
    """Append second derivatives to each height's original samples.

    Smooth each row with a five-sample, degree-two Savitzky–Golay filter before
    differencing at unit stride and offset. Each height requires at least 3
    samples for the stencil. Use a default row index on group for positional
    alignment.

    Args:
        group (pd.DataFrame): Numeric signals with two-level columns
            (height, sample) and one signal per row.
        height_columns (iterable of str): First-level height labels to
            process, in output order.

    Returns:
        pd.DataFrame: Original and derivative columns combined by index.
            Derivative columns have a default row index and use sec_1_ prefixes.
    """
    transformed_data_list = []
    for col in height_columns:
        col_columns = group.loc[:, pd.IndexSlice[col, :]]
        transformed_data_list.append(col_columns)
        height_data = np.vstack(col_columns.values)  # Shape: (num_rows, signal_length)
        height_data = savgol_moving_average_per_row(height_data, window_length=5, polyorder=2)

        # Compute second derivative
        height_der = compute_second_derivative_central_with_boundaries_subset(height_data, h=1, k=1)
        df_transformed = pd.DataFrame(height_der, columns=col_columns.columns[:height_der.shape[1]])
        new_columns = pd.MultiIndex.from_tuples(
            [(lvl1, 'sec_1_' + str(lvl2)) for lvl1, lvl2 in df_transformed.columns]
        )
        df_transformed.columns = new_columns
        transformed_data_list.append(df_transformed)

    transformed_data = pd.concat(transformed_data_list, axis=1)
    return transformed_data


def numerical_first_derivative_4_h_4_k(group, height_columns):
    """Append first derivatives to each height's original samples.

    Use every fourth sample with a finite-difference offset of four in the
    subset. The derivative labels use the leading source labels. Full boundary
    stencils require at least 8 selected samples per height. Use a default row
    index on group for positional alignment.

    Args:
        group (pd.DataFrame): Numeric signals with two-level columns
            (height, sample) and one signal per row.
        height_columns (iterable of str): First-level height labels to
            process, in output order.

    Returns:
        pd.DataFrame: Original and derivative columns combined by index.
            Derivative columns have a default row index and use fir_4_ prefixes.
    """
    transformed_data_list = []
    for col in height_columns:
        col_columns = group.loc[:, pd.IndexSlice[col, :]]
        transformed_data_list.append(col_columns)
        height_data = np.vstack(col_columns.values)  # Shape: (num_rows, signal_length)

        # Compute first derivative
        height_der = compute_first_derivative_central_with_boundaries_subset(height_data, h=4, k=4)
        df_transformed = pd.DataFrame(height_der, columns=col_columns.columns[:height_der.shape[1]])
        new_columns = pd.MultiIndex.from_tuples(
            [(lvl1, 'fir_4_' + str(lvl2)) for lvl1, lvl2 in df_transformed.columns]
        )
        df_transformed.columns = new_columns
        transformed_data_list.append(df_transformed)

    transformed_data = pd.concat(transformed_data_list, axis=1)
    return transformed_data


def numerical_second_derivative_4_h_4_k(group, height_columns):
    """Append second derivatives to each height's original samples.

    Use every fourth sample with a finite-difference offset of four in the
    subset. The derivative labels use the leading source labels. Full boundary
    stencils require at least 12 selected samples per height. Use a default row
    index on group for positional alignment.

    Args:
        group (pd.DataFrame): Numeric signals with two-level columns
            (height, sample) and one signal per row.
        height_columns (iterable of str): First-level height labels to
            process, in output order.

    Returns:
        pd.DataFrame: Original and derivative columns combined by index.
            Derivative columns have a default row index and use sec_4_ prefixes.
    """
    transformed_data_list = []
    for col in height_columns:
        col_columns = group.loc[:, pd.IndexSlice[col, :]]
        transformed_data_list.append(col_columns)
        height_data = np.vstack(col_columns.values)  # Shape: (num_rows, signal_length)

        # Compute second derivative
        height_der = compute_second_derivative_central_with_boundaries_subset(height_data, h=4, k=4)
        df_transformed = pd.DataFrame(height_der, columns=col_columns.columns[:height_der.shape[1]])
        new_columns = pd.MultiIndex.from_tuples(
            [(lvl1, 'sec_4_' + str(lvl2)) for lvl1, lvl2 in df_transformed.columns]
        )
        df_transformed.columns = new_columns
        transformed_data_list.append(df_transformed)

    transformed_data = pd.concat(transformed_data_list, axis=1)
    return transformed_data


def p_o_wer_conversion(group,height_columns,prefix = 'mag_'):
    """Extract sample magnitudes for each height.

    Args:
        group (pd.DataFrame): Numeric signals with two-level columns
            (height, sample) and one signal per row.
        height_columns (iterable of str): First-level height labels to
            process, in output order.
        prefix (str): Text prepended to each second-level sample label.

    Returns:
        pd.DataFrame: Magnitude features with a fresh default row index.
    """
    transformed_data_list = []

    for col in height_columns:
        col_columns = group.loc[:, pd.IndexSlice[col, :]]
        height_data = col_columns.values  # Shape: (num_rows, signal_length)
        height_data = np.apply_along_axis(power_convsersion, axis=0, arr=height_data)

        log_trans_data = pd.DataFrame(height_data, columns=col_columns.columns)
        new_columns = pd.MultiIndex.from_tuples(
            [(lvl1, prefix + str(lvl2)) for lvl1, lvl2 in log_trans_data.columns]
        )
        log_trans_data.columns = new_columns
        transformed_data_list.append(log_trans_data)
    transformed_data = pd.concat(transformed_data_list, axis=1)

    return transformed_data

def column_log_transform(group,height_columns, prefix='log_i_'):
    """Apply log1p magnitude compression to each height sample.

    Args:
        group (pd.DataFrame): Numeric signals with two-level columns
            (height, sample) and one signal per row.
        height_columns (iterable of str): First-level height labels to
            process, in output order.
        prefix (str): Text prepended to each second-level sample label.

    Returns:
        pd.DataFrame: Complex features with the original phases and a fresh
            default row index.
    """
    transformed_data_list = []

    for col in height_columns:
        col_columns = group.loc[:, pd.IndexSlice[col, :]]
        height_data = col_columns.values # Shape: (num_rows, signal_length)
        height_data = np.apply_along_axis(log_transform_complex, axis=0, arr=height_data)

        log_trans_data = pd.DataFrame(height_data, columns=col_columns.columns)
        new_columns = pd.MultiIndex.from_tuples(
            [(lvl1, prefix + str(lvl2)) for lvl1, lvl2 in log_trans_data.columns]
        )
        log_trans_data.columns = new_columns
        transformed_data_list.append(log_trans_data)
    transformed_data = pd.concat(transformed_data_list, axis=1)

    return transformed_data

def column_z_mag_transform(group,height_columns, prefix='z_mag'):
    """Standardize each height sample’s magnitudes across observations.

    Each sample column uses z_score_normalize_complex, including its
    zero-standard-deviation fallback.

    Args:
        group (pd.DataFrame): Numeric signals with two-level columns
            (height, sample) and one signal per row.
        height_columns (iterable of str): First-level height labels to
            process, in output order.
        prefix (str): Text prepended to each second-level sample label.

    Returns:
        pd.DataFrame: Transformed complex features with a fresh default row
            index.
    """
    transformed_data_list = []

    for col in height_columns:
        col_columns = group.loc[:, pd.IndexSlice[col, :]]
        height_data = col_columns.values  # Shape: (num_rows, signal_length)
        height_data = np.apply_along_axis(z_score_normalize_complex, axis=0, arr=height_data)

        center_height = pd.DataFrame(height_data, columns=col_columns.columns)
        new_columns = pd.MultiIndex.from_tuples(
            [(lvl1, prefix + str(lvl2)) for lvl1, lvl2 in center_height.columns]
        )
        center_height.columns = new_columns
        transformed_data_list.append(center_height)
    transformed_data = pd.concat(transformed_data_list, axis=1)

    return transformed_data

def h_t_transform(group, height_columns, bandwidth=16e8, receiption_time=109.863e-9, N=4096, Fs=30000, prefix='h(t)'):
    """Compute binned time-domain impulse responses for each height.

    Both the bin count and the samples per bin are rounded down. Rows must
    supply enough samples to fill the bins, with at least one sample per bin
    for finite bin means.

    Args:
        group (pd.DataFrame): Time-domain signals with two-level columns
            (height, sample) and one signal per row.
        height_columns (iterable of str): First-level height labels to
            process, in output order.
        bandwidth (float): Positive bandwidth in hertz. Its reciprocal is
            the bin duration.
        receiption_time (float): Reception window in seconds, long enough
            for at least one bin.
        N (int): Nominal sample count used to calculate samples per bin.
        Fs (float): Sampling-frequency parameter, unused by this
            implementation.
        prefix (str): Stem for the second-level bin labels.

    Returns:
        pd.DataFrame: Binned features with one-based suffixes and the
            original group index.
    """
    # Calculate bin parameters
    Bin_size = 1 / bandwidth
    Num_of_Bins = receiption_time / Bin_size
    rounded_Num_of_Bins = int(np.floor(Num_of_Bins))
    Samples_per_Bin = N / rounded_Num_of_Bins
    rounded_Samples_per_Bin = int(np.floor(Samples_per_Bin))

    transformed_data_list = []

    for col in height_columns:
        col_columns = group.loc[:, pd.IndexSlice[col, :]]
        height_data = col_columns.values

        height_data_transformed = np.array([
            process_row_h(row, rounded_Num_of_Bins, rounded_Samples_per_Bin)
            for row in height_data
        ])

        new_columns = pd.MultiIndex.from_tuples(
            [(col, f"{prefix}_{i + 1}") for i in range(height_data_transformed.shape[1])]
        )
        transformed_df = pd.DataFrame(height_data_transformed, columns=new_columns, index=group.index)
        transformed_data_list.append(transformed_df)

    transformed_data = pd.concat(transformed_data_list, axis=1)
    return transformed_data


def p_d_p_transform(group, height_columns, bandwidth=16e8, receiption_time=109.863e-9, N=4096, Fs=30000, prefix='pdp'):
    """Compute binned power delay profiles for each height.

    Both the bin count and the samples per bin are rounded down. Rows must
    supply enough samples to fill the bins, with at least one sample per bin
    for finite bin means. Each profile is the squared magnitude of the bin
    means.

    Args:
        group (pd.DataFrame): Time-domain signals with two-level columns
            (height, sample) and one signal per row.
        height_columns (iterable of str): First-level height labels to
            process, in output order.
        bandwidth (float): Positive bandwidth in hertz. Its reciprocal is
            the bin duration.
        receiption_time (float): Reception window in seconds, long enough
            for at least one bin.
        N (int): Nominal sample count used to calculate samples per bin.
        Fs (float): Sampling-frequency parameter, unused by this
            implementation.
        prefix (str): Stem for the second-level bin labels.

    Returns:
        pd.DataFrame: Binned features with zero-based suffixes and a default
            row index.
    """
    # Calculate bin parameters
    Bin_size = 1 / bandwidth
    Num_of_Bins = receiption_time / Bin_size
    rounded_Num_of_Bins = int(np.floor(Num_of_Bins))
    Samples_per_Bin = N / rounded_Num_of_Bins
    rounded_Samples_per_Bin = int(np.floor(Samples_per_Bin))

    transformed_data_list = []

    for col in height_columns:
        col_columns = group.loc[:, pd.IndexSlice[col, :]]
        height_data = col_columns.values  # Shape: (num_rows, signal_length)

        height_data_h = np.array([
            process_row_h(row, rounded_Num_of_Bins, rounded_Samples_per_Bin)
            for row in height_data
        ])

        height_data_pdp = [
            process_row_pdp(h_row) for h_row in height_data_h
        ]

        center_height = pd.DataFrame(height_data_pdp, columns=range(len(height_data_pdp[0])))
        new_columns = pd.MultiIndex.from_tuples(
            [(col, f"{prefix}_{i}") for i in center_height.columns]
        )
        center_height.columns = new_columns
        transformed_data_list.append(center_height)

    transformed_data = pd.concat(transformed_data_list, axis=1)
    return transformed_data
def project_PCA_8_var(group, height_columns, prefix='8_pc'):
    """Project each height onto components reaching an 80% cumulative variance
    threshold.

    Each height uses the eigenvectors of the uncentered Gram matrix X^H X,
    sorted by descending eigenvalue. The eigenvalues are normalized by their
    sum before selecting components. A meaningful threshold requires a positive
    eigenvalue sum.

    Args:
        group (pd.DataFrame): Numeric signals with two-level columns
            (height, sample) and one signal per row.
        height_columns (iterable of str): First-level height labels to
            process, in output order.
        prefix (str): Stem for the second-level component labels.

    Returns:
        pd.DataFrame: Component scores with the original group index and
            one-based component suffixes.
    """
    transformed_data_list = []

    for col in height_columns:
        col_columns = group.loc[:, pd.IndexSlice[col, :]]
        height_data = col_columns.values  # Shape: (num_rows, signal_length)

        covariance_matrix = np.dot(height_data.T.conj(), height_data)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        explained_variance_ratio = eigenvalues / np.sum(eigenvalues)
        cumulative_variance = np.cumsum(explained_variance_ratio)
        num_components = np.argmax(cumulative_variance >= 0.8) + 1
        selected_eigenvectors = eigenvectors[:, :num_components]
        transformed_data = np.dot(height_data, selected_eigenvectors)
        new_columns = pd.MultiIndex.from_tuples(
            [(col, f"{prefix}_{i + 1}") for i in range(transformed_data.shape[1])]
        )
        transformed_df = pd.DataFrame(transformed_data, columns=new_columns, index=group.index)
        transformed_data_list.append(transformed_df)

    transformed_group = pd.concat(transformed_data_list, axis=1)

    return transformed_group

def project_PCA_8_var_exclude_first(group, height_columns, prefix='8_pc_ex_1'):
    """Project each height onto leading components after dropping the dominant
    component.

    Each height uses the eigenvectors of the uncentered Gram matrix X^H X,
    sorted by descending eigenvalue. The leading eigenpair is dropped to
    suppress clutter. The 80% threshold is computed relative to the remaining
    eigenvalue sum. A meaningful threshold requires a positive eigenvalue sum
    after dropping the first component.

    Args:
        group (pd.DataFrame): Numeric signals with two-level columns
            (height, sample) and one signal per row.
        height_columns (iterable of str): First-level height labels to
            process, in output order.
        prefix (str): Stem for the second-level component labels.

    Returns:
        pd.DataFrame: Component scores with the original group index and
            one-based component suffixes.
    """
    transformed_data_list = []

    for col in height_columns:
        col_columns = group.loc[:, pd.IndexSlice[col, :]]
        height_data = col_columns.values  # Shape: (num_rows, signal_length)

        covariance_matrix = np.dot(height_data.T.conj(), height_data)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # exlcude first pc to handle clutter
        eigenvalues_excl_first = eigenvalues[1:]
        eigenvectors_excl_first = eigenvectors[:, 1:]

        total_variance_excl_first = np.sum(eigenvalues_excl_first)
        explained_variance_ratio_excl_first = eigenvalues_excl_first / total_variance_excl_first

        cumulative_variance = np.cumsum(explained_variance_ratio_excl_first)
        num_components = np.argmax(cumulative_variance >= 0.8) + 1

        selected_eigenvectors = eigenvectors_excl_first[:, :num_components]

        transformed_data = np.dot(height_data, selected_eigenvectors)

        new_columns = pd.MultiIndex.from_tuples(
            [(col, f"{prefix}_{i + 1}") for i in range(transformed_data.shape[1])]
        )

        transformed_df = pd.DataFrame(transformed_data, columns=new_columns, index=group.index)
        transformed_data_list.append(transformed_df)

    transformed_group = pd.concat(transformed_data_list, axis=1)

    return transformed_group
def row_height_drop_var_explained_1_and_renormalize(row, height_columns):
    """Remove the first variance feature and rescale the remaining fractions.

    Divide the remaining values by their sum when it is positive. Keep the
    remaining values as supplied otherwise.

    Args:
        row (pd.Series): One observation indexed by (height, feature), with
            numeric variance fractions.
        height_columns (iterable of str): First-level height labels to
            process, in output order.

    Returns:
        list[pd.Series]: One series per height with features relabeled
            1_dropped_var_explained_N starting at N=2.
    """
    height_transformed_data = []


    for col in height_columns:
        var_explained_data = row[col]

        remaining_variances = var_explained_data.loc[var_explained_data.index != 'var_explained_1'].values

        remaining_sum = sum(remaining_variances)

        if remaining_sum > 0:
            renormalized_variances = [val / remaining_sum for val in remaining_variances]
        else:
            renormalized_variances = remaining_variances

        new_data = pd.Series(renormalized_variances, index=pd.MultiIndex.from_product(
            [[col],
             [f'1_dropped_var_explained_{i + 2}' for i in range(len(renormalized_variances))]]))

        height_transformed_data.append(new_data)

    return height_transformed_data



def height_wavelet_transform(group, height_columns, J=6, Q=32, max_order=2):
    """Extract flattened second-order wavelet scattering features by height.

    Min-max scale each signal row before scattering. For complex data, scale
    and scatter the real and imaginary parts independently and place the real
    features first. Use nonconstant rows or parts for finite scaled values.

    Args:
        group (pd.DataFrame): Numeric signals with two-level columns
            (height, sample) and one signal per row.
        height_columns (iterable of str): First-level height labels to
            process, in output order.
        J (int): Maximum log-scale passed to Scattering1D.
        Q (int or tuple[int, int]): Wavelets per octave passed to
            Scattering1D.
        max_order (int): Maximum scattering order to compute. Use 2 to
            obtain second-order features.

    Returns:
        pd.DataFrame: Second-order features with (height_column,
            scattering_coeff) columns and a fresh default row index.
    """
    transformed_data_list = []

    for col in height_columns:
        height_data = group[col].values  # Shape: (num_rows, signal_length)
        # height_data = np.apply_along_axis(log_transform_complex, axis =0, arr =height_data)
        # height_data = np.apply_along_axis(z_score_normalize_complex, axis =0, arr = height_data)
        is_complex = np.iscomplexobj(height_data)

        signal_length = height_data.shape[1]

        scattering_transform = Scattering1D(J=J, shape=signal_length, Q=Q, max_order=max_order)

        meta = scattering_transform.meta()
        order2_indices = np.where(meta['order'] == 2)[0]

        if is_complex:
            real_part = min_max_normalize_rows(np.real(height_data))
            imag_part = min_max_normalize_rows(np.imag(height_data))

            real_coeffs = scattering_transform(real_part)  # Shape: (num_rows, num_coeffs)
            real_second_order = real_coeffs[:, order2_indices].reshape(height_data.shape[0], -1)  # Shape: (num_rows, num_order2_coeffs)

            imag_coeffs = scattering_transform(imag_part)  # Shape: (num_rows, num_coeffs)
            imag_second_order = imag_coeffs[:, order2_indices].reshape(height_data.shape[0], -1)  # Shape: (num_rows, num_order2_coeffs)

            combined_coeffs = np.hstack(
                [real_second_order, imag_second_order])  # Shape: (num_rows, 2 * num_order2_coeffs)

            num_real = real_second_order.shape[1]
            num_imag = imag_second_order.shape[1]

            first_level = [col] * (num_real + num_imag)
            second_level_real = [f'scattering_order2_real_coeff_{i}' for i in range(num_real)]
            second_level_imag = [f'scattering_order2_imag_coeff_{i}' for i in range(num_imag)]
            second_level = second_level_real + second_level_imag

            multi_index = pd.MultiIndex.from_tuples(zip(first_level, second_level),
                                                    names=['height_column', 'scattering_coeff'])
        else:
            height_data = min_max_normalize_rows(height_data)
            coeffs = scattering_transform(height_data)  # Shape: (num_rows, num_coeffs)
            combined_coeffs = coeffs[:, order2_indices].reshape(height_data.shape[0], -1)  # Shape: (num_rows, num_order2_coeffs)
            num_scatter = combined_coeffs.shape[1]
            first_level = [col] * num_scatter
            second_level = [f'scattering_order2_coeff_{i}' for i in range(num_scatter)]
            multi_index = pd.MultiIndex.from_tuples(zip(first_level, second_level),
                                                    names=['height_column', 'scattering_coeff'])
        coeffs_df = pd.DataFrame(combined_coeffs, columns=multi_index)

        transformed_data_list.append(coeffs_df)

    transformed_data = pd.concat(transformed_data_list, axis=1)

    return transformed_data



def transform_height_columns(group, transform_function):
    """Apply a callback to the height blocks and reattach other columns.

    Height labels start with height. Non-height columns receive a fresh default
    row index before concatenation.

    Args:
        group (pd.DataFrame): Observations with two-level columns and string
            first-level labels.
        transform_function (Callable): Accepts the group and sorted height
            labels and returns a feature DataFrame. Use a default row index for
            positional alignment.

    Returns:
        pd.DataFrame: Transformed features followed by non-height columns,
            concatenated by row index.
    """
    first_level_columns = set(group.columns.get_level_values(0))
    height_columns = sorted([col for col in first_level_columns if col.startswith('height')])

    non_height_columns = [col for col in group.columns if col[0] not in height_columns]

    transformed_data = transform_function(group, height_columns)

    non_height_data = group[non_height_columns].reset_index(drop=True)

    transformed_group = pd.concat([transformed_data, non_height_data], axis=1)

    return transformed_group


def transform_height_columns_row_wise(group, transform_function):
    """Apply a callback to each observation and reattach its non-height values.

    Height labels start with height.

    Args:
        group (pd.DataFrame): Observations with two-level columns and string
            first-level labels.
        transform_function (Callable): Accepts a row Series and sorted
            height labels. Returns a nonempty sequence of feature Series for
            concatenation.

    Returns:
        pd.DataFrame: One transformed observation per input row, with
            feature columns followed by original non-height columns.
    """
    first_level_columns = set(group.columns.get_level_values(0))
    height_columns = sorted([col for col in first_level_columns if col.startswith('height')])

    non_height_columns = [col for col in group.columns if col[0] not in height_columns]

    def transform_row(row):
        """Build one output row from height features and the other input values.

        Args:
            row (pd.Series): One observation supplied by DataFrame.apply and
                aligned to the enclosing group columns.

        Returns:
            pd.Series: Callback-generated height features followed by
                non-height values.
        """
        row_series = pd.Series(row, index=group.columns)
        non_height_part = row_series[non_height_columns]

        height_transformed_data = transform_function(row_series, height_columns)

        transformed_row = pd.concat([pd.concat(height_transformed_data), pd.Series(non_height_part)])
        return transformed_row

    transformed_group = group.apply(transform_row, axis=1)

    return transformed_group


def transform_location_wise_height_row_group(df, transform_function_list):
    """Apply row-wise height transforms within location groups.

    Regroup by location before each callback. An empty callback list returns
    the input frame.

    Args:
        df (pd.DataFrame): Observations with two-level columns including
            (meta, location).
        transform_function_list (iterable of Callable): Callbacks applied in
            order through transform_height_columns_row_wise.

    Returns:
        pd.DataFrame: Transformed observations with the row index reset
            after each callback.
    """
    for transform_function in transform_function_list:
        df = df.groupby(('meta', 'location')).apply(
            lambda group: transform_height_columns_row_wise(group, transform_function)
        )
        df = df.reset_index(drop=True)
    return df



def transform_location_wise_height_group(df, transform_function_list):
    """Apply column-wise height transforms within location groups.

    Group by location once and run all callbacks on each group before
    concatenation. Prints the input shape.

    Args:
        df (pd.DataFrame): Observations with two-level columns including
            (meta, location).
        transform_function_list (iterable of Callable): Callbacks applied in
            order through transform_height_columns.

    Returns:
        pd.DataFrame: Transformed observations concatenated with a fresh
            default row index.
    """
    print('shape of df',df.shape)
    grouped_df = df.groupby(('meta', 'location'))
    transformed_groups = []
    for name, group in grouped_df:
        for transform_function in transform_function_list:
            group = transform_height_columns(group, transform_function)
        transformed_groups.append(group)

    transformed_df = pd.concat(transformed_groups).reset_index(drop=True)

    return transformed_df


