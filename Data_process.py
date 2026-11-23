import math

import pandas as pd
import numpy as np
from kymatio import Scattering1D
from scipy.signal import savgol_filter


def z_score_normalize_column(data):
    mean_cols = center_array_columns(data)
    std_cols = data.std(axis=0)
    return (data - mean_cols) / std_cols

def center_array_columns(data):
   return data - data.mean(axis=0)

def power_convsersion(A):
    return np.abs(A)
def min_max_normalize_rows(arr):
    """
    Perform min-max normalization over the rows of a NumPy array.

    Parameters:
    arr (numpy.ndarray): Input array where each row will be normalized.

    Returns:
    numpy.ndarray: Normalized array with values scaled between 0 and 1.
    """
    row_min = arr.min(axis=1, keepdims=True)
    row_max = arr.max(axis=1, keepdims=True)

    normalized = (arr - row_min) / (row_max - row_min)

    return normalized
def log_transform_complex(complex_array):
    """
    Apply a logarithmic transformation to the magnitudes of complex numbers.

    :param complex_array: Input array of complex numbers
    :return: Log-transformed complex array
    """
    amplitudes = np.abs(complex_array)

    log_amplitudes = np.log1p(amplitudes)

    phase = np.angle(complex_array)
    log_transformed_complex_array = log_amplitudes * np.exp(1j * phase)

    return log_transformed_complex_array


def z_score_normalize_complex(complex_array):
    """
    Apply Z-score normalization to the magnitudes of complex numbers.

    :param complex_array: Input array of complex numbers
    :return: Z-score normalized complex array
    """
    amplitudes = np.abs(complex_array)

    mean_amplitude = np.mean(amplitudes)
    std_amplitude = np.std(amplitudes)

    if std_amplitude == 0:
        return complex_array - mean_amplitude  # If all values are the same, just subtract the mean

    z_normalized_amplitudes = (amplitudes - mean_amplitude) / std_amplitude

    phase = np.angle(complex_array)
    z_normalized_complex_array = z_normalized_amplitudes * np.exp(1j * phase)

    return z_normalized_complex_array

def process_row_h(row, rounded_Num_of_Bins, rounded_Samples_per_Bin):
    """
    Processes a single row to compute the impulse response h(t).
    Parameters:
        row (array-like): A single row of complex values representing the impulse response.
        rounded_Num_of_Bins (int): Number of bins to divide the row into.
        rounded_Samples_per_Bin (int): Number of samples per bin.
    Returns:
        list: Binned average impulse response values (h(t)).
    """
    row = np.asarray(row)  # Ensure input is a NumPy array
    reshaped = row[:rounded_Num_of_Bins * rounded_Samples_per_Bin].reshape(rounded_Num_of_Bins, rounded_Samples_per_Bin)
    return np.mean(reshaped, axis=1)

def process_row_pdp(h_row):
    """
    Processes a row of h(t) to compute the Power Delay Profile (PDP).
    Parameters:
        h_row (list): A row of impulse response values h(t).
    Returns:
        list: Power Delay Profile (PDP) values for the given row.
    """
    h_row = np.asarray(h_row)  # Ensure input is a NumPy array
    return np.abs(h_row) ** 2

def compute_kth_block_column_difference(A, k = 2):
    """
    Compute the difference between every k-th column and its predecessor in array A.

    Parameters:
    - A (np.ndarray): Input array of shape (m, n)
    - k (int): The stride for selecting columns

    Returns:
    - difference (np.ndarray): Array containing the differences, shape (m, min(n//k - 1, ...))
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
    """
    Compute the difference between the (i + k)-th and i-th columns of array A.

    Parameters:
    - A (np.ndarray): Input array of shape (m, n)
    - k (int): The interval for difference

    Returns:
    - difference (np.ndarray): Array of shape (m, n - k)
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
    """
    Compute the first derivative of each row in array A using central differences
    with second-order accuracy by subsetting the data with step size h.
    Handles boundaries using forward and backward differences.

    Parameters:
    - A (np.ndarray): Input array of shape (m, n)
    - h (float): Step size between columns in the independent variable
    - k (int): The stride for selecting columns (must be >=1)

    Returns:
    - derivative_full (np.ndarray): Array containing the first derivatives,
                                    shape (m, n_subset)
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
    """
    Compute the second derivative of each row in array A using central differences
    with second-order accuracy by subsetting the data with step size h.
    Handles boundaries using forward and backward differences.

    Parameters:
    - A (np.ndarray): Input array of shape (m, n)
    - h (float): Step size between columns in the independent variable
    - k (int): The stride for selecting columns (must be >=1)

    Returns:
    - second_derivative_full (np.ndarray): Array containing the second derivatives,
                                           shape (m, n_subset)
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
    """
    Apply a Savitzky-Golay filter to each row of a 2D NumPy array.

    Parameters:
    - A (np.ndarray): Input array of shape (m, n)
    - window_length (int): Length of the filter window (must be odd and >= polyorder + 2)
    - polyorder (int): Order of the polynomial fit

    Returns:
    - smoothed_A (np.ndarray): Smoothed array of shape (m, n)
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
    # will always reshape it to have between 16 and 64 number of rows, so same number of singular values
    # only for subbandwidth problem, will not work with other vectors
    # dont use without talking to vincent or something will go wrong
    # goal is to alwasy makesure each sub bandwidth is its own feature and is of sufficient resolusion
   # while at the same time, not overalping with another feature like when we do first and second derivitves same time
    num_rows, num_cols = adjust_rows_and_columns(vector)


    required_length = num_rows * num_cols

    total_padding = required_length - len(vector)
    pad_left = total_padding // 2
    pad_right = total_padding - pad_left

    padded_vector = np.pad(vector, (pad_left, pad_right), mode='constant')

    reshaped_matrix = padded_vector.reshape(num_rows, num_cols).T

    return reshaped_matrix


def sub_bandwidth_compute_singular_values(matrix):
    """
    Compute the singular values of a normalized matrix and return the proportion of variance explained.

    :param matrix: Input matrix (numpy array, real or complex)
    :return: Proportion of variance explained by each singular value
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
    """
    Compute the lower triangular covariance matrix from a complex input array,
    using the Hermitian (conjugate) transpose.

    :param matrix: Input matrix (numpy array, real or complex)
    :return: Lower triangular covariance matrix
    """
    matrix = np.asarray(matrix, dtype=np.complex128)

    normalized_matrix = np.apply_along_axis(z_score_normalize_complex, axis=1, arr=matrix)

    matrix_centered = center_array_columns(normalized_matrix)

    matrix_hermitian = np.conjugate(matrix_centered.T)

    covariance_matrix = np.dot(matrix_centered, matrix_hermitian) / (matrix_centered.shape[0] - 1)

    return covariance_matrix[np.triu_indices_from(covariance_matrix)]
def row_height_full_singular_values(row, height_columns):

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
    """
    Custom row-wise transformation function to compute singular values for the middle 75%
    of rows from the height columns.

    :param row: Pandas Series representing a row of data
    :param height_columns: List of height columns to transform
    :return: List of transformed height data
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
    """
    Custom row-wise transformation function to compute the lower triangular covariance matrix
    for the full data from the height columns.

    :param row: Pandas Series representing a row of data
    :param height_columns: List of height columns to transform
    :return: List of transformed height data
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
    """
    Custom row-wise transformation function to compute the lower triangular covariance matrix
    for the middle 75% of rows from the height columns.

    :param row: Pandas Series representing a row of data
    :param height_columns: List of height columns to transform
    :return: List of transformed height data
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
    """
    Applies h(t) transformation to a group of data columns.
    data must be in the time domain
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
    """
    Applies PDP transformation to a group of data columns.
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
    """
    Drop 'var_explained_1' and renormalize the remaining variance explained columns.

    :param row: Pandas Series representing a row of data
    :param height_columns: List of height columns to process.
    :return: List of renormalized height data without 'var_explained_1'.
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
    """
    Apply a scattering wavelet transform to each column in the group,
    extracting only second-order scattering coefficients and labeling them
    based on their origin (real or imaginary part).

    :param group: Pandas DataFrame representing the group of data.
    :param height_columns: List of height columns to process.
    :param J: Maximum log-scale for scattering transform.
    :param Q: Number of wavelets per octave.
    :param max_order: Maximum order of scattering coefficients to compute (set to 2).
    :return: DataFrame with second-order scattering wavelet-transformed height data.
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
    """
    Transform height columns in a group using a column-wise transformation function.

    :param group: Input group (pandas DataFrame with multi-level columns)
    :param transform_function: Custom function to apply column-wise transformations.
    :return: Transformed group with updated height columns.
    """
    first_level_columns = set(group.columns.get_level_values(0))
    height_columns = sorted([col for col in first_level_columns if col.startswith('height')])

    non_height_columns = [col for col in group.columns if col[0] not in height_columns]

    transformed_data = transform_function(group, height_columns)

    non_height_data = group[non_height_columns].reset_index(drop=True)

    transformed_group = pd.concat([transformed_data, non_height_data], axis=1)

    return transformed_group


def transform_height_columns_row_wise(group, transform_function):
    """
    Transform height columns in a group using a row-wise transformation function.

    :param group: Input group (pandas DataFrame with multi-level columns)
    :param transform_function: Custom function to apply row-wise transformations.
    :return: Transformed group with updated height columns.
    """
    first_level_columns = set(group.columns.get_level_values(0))
    height_columns = sorted([col for col in first_level_columns if col.startswith('height')])

    non_height_columns = [col for col in group.columns if col[0] not in height_columns]

    def transform_row(row):
        row_series = pd.Series(row, index=group.columns)
        non_height_part = row_series[non_height_columns]

        height_transformed_data = transform_function(row_series, height_columns)

        transformed_row = pd.concat([pd.concat(height_transformed_data), pd.Series(non_height_part)])
        return transformed_row

    transformed_group = group.apply(transform_row, axis=1)

    return transformed_group


def transform_location_wise_height_row_group(df, transform_function_list):
    """
    Apply multiple height-wise row-wise transformation functions sequentially to the DataFrame grouped by 'meta' and 'location'.

    :param df: Input DataFrame with multi-level columns.
    :param transform_function_list: List of height-wise row-wise transformation functions to be applied sequentially.
    :return: Transformed DataFrame.
    """
    for transform_function in transform_function_list:
        df = df.groupby(('meta', 'location')).apply(
            lambda group: transform_height_columns_row_wise(group, transform_function)
        )
        df = df.reset_index(drop=True)
    return df



def transform_location_wise_height_group(df, transform_function_list):
    """
    Apply multiple height-wise transformation functions sequentially to the DataFrame grouped by 'meta' and 'location',
    performing column-wise transformations for each height column within each group.

    :param df: Input DataFrame with multi-level columns.
    :param transform_function_list: List of height-wise transformation functions to be applied sequentially.
    :return: Transformed DataFrame.
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


