"""Load and combine the original two-header-row frequency CSV files."""


import glob
import os
from argparse import Namespace
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd


def get_csv_files(
    data_folder_base: str = 'Final_data',
    domain: Optional[str] = None,
    process_method: str = 'unprocessed',
) -> List[str]:
    """Return CSV paths in the order returned by glob.

    Args:
        data_folder_base: Base directory for input data.
        domain: Domain directory. Supply a domain when calling this helper.
        process_method: Subdirectory within the domain.

    Returns:
        Matching CSV paths.

    Raises:
        TypeError: The domain is missing or is not a valid path component.
    """
    target_path = os.path.join(data_folder_base, domain, process_method)
    return glob.glob(os.path.join(target_path, '*.csv'))


def load_csv_files(
    csv_files: Sequence[str], args: Namespace
) -> List[pd.DataFrame]:
    """Load and clean each CSV in the given order.

    Args:
        csv_files: Paths to the input CSVs.
        args: Configuration supplying the domain for column labels.

    Returns:
        One DataFrame per input path.
    """
    dataframes = []
    for file in csv_files:
        df = load_and_process_csv(file, args)
        dataframes.append(df)
        print(f"Successfully loaded: {file}")
    return dataframes


def load_and_process_csv(file: str, args: Namespace) -> pd.DataFrame:
    """Read a two-header-row CSV and apply the existing input cleanup.

    Args:
        file: Input CSV path, also used as the location label.
        args: Configuration supplying the domain for column labels.

    Returns:
        The input frame with complex signals, height labels, and location.
    """
    df = pd.read_csv(file, header=[0, 1])
    df = convert_x_columns_to_complex(df)
    confirm_x_columns_complex_and_check_nans(df)
    df = rename_columns(df, args)
    df = remove_unwanted_columns(df, ['avgdepth5', 'avgdepth6'])
    df = append_meta_column(df, 'location', file)
    print_column_dtypes(df, 'y')
    return df


def convert_x_columns_to_complex(df):
    """Convert all columns with 'x' in the first level of the multi-index to complex numbers."""
    x_columns = df.columns.get_level_values(0) == 'x'
    for col in df.columns[x_columns]:
        df[col] = df[col].apply(lambda x: complex(x.replace(' ', '')) if isinstance(x, str) else x)
    if not all(df['x'].dtypes == np.complex128):
        raise TypeError("Not all columns in the DataFrame are of dtype complex128.")
    return df
def confirm_x_columns_complex_and_check_nans(df):
    x_columns = df.columns.get_level_values(0) == 'x'

    all_complex = True

    for col in df.columns[x_columns]:
        if not pd.api.types.is_complex_dtype(df[col]):
            print(f"Warning: Column {col} is not of complex dtype. Current dtype: {df[col].dtype}")
            all_complex = False

        if df[col].isnull().any():
            nan_count = df[col].isnull().sum()
            print(f"Note: Column {col} contains {nan_count} NaN values.")

    if all_complex:
        print("All 'x' level columns are confirmed to be of complex type.")
    else:
        print("Some 'x' level columns are not of complex type.")

    return all_complex
def rename_columns(df, args):
    """Rename columns in the DataFrame."""
    df = rename_shed_y(df)
    df = rename_x_columns_based_on_height(df, args)
    return df

def rename_shed_y(df):
    """Rename 'shed_y' to 'shed' in the second level of the MultiIndex columns."""
    if 'shed_y' in df.columns.get_level_values(1):
        df = df.rename(columns={'shed_y': 'shed'}, level=1)
        print("Renamed 'shed_y' to 'shed'")
    return df

def rename_x_columns_based_on_height(df, args):
    """Rename the first level 'x' columns based on heights."""
    new_column_names = []
    for col in df.columns:
        if col[0] == 'x':
            height_num = (df.columns.get_loc(col) // 4096) + 1
            new_col = (f'height_{height_num}', f'{args.domain}_{col[1]}') + col[2:]
        else:
            new_col = col
        new_column_names.append(new_col)
    df.columns = pd.MultiIndex.from_tuples(new_column_names, names=df.columns.names)
    return df

def remove_unwanted_columns(df, columns_to_remove):
    """Remove specific columns from the 'y' level of the DataFrame if they exist."""
    columns_to_drop = [('y', col) for col in columns_to_remove if col in df['y'].columns]
    if columns_to_drop:
        df = df.drop(columns=columns_to_drop)
        print(f"Dropped columns: {columns_to_drop}")
    return df

def append_meta_column(df, column_name, file_name):
    """Append a new column with the first index 'meta' and the second index as the file name."""
    df[('meta', column_name)] = file_name
    print(f"Appended meta column with file name: {file_name}")
    return df

def print_column_dtypes(df, level):
    """Print the dtypes of all columns for a specific level."""
    if level in df.columns.get_level_values(0):
        print(f"Data types of '{level}' index columns:")
        print(df[level].dtypes)
    else:
        print(f"No columns with '{level}' in the first level index.")

def check_and_concatenate_dataframes(
    df_list: Sequence[pd.DataFrame],
) -> Optional[pd.DataFrame]:
    """Concatenate frames after checking their columns and dtypes.

    Args:
        df_list: Input frames in discovery order.

    Returns:
        Frames stacked in input order with the first frame's column order,
        or None if the input is empty or the schemas disagree.
    """
    if not df_list:
        print("No DataFrames to concatenate.")
        return None

    reference_df = df_list[0]
    reference_columns = reference_df.columns
    reference_types = reference_df.dtypes

    for i, df in enumerate(df_list[1:], start=1):
        if set(df.columns) != set(reference_columns):
            print(f"Column mismatch found in DataFrame {i}:")
            print(f" - Columns in DataFrame {i} but not in reference: {set(df.columns) - set(reference_columns)}")
            print(f" - Columns in reference but not in DataFrame {i}: {set(reference_columns) - set(df.columns)}")
            return None

        for col in reference_columns:
            if df.dtypes[col] != reference_types[col]:
                print(f"Type mismatch in DataFrame {i}, column '{col}':")
                print(f" - {df.dtypes[col]} (current) vs {reference_types[col]} (reference)")
                return None

    print("All DataFrames have matching columns and data types. Proceeding to reorder columns and concatenate them row-wise.")
    reordered_dfs = [df[reference_columns] for df in df_list]
    concatenated_df = pd.concat(reordered_dfs, axis=0)
    print("Concatenation complete.")
    return concatenated_df

def load_initial_dataset(args: Namespace) -> Optional[pd.DataFrame]:
    """Load the input CSVs and combine them in file discovery order.

    Paths are relative to the working directory. The first discovered file
    supplies the reference column order. Loading prints the existing dtype
    and missing-value diagnostics.

    Args:
        args: Configuration namespace with the input domain.

    Returns:
        Combined two-level-column DataFrame, or None if there are no files
        or their columns or dtypes disagree.

    Raises:
        OSError: An input file cannot be read.
        ValueError: A signal string cannot be parsed as complex, or the CSV
            parser rejects the input.
        TypeError: A signal column fails the complex128 dtype check.
    """
    csv_files = get_csv_files(domain=args.domain)
    dataframes = load_csv_files(csv_files, args)
    return check_and_concatenate_dataframes(dataframes)
