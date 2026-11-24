"""Load and combine the original two-header-row frequency CSV files."""


import glob as glob
from argparse import Namespace
from typing import Optional, Sequence

import pandas as pd


from akela_shared.csv_loader import (
    append_meta_column as append_meta_column,
    confirm_x_columns_complex_and_check_nans as confirm_x_columns_complex_and_check_nans,
    convert_x_columns_to_complex as convert_x_columns_to_complex,
    get_csv_files as get_csv_files,
    load_and_process_csv as load_and_process_csv,
    load_csv_files as load_csv_files,
    print_column_dtypes as print_column_dtypes,
    remove_unwanted_columns as remove_unwanted_columns,
    rename_columns as rename_columns,
    rename_shed_y as rename_shed_y,
    rename_x_columns_based_on_height as rename_x_columns_based_on_height,
)


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


class CsvInputLoader:
    """Load raw CSVs using one preprocessing configuration.

    Args:
        args: Configuration supplying the domain for discovery and labels.
    """

    def __init__(self, args: Namespace) -> None:
        """Store the configuration without reading any files."""
        self._args = args

    def load(self) -> Optional[pd.DataFrame]:
        """Read and combine the configured domain's raw CSVs.

        Returns:
            Combined DataFrame in discovery order, or None for empty input
            or incompatible schemas.

        Raises:
            OSError: An input file cannot be read.
            ValueError: Signal conversion or CSV parsing fails.
            TypeError: The domain is invalid or a signal dtype check fails.
        """
        csv_files = get_csv_files(domain=self._args.domain)
        dataframes = load_csv_files(csv_files, self._args)
        return check_and_concatenate_dataframes(dataframes)


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
    return CsvInputLoader(args).load()
