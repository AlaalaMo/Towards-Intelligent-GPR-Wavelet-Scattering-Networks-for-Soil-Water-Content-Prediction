"""Write the frequency manifest and orchestrate dataset preprocessing.

main obtains ordered Namespace objects from preprocessing_config and
submits each to a worker. process_configuration resolves the three
transform fields with globals()[name], using the explicit imports from
Data_process below. An unknown name raises KeyError in that worker.
The argument namespace is used for dispatch and output naming. The group
helpers in Data_process pass a location-group DataFrame or row Series and
sorted height labels to each callback, leaving other arguments at their
function defaults.

For example, the wavelet-only configuration has height_wise_transforms set
to ['height_wavelet_transform'] and the other two transform fields set to
False. It uses Data_process.height_wavelet_transform with J=6, Q=32, and
max_order=2. To add a configuration, use the options in preprocessing_config
and ensure the exact callback name is explicitly imported here.
"""


import glob
import json
import multiprocessing
import os
import re
import sys
from argparse import Namespace
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from datetime import datetime

import pandas as pd

from akela_shared import csv_loader

from Data_process import transform_location_wise_height_group, transform_location_wise_height_row_group
# row wise and height wise
# transform_location_wise_height_row_group, apply firt two then optionally row_height_drop_var_explained_1_and_renormalize
from Data_process import row_height_full_singular_values, row_height_trimmed_singular_values, row_height_drop_var_explained_1_and_renormalize
from Data_process import row_height_trimmed_covariance, row_height_full_covariance
# location wise, all transform_location_wise_height_group
from Data_process import column_center_by_loc, column_log_transform, column_z_mag_transform, p_o_wer_conversion
from Data_process import numerical_first_second_1_h_1_k, numerical_first_second_4_h_4_k, column_difference_2, column_difference_16, column_difference_block_2, column_difference_block_16
from Data_process import numerical_first_derivative_1_h_1_k, numerical_first_derivative_4_h_4_k, numerical_second_derivative_1_h_1_k, numerical_second_derivative_4_h_4_k
# height wise then row wise, stand alone
# transform_location_wise_height_group
from Data_process import height_wavelet_transform
# height wise and location wise

# applied height wise and location wise, must center by loc
from Data_process import project_PCA_8_var, project_PCA_8_var_exclude_first
concat_flag = False
concat_df = None

from preprocessing_config import generate_configurations
from preprocessing_data_loader import load_initial_dataset


def check_height_column_types(df):
    # Loop through all first-level columns
    height_check = 0
    for col in df.columns.get_level_values(0).unique():
        # Only process columns that start with 'height'
        if col.startswith('height'):
            # Get the sub-dataframe for the current column (based on the first-level column name)
            sub_df = df.xs(col, axis=1, level=0)
            height_check = height_check +1
            # Get the unique data types in the sub-dataframe
            unique_types = sub_df.dtypes.unique()

            # Print the number of unique data types and an example value for each type
            print(f"First-level column: {col}, Unique types count: {len(unique_types)}")
            for dtype in unique_types:
                # Find an example value for each data type
                example_value = sub_df.select_dtypes(include=[dtype]).iloc[0, 0]
                print(f"Data type: {dtype}, Example value: {example_value}")

            # Check for errors
            if len(unique_types) > 2:
                raise ValueError(f"Error: More than 2 unique types found for column '{col}'.")

            if any(dtype == 'object' for dtype in unique_types):
                raise ValueError(f"Error: Object type found in column '{col}'.")

            # If no error, print the single type if all columns have the same type
            if len(unique_types) == 1:
                print(f"All columns in '{col}' are of a single type: {unique_types[0]}")
            else:
                print(f"Two different types found in '{col}', types: {unique_types}")
    if height_check !=3:
        raise ValueError(f"Error: Incorrect number of height columns.")

def abbreviate_transform_name(transform_name):
        parts = transform_name.split('_')
        abbreviation = ''.join([part[0] for part in parts])
        return abbreviation


def process_configuration(args: Namespace) -> None:
    """Apply one configured transform sequence to a copy of the loaded data.

    Args:
        args: Namespace containing the output location and transform names.

    Raises:
        KeyError: A configured name is absent from the module namespace.
        TypeError: An output height column contains object data.
        ValueError: The output height groups fail validation.

    Writes the transformed CSV and prints the existing diagnostics.
    """
    global concat_flag, concat_df
    location_wise_transforms_fucntions = []
    if hasattr(args, 'location_wise_transforms') and args.location_wise_transforms:
        location_wise_transforms_fucntions = [
            globals()[func_name] for func_name in args.location_wise_transforms
        ]
    height_wise_transform_functions = []
    if hasattr(args, 'height_wise_transforms') and args.height_wise_transforms:
        height_wise_transform_functions = [
            globals()[func_name] for func_name in args.height_wise_transforms
        ]

    # Prepare the list of height-wise row-wise transformation functions
    location_wise_height_wise_row_wise_transforms = []
    if hasattr(args, 'location_wise_height_wise_row_wise_transforms') and args.location_wise_height_wise_row_wise_transforms:
        location_wise_height_wise_row_wise_transforms = [
            globals()[func_name] for func_name in args.location_wise_height_wise_row_wise_transforms
        ]
    transform_args = {key: value for key, value in vars(args).items() if re.search(r'transforms$', key)}

    transform_names = '_'.join('_'.join(value) if isinstance(value, list) else str(value)
                               for key, value in transform_args.items())

    transform_names = transform_names if transform_names else 'unprocessed'

    print(transform_names)

    transform_names = transform_names.strip('_')  # Remove leading/trailing underscores if any

    output_dir = f'Final_data/{args.domain}/{args.save_location}/{abbreviate_transform_name(transform_names)}'



    concatenated_df = None
    concatenated_df = deepcopy(concat_df)

    if concatenated_df is not None:
        print("Processing completed successfully. Resulting DataFrame shape:", concatenated_df.shape)
    else:
        raise RuntimeError(
            f"Worker PID {os.getpid()} has no input DataFrame for {transform_names!r}. "
            f"Multiprocessing start method: {multiprocessing.get_start_method()}."
        )
    if location_wise_transforms_fucntions:
        concatenated_df = transform_location_wise_height_group(concatenated_df, location_wise_transforms_fucntions)
    if height_wise_transform_functions:
        concatenated_df = transform_location_wise_height_group(concatenated_df, height_wise_transform_functions)
    if location_wise_height_wise_row_wise_transforms:
        concatenated_df = transform_location_wise_height_row_group(concatenated_df,
                                                                   location_wise_height_wise_row_wise_transforms)
    concatenated_df = concatenated_df.rename(columns={('y', 'shed'): ('meta', 'shed'), ('y', 'site'): ('meta', 'site')})
    if os.path.exists(output_dir):
        csv_files = glob.glob(os.path.join(output_dir, "*.csv"))
        if csv_files:
            for csv_file in csv_files:
                df = pd.read_csv(csv_file, header=[0, 1])
                first_level_columns = df.columns.get_level_values(0).unique()
                for col in first_level_columns:
                    second_level_count = len(df.xs(col, axis=1, level=0).columns)
                    print(f"First-level column: {col}, Second-level columns count: {second_level_count}")

                # Call the function to check 'height' columns
    else:
        os.makedirs(output_dir, exist_ok=True)
    output_file = f'{output_dir}/transformed.csv'

    print(f'for Transforms: {transform_names}')
    first_level_columns = concatenated_df.columns.get_level_values(0).unique()
    for col in first_level_columns:
        second_level_count = len(concatenated_df.xs(col, axis=1, level=0).columns)
        print(f"First-level column: {col}, Second-level columns count: {second_level_count}")

        if col.startswith('height'):
            sub_df = concatenated_df.xs(col, axis=1, level=0)
            if any(sub_df.dtypes == object):
                raise TypeError(f"for Transforms: {transform_names}\n Output file attempt: {output_file}\n Column '{col}' contains string data, which is not allowed.")

    check_height_column_types(concatenated_df)
    print(f"Saving to: {output_file}",flush=True)
    concatenated_df.to_csv(output_file, index=False)


def parallel_main(arg: Namespace) -> None:
    """Run the configured task in a worker and propagate its exceptions.

    Args:
        arg: Configuration passed to process_configuration.
    """
    process_configuration(arg)


def main() -> None:
    """Run the configured dataset preprocessing pipeline."""
    global concat_flag, concat_df
    print(f"Working directory: {os.getcwd()}", flush=True)
    print(f"Python: {sys.executable} ({sys.version.split()[0]})", flush=True)
    print(f"CSV loader module: {csv_loader.__file__}", flush=True)
    configurations, configs_as_dict = generate_configurations()
    current_date = datetime.now().strftime('%Y%m%d')

    directory = os.path.join('Final_data', configurations[0].domain)

    os.makedirs(directory, exist_ok=True)

    filename = os.path.join(directory, f'{configurations[0].domain}_configs_{current_date}.json')

    with open(filename, 'w') as f:
        json.dump(configs_as_dict, f, indent=4)

    print("Configs saved to", filename)
    # to do 2nd order back scatter per row - done
    # make sure reshaping to sub bandwidth limits overlap of bandwiths of different features - done
    #singular value groups determined by nearest neightbor instead of sub bandwidths
    # GSVD between different locations to determine clutter
    concat_df = load_initial_dataset(configurations[0])
    if concat_df is None:
        raise RuntimeError(
            f"No usable input DataFrame for domain {configurations[0].domain!r}. "
            "See the CSV discovery and schema diagnostics above."
        )
    concat_flag = True

    # Parallelizing tasks using ProcessPoolExecutor
    print(f"Input shape: {concat_df.shape}. Running {len(configurations)} "
          f"configurations using {multiprocessing.get_start_method()!r}.",
          flush=True)
    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(parallel_main, arg): arg for arg in configurations}

        for future in as_completed(futures):
            try:
                # This will raise any exception that was raised inside parallel_main
                future.result()
            except Exception:
                print(f"Preprocessing failed for {futures[future]!r}",
                      file=sys.stderr, flush=True)
                raise


if __name__ == "__main__":
    main()
