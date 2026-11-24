import pandas as pd
import glob
import re
from concurrent.futures import ProcessPoolExecutor, as_completed

from datetime import datetime
import os
import json
import numpy as np
from argparse import Namespace
from copy import deepcopy
from Data_process import transform_location_wise_height_group, transform_location_wise_height_row_group
# row wise and height wise
# transform_location_wise_height_row_group, apply firt two then optionally row_height_drop_var_explained_1_and_renormalize
from Data_process import row_height_full_singular_values, row_height_trimmed_singular_values, row_height_drop_var_explained_1_and_renormalize
from Data_process import row_height_trimmed_covariance, row_height_full_covariance
# location wise, all transform_location_wise_height_group
from Data_process import column_center_by_loc, column_log_transform, column_z_mag_transform, p_o_wer_conversion
from Data_process import numerical_first_second_1_h_1_k, numerical_first_second_4_h_4_k, column_difference_2, column_difference_16, column_difference_block_2, column_difference_block_16
from Data_process import numerical_first_derivative_1_h_1_k, numerical_first_derivative_4_h_4_k, numerical_second_derivative_1_h_1_k, numerical_second_derivative_4_h_4_k
from Data_process import h_t_transform, p_d_p_transform
# height wise then row wise, stand alone
# transform_location_wise_height_group
from Data_process import height_wavelet_transform
# height wise and location wise

# applied height wise and location wise, must center by loc
from Data_process import project_PCA_8_var, project_PCA_8_var_exclude_first
concat_flag = False
concat_df = None

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

def check_and_concatenate_dataframes(df_list):
    """Check MultiIndex column types and concatenate DataFrames if they match."""
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

def abbreviate_transform_name(transform_name):
        parts = transform_name.split('_')
        abbreviation = ''.join([part[0] for part in parts])
        return abbreviation


def main(args):
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
    # if abbreviate_transform_name(transform_names) != "powcnfs1h1kccblcltczmtpP8vF":
    #     return



    concatenated_df = None
    concatenated_df = deepcopy(concat_df)

    if concatenated_df is not None:
        print("Processing completed successfully. Resulting DataFrame shape:", concatenated_df.shape)
    else:
        print("Processing failed due to DataFrame inconsistencies.")
        return
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



if __name__ == "__main__":
    # freq_configs = [
    #     Namespace(domain='freq_mean_sweep',
    #               save_location='up_to_may_2023',
    #               location_wise_transforms=['column_center_by_loc'],
    #               height_wise_transforms=False,
    #               location_wise_height_wise_row_wise_transforms=['row_height_trimmed_singular_values']),
    #
    #     # Namespace(domain='freq_mean_sweep',
    #     #           save_location='up_to_may_2023',
    #     #           location_wise_transforms=['column_center_by_loc'],
    #     #           height_wise_transforms=False,
    #     #           location_wise_height_wise_row_wise_transforms=['row_height_trimmed_singular_values',
    #     #                                                          'row_height_drop_var_explained_1_and_renormalize']),
    #     Namespace(domain='freq_mean_sweep',
    #               save_location='up_to_may_2023',
    #               location_wise_transforms=['column_center_by_loc'],
    #               height_wise_transforms=False,
    #               location_wise_height_wise_row_wise_transforms=['row_height_full_singular_values']),
    #     # Namespace(domain='freq_mean_sweep',
    #     #           save_location='up_to_may_2023',
    #     #           location_wise_transforms=['column_center_by_loc'],
    #     #           height_wise_transforms=False,
    #     #           location_wise_height_wise_row_wise_transforms=['row_height_full_covariance']),
    #     # Namespace(domain='freq_mean_sweep',
    #     #           save_location='up_to_may_2023',
    #     #           location_wise_transforms=['column_center_by_loc'],
    #     #           height_wise_transforms=False,
    #     #           location_wise_height_wise_row_wise_transforms=['row_height_trimmed_covariance']),
    #     # Namespace(domain='freq_mean_sweep',
    #     #           save_location='up_to_may_2023',
    #     #           location_wise_transforms=['column_center_by_loc'],
    #     #           height_wise_transforms=False,
    #     #           location_wise_height_wise_row_wise_transforms=['row_height_full_singular_values',
    #     #                                                          'row_height_drop_var_explained_1_and_renormalize']),
    #     # Namespace(domain='freq_mean_sweep',
    #     #           save_location='up_to_may_2023',
    #     #           location_wise_transforms=['column_center_by_loc'],
    #     #           height_wise_transforms=['project_PCA_8_var'],
    #     #           location_wise_height_wise_row_wise_transforms=False),
    #
    #     # Namespace(domain='freq_mean_sweep',
    #     #           save_location='up_to_may_2023',
    #     #           location_wise_transforms=['column_center_by_loc'],
    #     #           height_wise_transforms=['project_PCA_8_var_exclude_first'],
    #     #           location_wise_height_wise_row_wise_transforms=False),
    #
    #     Namespace(domain='freq_mean_sweep',
    #               save_location='up_to_may_2023',
    #               location_wise_transforms=['column_center_by_loc'],
    #               height_wise_transforms= False,
    #               location_wise_height_wise_row_wise_transforms=False),
    #     Namespace(domain='freq_mean_sweep',
    #               save_location='up_to_may_2023',
    #               location_wise_transforms=['column_center_by_loc'],
    #               height_wise_transforms=['height_wavelet_transform'],
    #               location_wise_height_wise_row_wise_transforms=False)
    # ]
    #
    # location_wise_transforms_options = [
    #     False,
    #     ['column_center_by_loc'],
    #     # ['column_center_by_loc', 'column_log_transform'],
    #     ['column_center_by_loc', 'column_log_transform', 'column_z_mag_transform'],
    #     # ['column_center_by_loc', 'column_z_mag_transform'],
    #     ['p_o_wer_conversion','column_center_by_loc'],
    #     ['p_o_wer_conversion', 'numerical_first_second_1_h_1_k','column_center_by_loc','column_log_transform', 'column_z_mag_transform'],
    #     ['numerical_first_second_1_h_1_k','column_center_by_loc','column_log_transform', 'column_z_mag_transform'],
    #     ['p_o_wer_conversion',  'column_difference_2','column_center_by_loc','column_log_transform', 'column_z_mag_transform'],
    #     ['p_o_wer_conversion',  'column_difference_16','column_center_by_loc','column_log_transform', 'column_z_mag_transform'],
    #     ['p_o_wer_conversion',  'column_difference_block_2','column_center_by_loc','column_log_transform', 'column_z_mag_transform'],
    #     ['column_difference_block_2','column_center_by_loc','column_log_transform', 'column_z_mag_transform'],
    #     ['p_o_wer_conversion',  'column_difference_block_16','column_center_by_loc','column_log_transform', 'column_z_mag_transform'],
    #     ['column_difference_block_16','column_center_by_loc','column_log_transform', 'column_z_mag_transform'],
    #     ['p_o_wer_conversion', 'numerical_first_second_4_h_4_k','column_center_by_loc','column_log_transform', 'column_z_mag_transform'],
    #     ['numerical_first_second_4_h_4_k', 'column_center_by_loc', 'column_log_transform', 'column_z_mag_transform']
    # ]


    def update_location_wise_transforms(location_wise_transforms_options):
        """
        Updates the location-wise transform options by expanding specific keywords into their respective
        first and second derivative options.

        Parameters:
            location_wise_transforms_options (list): A list of location-wise transform options, where each
                                                     option can be a list or another type.

        Returns:
            list: Updated list of location-wise transform options with expanded derivatives.
        """
        location_wise_transforms_options_updated = []

        for option in location_wise_transforms_options:
            if isinstance(option, list):
                location_wise_transforms_options_updated.append(option)

                if 'numerical_first_second_1_h_1_k' in option:
                    first_option = [
                        x if x != 'numerical_first_second_1_h_1_k' else 'numerical_first_derivative_1_h_1_k'
                        for x in option
                    ]
                    second_option = [
                        x if x != 'numerical_first_second_1_h_1_k' else 'numerical_second_derivative_1_h_1_k'
                        for x in option
                    ]
                    location_wise_transforms_options_updated.append(first_option)
                    location_wise_transforms_options_updated.append(second_option)

                elif 'numerical_first_second_4_h_4_k' in option:
                    first_option = [
                        x if x != 'numerical_first_second_4_h_4_k' else 'numerical_first_derivative_4_h_4_k'
                        for x in option
                    ]
                    second_option = [
                        x if x != 'numerical_first_second_4_h_4_k' else 'numerical_second_derivative_4_h_4_k'
                        for x in option
                    ]
                    location_wise_transforms_options_updated.append(first_option)
                    location_wise_transforms_options_updated.append(second_option)
            else:
                location_wise_transforms_options_updated.append(option)

        return location_wise_transforms_options_updated


    # location_wise_transforms_options_updated = update_location_wise_transforms(location_wise_transforms_options)


    def filter_non_center_options(location_wise_transforms_options):
        """
        Filters out 'column_center_by_loc' from any list in the location-wise transform options.

        Parameters:
            location_wise_transforms_options (list): A list of location-wise transform options, where each
                                                     option can be a list or another type.

        Returns:
            list: A list of filtered options with 'column_center_by_loc' removed from the lists that contain it.
        """
        non_center_filtered_options = []

        for option in location_wise_transforms_options:
            if isinstance(option, list) and 'column_center_by_loc' in option:
                # Remove 'column_center_by_loc' from the option list
                filtered_option = [transform for transform in option if transform != 'column_center_by_loc']
                non_center_filtered_options.append(filtered_option)

        return non_center_filtered_options


    # non_center_filtered_options_freq = filter_non_center_options(location_wise_transforms_options_updated)
    # location_wise_transforms_options_singular_value = [
    #
    #     ['p_o_wer_conversion', 'column_difference_block_2', 'column_center_by_loc', 'column_log_transform',
    #      'column_z_mag_transform'],
    #     ['column_difference_block_2', 'column_center_by_loc', 'column_log_transform', 'column_z_mag_transform'],
    #     ['p_o_wer_conversion', 'column_difference_block_16', 'column_center_by_loc', 'column_log_transform',
    #      'column_z_mag_transform'],
    #     ['column_difference_block_16', 'column_center_by_loc', 'column_log_transform', 'column_z_mag_transform'],
    #
    # ]

    def generate_variations(configs, transform_options):
        new_configs = []
        for config in configs:
            for transform_option in transform_options:
                new_config = deepcopy(config)
                new_config.location_wise_transforms = transform_option
                new_configs.append(new_config)
        return new_configs
    range_configs = [
        Namespace(domain='range_mean_sweep',
                  save_location='up_to_may_2023',
                  location_wise_transforms=['column_center_by_loc'],
                  height_wise_transforms=False,
                  location_wise_height_wise_row_wise_transforms=False),
    ]
    range_location_wise_transforms_options = [
        False,
        ['column_center_by_loc'],
        ['column_center_by_loc', 'column_log_transform', 'column_z_mag_transform'],
        ['h_t_transform','column_center_by_loc', 'column_log_transform', 'column_z_mag_transform'],
        ['p_d_p_transform','column_center_by_loc', 'column_log_transform', 'column_z_mag_transform'],
        ['p_o_wer_conversion', 'column_center_by_loc'],
        ['p_o_wer_conversion', 'numerical_first_second_1_h_1_k', 'column_center_by_loc', 'column_log_transform',
         'column_z_mag_transform'],
        ['h_t_transform','numerical_first_second_1_h_1_k', 'column_center_by_loc', 'column_log_transform', 'column_z_mag_transform'],
        ['h_t_transform','column_difference_2', 'column_center_by_loc', 'column_log_transform', 'column_z_mag_transform'],
        ['numerical_first_second_1_h_1_k', 'column_center_by_loc', 'column_log_transform', 'column_z_mag_transform'],
        ['p_d_p_transform', 'numerical_first_second_1_h_1_k', 'column_center_by_loc', 'column_log_transform', 'column_z_mag_transform'],
        ['p_d_p_transform', 'column_difference_2', 'column_center_by_loc', 'column_log_transform', 'column_z_mag_transform'],
        ['numerical_first_second_4_h_4_k', 'column_center_by_loc', 'column_log_transform', 'column_z_mag_transform']
       ]
    # all_configs = generate_variations(freq_configs, location_wise_transforms_options_updated)
    # non_center_configs = generate_variations(freq_configs,non_center_filtered_options_freq)

    location_wise_transforms_options_updated_range = update_location_wise_transforms(range_location_wise_transforms_options)
    all_configs_range = generate_variations(range_configs, location_wise_transforms_options_updated_range)
    non_center_filtered_options_range = filter_non_center_options(range_location_wise_transforms_options)
    non_center_configs_range = generate_variations(range_configs, non_center_filtered_options_range)
    def namespace_to_dict(namespace_obj):
        return vars(namespace_obj)

    # configs_as_dict = [namespace_to_dict(config) for config in (all_configs + non_center_configs)]
    configs_as_dict = [namespace_to_dict(config) for config in (all_configs_range + non_center_configs_range)]
    # configs_as_dict = configs_as_dict_range + configs_as_dict_freq

    def clean_filter_and_deduplicate_dicts(list_of_dicts):
        """
        Processes a list of dictionaries by:
        1. Replacing empty lists with `False`.
        2. Removing dictionaries containing any list with the substring 'wavelet'.
        3. Removing duplicate dictionaries.

        Args:
            list_of_dicts (list): List of dictionaries to process.

        Returns:
            list: Filtered, updated, and deduplicated list of dictionaries.
        """
        filtered_dicts = []

        for d in list_of_dicts:
            # Check if the dictionary should be filtered out
            should_filter = any(
                isinstance(value, list) and any('wavelet' in str(item) for item in value)
                for value in d.values()
            )
            if not should_filter:
                # Update empty lists to False
                for key, value in d.items():
                    if isinstance(value, list) and not value:
                        d[key] = False
                filtered_dicts.append(d)

        # Deduplicate dictionaries
        seen = set()
        unique_dicts = []
        for d in filtered_dicts:
            # Convert lists to tuples for hashing
            hashable_dict = {k: tuple(v) if isinstance(v, list) else v for k, v in d.items()}
            # Use frozenset of items for deduplication
            frozen = frozenset(hashable_dict.items())
            if frozen not in seen:
                seen.add(frozen)
                unique_dicts.append(d)

        return unique_dicts


    configs_as_dict = clean_filter_and_deduplicate_dicts(configs_as_dict)
    current_date = datetime.now().strftime('%Y%m%d')

    directory = os.path.join('Final_data', all_configs_range[0].domain)

    os.makedirs(directory, exist_ok=True)

    filename = os.path.join(directory, f'{all_configs_range[0].domain}_configs_{current_date}.json')

    with open(filename, 'w') as f:
        json.dump(configs_as_dict, f, indent=4)

    print("Configs saved to", filename)
    # to do 2nd order back scatter per row - done
    # make sure reshaping to sub bandwidth limits overlap of bandwiths of different features - done
    #singular value groups determined by nearest neightbor instead of sub bandwidths
    # GSVD between different locations to determine clutter
    csv_files = get_csv_files(domain=all_configs_range[1].domain)
    dataframes = load_csv_files(csv_files, all_configs_range[1])
    concat_df = check_and_concatenate_dataframes(dataframes)
    concat_flag = True
    main(all_configs_range[4])
    # main(all_configs[182])
    def parallel_main(arg):
        main(arg)  # Your main function without try-except block


    # Parallelizing tasks using ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(parallel_main, arg) for arg in (all_configs_range + non_center_configs_range)]

        for future in as_completed(futures):
            try:
                # This will raise any exception that was raised inside parallel_main
                future.result()
            except MemoryError as e:
                print(f"MemoryError encountered in task: {e}")
                raise e  # Re-raise if you want to propagate the error further
            except Exception as e:
                print(f"Other exception encountered: {e}")
                raise e  # Re-raise other exceptions as well