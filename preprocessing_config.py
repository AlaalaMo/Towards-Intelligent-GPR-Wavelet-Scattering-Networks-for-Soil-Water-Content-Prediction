"""Build the ordered frequency preprocessing configurations in memory.

Each Namespace has a domain, save location, and three ordered transform
fields. Each active list item is the exact name of a callback imported from
Data_process by preprocessing_main. False and empty lists skip a stage.
The orchestrator writes the existing JSON manifest for downstream readers.

To add a configuration, choose the stage appropriate to the callback
signature. Add a location transform sequence to the options or a new height
or row-wise template, then check the resulting combinations and output
abbreviations. Template location transforms are replaced during expansion.
Trace a configuration by reading its transform lists in stage order and
looking up each name in Data_process.
"""


from argparse import Namespace
from copy import deepcopy
from typing import Any, Dict, List, Tuple


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

def generate_variations(configs, transform_options):
    new_configs = []
    for config in configs:
        for transform_option in transform_options:
            new_config = deepcopy(config)
            new_config.location_wise_transforms = transform_option
            new_configs.append(new_config)
    return new_configs

def namespace_to_dict(namespace_obj):
    return vars(namespace_obj)

def clean_filter_and_deduplicate_dicts(list_of_dicts):
    """
    Processes a list of dictionaries by:
    Retains the first occurrence, including wavelet entries and empty lists.
    The historical filtering and normalization steps are commented out.

    Args:
        list_of_dicts (list): List of dictionaries to process.

    Returns:
        list: Filtered, updated, and deduplicated list of dictionaries.
    """
    filtered_dicts = []

    for d in list_of_dicts:
        # Check if the dictionary should be filtered out
        # should_filter = any(
        #     isinstance(value, list) and any('wavelet' in str(item) for item in value)
        #     for value in d.values()
        # )
        # if not should_filter:
        #     # Update empty lists to False
        #     for key, value in d.items():
        #         if isinstance(value, list) and not value:
        #             d[key] = False
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


def generate_configurations() -> Tuple[List[Namespace], List[Dict[str, Any]]]:
    """Build the original ordered task list and configuration manifest records.

    Returns:
        A pair containing the namespaces submitted to workers and the
        first-occurrence manifest records in their original order.
    """
    freq_configs = [
        # Namespace(domain='freq_mean_sweep',
        #           save_location='up_to_may_2023',
        #           location_wise_transforms=['column_center_by_loc'],
        #           height_wise_transforms=False,
        #           location_wise_height_wise_row_wise_transforms=['row_height_trimmed_singular_values']),

        # Namespace(domain='freq_mean_sweep',
        #           save_location='up_to_may_2023',
        #           location_wise_transforms=['column_center_by_loc'],
        #           height_wise_transforms=False,
        #           location_wise_height_wise_row_wise_transforms=['row_height_trimmed_singular_values',
        #                                                          'row_height_drop_var_explained_1_and_renormalize']),
        # Namespace(domain='freq_mean_sweep',
        #           save_location='up_to_may_2023',
        #           location_wise_transforms=['column_center_by_loc'],
        #           height_wise_transforms=False,
        #           location_wise_height_wise_row_wise_transforms=['row_height_full_singular_values']),
        # Namespace(domain='freq_mean_sweep',
        #           save_location='up_to_may_2023',
        #           location_wise_transforms=['column_center_by_loc'],
        #           height_wise_transforms=False,
        #           location_wise_height_wise_row_wise_transforms=['row_height_full_covariance']),
        # Namespace(domain='freq_mean_sweep',
        #           save_location='up_to_may_2023',
        #           location_wise_transforms=['column_center_by_loc'],
        #           height_wise_transforms=False,
        #           location_wise_height_wise_row_wise_transforms=['row_height_trimmed_covariance']),
        # Namespace(domain='freq_mean_sweep',
        #           save_location='up_to_may_2023',
        #           location_wise_transforms=['column_center_by_loc'],
        #           height_wise_transforms=False,
        #           location_wise_height_wise_row_wise_transforms=['row_height_full_singular_values',
        #                                                          'row_height_drop_var_explained_1_and_renormalize']),
        # Namespace(domain='freq_mean_sweep',
        #           save_location='up_to_may_2023',
        #           location_wise_transforms=['column_center_by_loc'],
        #           height_wise_transforms=['project_PCA_8_var'],
        #           location_wise_height_wise_row_wise_transforms=False),

        # Namespace(domain='freq_mean_sweep',
        #           save_location='up_to_may_2023',
        #           location_wise_transforms=['column_center_by_loc'],
        #           height_wise_transforms=['project_PCA_8_var_exclude_first'],
        #           location_wise_height_wise_row_wise_transforms=False),

        Namespace(domain='freq_mean_sweep',
                  save_location='up_to_may_2023',
                  location_wise_transforms=['column_center_by_loc'],
                  height_wise_transforms= False,
                  location_wise_height_wise_row_wise_transforms=False),
        Namespace(domain='freq_mean_sweep',
                  save_location='up_to_may_2023',
                  location_wise_transforms=False,
                  height_wise_transforms=['height_wavelet_transform'],
                  location_wise_height_wise_row_wise_transforms=False)
    ]

    location_wise_transforms_options = [
        False,
        ['column_center_by_loc'],
        ['column_center_by_loc', 'column_log_transform'],
        ['column_center_by_loc', 'column_log_transform', 'column_z_mag_transform'],
        ['column_center_by_loc', 'column_z_mag_transform'],
        ['p_o_wer_conversion','column_center_by_loc'],
        ['p_o_wer_conversion', 'numerical_first_second_1_h_1_k','column_center_by_loc','column_log_transform', 'column_z_mag_transform'],
        ['numerical_first_second_1_h_1_k','column_center_by_loc','column_log_transform', 'column_z_mag_transform'],
        ['p_o_wer_conversion',  'column_difference_2','column_center_by_loc','column_log_transform', 'column_z_mag_transform'],
        ['p_o_wer_conversion',  'column_difference_16','column_center_by_loc','column_log_transform', 'column_z_mag_transform'],
        # ['p_o_wer_conversion',  'column_difference_block_2','column_center_by_loc','column_log_transform', 'column_z_mag_transform'],
        # ['column_difference_block_2','column_center_by_loc','column_log_transform', 'column_z_mag_transform'],
        # ['p_o_wer_conversion',  'column_difference_block_16','column_center_by_loc','column_log_transform', 'column_z_mag_transform'],
        # ['column_difference_block_16','column_center_by_loc','column_log_transform', 'column_z_mag_transform'],
        # ['p_o_wer_conversion', 'numerical_first_second_4_h_4_k','column_center_by_loc','column_log_transform', 'column_z_mag_transform'],
        # ['numerical_first_second_4_h_4_k', 'column_center_by_loc', 'column_log_transform', 'column_z_mag_transform']
    ]


    location_wise_transforms_options_updated = update_location_wise_transforms(location_wise_transforms_options)


    non_center_filtered_options_freq = filter_non_center_options(location_wise_transforms_options_updated)
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


    all_configs = generate_variations(freq_configs, location_wise_transforms_options_updated)
    non_center_configs = generate_variations(freq_configs,non_center_filtered_options_freq)

    configs_as_dict = [namespace_to_dict(config) for config in (all_configs + non_center_configs)]
    # configs_as_dict = configs_as_dict_range + configs_as_dict_freq

    configs_as_dict = clean_filter_and_deduplicate_dicts(configs_as_dict)
    return all_configs + non_center_configs, configs_as_dict
