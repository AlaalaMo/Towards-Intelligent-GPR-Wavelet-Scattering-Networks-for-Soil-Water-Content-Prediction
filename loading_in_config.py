import argparse
import pandas as pd
import os
import json
import re
import numpy as np
import random
from sklearn.preprocessing import StandardScaler

def abbreviate_transform_name(transform_args):
    abbreviation = ''
    for key in ['location_wise_transforms', 'height_wise_transforms', 'location_wise_height_wise_row_wise_transforms']:
        value = transform_args.get(key)
        if value == False or value == 'False':
            abbreviation += 'F'
        elif isinstance(value, list):
            for transform_name in value:
                parts = transform_name.split('_')
                abbrev = ''.join([part[0] for part in parts])
                abbreviation += abbrev
        elif isinstance(value, str):
            parts = value.split('_')
            abbrev = ''.join([part[0] for part in parts])
            abbreviation += abbrev
    return abbreviation


def get_subset_columns(df: pd.DataFrame, height_column_type: str, location_type: str) -> pd.DataFrame:
    """
    Subset the DataFrame to include 'y' and height columns, handling complex data efficiently.
    Converts all complex numbers (including those stored as strings) to real and imaginary float columns.
    Preserves the original MultiIndex structure and reports any remaining problematic values.
    """
    if location_type.lower() == 'all':
        print('Using all locations')
    else:
        df = df[df[('meta', 'location')].str.contains(location_type)]
        print(f'Subsetted to location type: {location_type}')

    y_columns = df.loc[:,
                ((df.columns.get_level_values(0) == 'y') & df.columns.get_level_values(1).str.startswith('avg'))]
    meta_columns = df.loc[:, ((df.columns.get_level_values(0) == 'meta') & df.columns.get_level_values(
        1).str.startswith('location'))]

    for col in meta_columns.columns:
        unique_values = meta_columns[col].unique()
        print(f"Split CV folds on '{col}' unique values: {unique_values}")

    print("Ensure that the meta subset is correctly specified in loading config file.")

    if height_column_type.lower() == 'all':
        height_columns = df.loc[:, df.columns.get_level_values(0).str.startswith('height')]
        print('Using all heights')
    else:
        height_columns = df.loc[:, df.columns.get_level_values(0) == height_column_type]
        print(f'Subsetted to height type: {height_column_type}')

    def parse_complex(value):
        if pd.isna(value):
            return None
        if isinstance(value, complex):
            return value
        if isinstance(value, (int, float)):
            return complex(value)
        if isinstance(value, str):
            # Remove parentheses if present
            value = value.strip('()')
            try:
                return complex(value)
            except ValueError:
                # If complex() fails, try parsing with regex
                match = re.match(r'^([-+]?\d*\.?\d*)([+-])(\d*\.?\d*)j$', value)
                if match:
                    real = float(match.group(1) or 0)
                    imag = float(match.group(2) + (match.group(3) or '1'))
                    return complex(real, imag)
        return None

    problematic_values = {}

    def safe_parse_complex(value, col):
        result = parse_complex(value)
        if result is None:
            if col not in problematic_values:
                problematic_values[col] = []
            problematic_values[col].append(value)
        return result

    # Convert all height columns to complex
    height_columns = height_columns.apply(lambda col: col.map(lambda x: safe_parse_complex(x, col.name)))

    # Report problematic values
    if problematic_values:
        print("\nProblematic values detected:")
        for col, values in problematic_values.items():
            print(f"Column {col}:")
            for value in values[:10]:  # Limit to first 10 problematic values per column
                print(f"  {value}")
            if len(values) > 10:
                print(f"  ... and {len(values) - 10} more.")
        raise ValueError("Problematic values found. Please fix these issues in your data.")

    # Split complex columns into real and imaginary parts while preserving MultiIndex
    new_columns = []
    new_data = []

    for col in height_columns.columns:
        real_values = height_columns[col].apply(lambda x: x.real if x is not None else None)
        imag_values = height_columns[col].apply(lambda x: x.imag if x is not None else None)

        if isinstance(col, tuple):
            new_columns.extend([(col[0], f're_{col[1]}'), (col[0], f'im_{col[1]}')])
        else:
            new_columns.extend([(f're_{col}',), (f'im_{col}',)])

        new_data.extend([real_values, imag_values])

    height_columns_split = pd.DataFrame(dict(zip(new_columns, new_data)))

    # Ensure the new DataFrame has a MultiIndex
    if not isinstance(height_columns_split.columns, pd.MultiIndex):
        height_columns_split.columns = pd.MultiIndex.from_tuples(height_columns_split.columns)

    print("Converted complex columns to real and imaginary parts.")

    # Check for non-float columns
    non_float_columns = height_columns_split.columns[height_columns_split.dtypes != 'float64']

    if not non_float_columns.empty:
        problematic_values = height_columns_split[non_float_columns].head()
        raise TypeError(
            f"Error: The following columns are not of type float: {non_float_columns.tolist()}\n"
            f"Sample problematic values:\n{problematic_values}\n"
            f"Please check these columns and ensure all values can be converted to floats."
        )

    final_dataframe = pd.concat([meta_columns, y_columns, height_columns_split], axis=1)

    return final_dataframe


def get_transformed_data(config_file_location, config_index):

    with open(config_file_location, 'r') as f:
        config_list = json.load(f)

    if config_index < 0 or config_index >= len(config_list):
        print(f"Invalid config_index {config_index}. Must be between 0 and {len(config_list) - 1}.")
        exit(1)

    config = config_list[config_index]

    domain = config.get('domain')
    save_location = config.get('save_location')

    transform_args = {key: value for key, value in config.items() if re.search(r'transforms$', key)}
    abbreviation = abbreviate_transform_name(transform_args)
    print("transform_args:",transform_args)
    print(f"Generated abbreviation: {abbreviation}")

    output_dir = os.path.join('Final_data', domain, save_location, abbreviation)
    output_file = os.path.join(output_dir, 'transformed.csv')

    if not os.path.exists(output_file):
        print(f"Transformed data file does not exist: {output_file}")
        exit(1)

    df = pd.read_csv(output_file, header=[0, 1])

    print(f"Loaded DataFrame from {output_file}")
    print(f"DataFrame shape: {df.shape}")

    return df, config
