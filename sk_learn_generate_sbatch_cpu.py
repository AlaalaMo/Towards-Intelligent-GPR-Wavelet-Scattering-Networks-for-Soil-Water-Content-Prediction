import glob
import os
import argparse
import json
from itertools import product


def clean_old_files(directory='sh_files', extension_list=['.sh']):
    """
    Removes old files in the specified directory starting with 'run' and ending with the extensions provided.
    """
    if not os.path.isdir(directory):
        print(f"Directory '{directory}' does not exist.")
        return

    for ext in extension_list:
        # Adjust glob pattern to match files starting with 'run' and ending with the given extension
        files = glob.glob(os.path.join(directory, f'run*{ext}'))
        for file in files:
            try:
                os.remove(file)
                print(f"Removed old file: {file}")
            except OSError as e:
                print(f"Error removing file {file}: {e}")
def load_json(file_path):
    """
    Loads a JSON file and returns its content.
    """
    with open(file_path, 'r') as f:
        return json.load(f)


def generate_sh_files(config_list, args):
    """
    Generates .sh files based on the provided configuration list and arguments.
    Also tracks all generated .sh filenames and stores them in 'generated_sh_files.txt'.
    """
    cores_cpu = 8
    ram_mem = 8
    model_type = args.model_type
    if model_type == 'XGBRegressor' or model_type == 'RandomForestRegressor':
        cores_cpu = 16
        ram_mem = 16
    sk_model_params_location_name = args.sk_model_params_location_name
    split_function = args.split_function
    hyperparams_dir = args.hyperparams_dir
    config_file = args.config_file_location
    results_csv_dir = args.results_csv_dir
    height_column_types = args.height_column_type
    locations = args.location
    random_seed = args.random_seed
    # Create directories for the .sh files and output logs if they don't exist
    sh_files_dir = 'sh_files'
    out_files_dir = f'{results_csv_dir}/{model_type}/out'
    e_files_dir = f'{results_csv_dir}/{model_type}/error'
    os.makedirs(sh_files_dir, exist_ok=True)
    os.makedirs(out_files_dir, exist_ok=True)
    os.makedirs(e_files_dir, exist_ok=True)

    generated_sh_files = []
    PCA = ["--apply_pca",""]
    for idx_config, config in enumerate(config_list):
        for height_column_type, location,pca_apply in product(height_column_types, locations,PCA):
            current_model_type = model_type
            hyperparams_filename = f"{sk_model_params_location_name}.json"
            hyperparams_path = os.path.join(hyperparams_dir, hyperparams_filename)
            if not os.path.exists(hyperparams_path):
                print(f"Hyperparameters file not found for model '{current_model_type}': {hyperparams_path}")
                raise FileNotFoundError(f"Hyperparameters file not found for model '{current_model_type}': {hyperparams_path}")
            try:
                hyperparams = load_json(hyperparams_path)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON from '{hyperparams_path}': {e}")
                continue

            filename = f"run_{current_model_type}_{height_column_type}_{location}_{pca_apply}_config{idx_config}"

            # Build the full path for the .sh file in the new directory
            sh_filename = os.path.join(sh_files_dir, f"{filename}.sh")
            # Define output and error files
            out_filename = os.path.join(out_files_dir, f"{filename}.out")
            err_filename = os.path.join(e_files_dir, f"{filename}.err")

            # Add the SLURM output and error directives with paths
            sh_content = f"""#!/bin/bash
#SBATCH -N 1
#SBATCH -n {cores_cpu}
#SBATCH --mem={ram_mem}g
#SBATCH -J "my_sub_{filename}"
#SBATCH -p short
#SBATCH -o {out_filename}      # Standard output log
#SBATCH -e {err_filename}      # Standard error log
echo "STARTING JOB $SLURM_JOB_ID"

# Activate Conda environment
source /home/vfilardi/miniconda3/etc/profile.d/conda.sh
conda activate soil_2024 || exit 1

# Execute the Python script with the specified parameters
python main_fitting.py {pca_apply}\\
    --config_file "{config_file}" \\
    --config_index {idx_config} \\
    --height_column_type "{height_column_type}" \\
    --location "{location}" \\
    --model_type "{current_model_type}" \\
    --hyperparams "{hyperparams_dir}/{current_model_type.lower()}_params.json" \\
    --results_csv "{results_csv_dir}/{current_model_type}_results.csv" \\
    --split_function "{split_function}" \\
    --bootstrap_folder "{results_csv_dir}/{current_model_type}/bootstrap_metrics" \\
    --random_seed {random_seed}

# Deactivate Conda environment
conda deactivate

echo "ENDING JOB $SLURM_JOB_ID"
"""

            try:
                with open(sh_filename, 'w', newline='\n') as sh_file:
                    sh_file.write(sh_content)
                print(f"Generated .sh file: {sh_filename}")
            except Exception as e:
                print(f"Failed to write .sh file '{sh_filename}': {e}")
                continue

            generated_sh_files.append(sh_filename)

    tracking_file = f'{args.model_type}_generated_sh_files.txt'
    try:
        with open(tracking_file, 'w') as tf:
            for sh_file in generated_sh_files:
                tf.write(f"{sh_file}\n")
        print(f"All generated .sh files have been listed in '{tracking_file}'.")
    except Exception as e:
        print(f"Failed to write tracking file '{tracking_file}': {e}")
def main():
    parser = argparse.ArgumentParser(description="Generate .sh files based on configuration.")
    parser.add_argument('--config_file_location', type=str, required=True,
                        help='Path to the main configuration JSON file (list of dicts).')
    parser.add_argument('--model_type', type=str, required=True,
                        help='Model type to use (e.g., LinearRegression).')
    parser.add_argument('--sk_model_params_location_name', type=str, required=True,
                        help='Name of the hyperparameters JSON file without extension (e.g., linear_regression_params).')
    parser.add_argument('--split_function', type=str, required=True,
                        help='Function to split data (e.g., test_train_80_20).')
    parser.add_argument('--hyperparams_dir', type=str, default='SK_learn_models',
                        help='Directory where hyperparameters JSON files are stored.')
    parser.add_argument('--results_csv_dir', type=str, default='SK_learn_models/Results_per_sk_model',
                        help='Directory where results CSV files are stored.')
    parser.add_argument('--height_column_type', type=str, nargs='+', required=True,
                        help='List of height column types to iterate over (e.g., height_1 height_2, height_3, all).')
    parser.add_argument('--location', type=str, nargs='+', required=True,
                        help='List of locations to iterate over (e.g., all, camp_1, camp_2, wood, shed).')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42. Set to 0 for full randomness)')
    args = parser.parse_args()

    if not os.path.isdir(args.hyperparams_dir):
        print(f"Hyperparameters directory not found: {args.hyperparams_dir}")
        return

    if not os.path.isdir(args.results_csv_dir):
        print(f"Results CSV directory not found: {args.results_csv_dir}")

    if not os.path.exists(args.config_file_location):
        print(f"Configuration file not found: {args.config_file_location}")
        return

    try:
        config_list = load_json(args.config_file_location)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from '{args.config_file_location}': {e}")
        return

    if not isinstance(config_list, list):
        print(f"Configuration file '{args.config_file_location}' is not a list of configurations.")
        return

    clean_old_files()

    generate_sh_files(config_list, args)

    print("File generation completed.")


if __name__ == "__main__":
    main()
