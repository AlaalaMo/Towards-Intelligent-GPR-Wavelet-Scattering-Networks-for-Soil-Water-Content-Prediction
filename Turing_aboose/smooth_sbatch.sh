#!/bin/bash
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -p long
#SBATCH --mail-type=BEGIN,END        # send email when job begins and ends
#SBATCH --mail-user=vfilardi@wpi.edu
#SBATCH -J "sbatch_submission_manager"  # Job name
#SBATCH -o generate_submit_%j.out      # Standard output log
#SBATCH -e generate_submit_%j.err      # Standard error log

echo "STARTING JOB $SLURM_JOB_ID"

# Activate Conda environment
source /home/vfilardi/miniconda3/etc/profile.d/conda.sh
conda activate soil_2024

# Define variables
CONFIG_FILE_LOCATION="Final_data/freq_mean_sweep/freq_configs_20241006.json"
MODEL_TYPE="LinearRegression"  # Ensure consistency: use the same case and format
SK_MODEL_PARAMS_LOCATION_NAME="linearregression_params"
SPLIT_FUNCTION="test_train_80_20"
HYPERPARAMS_DIR="SK_learn_models"
RESULTS_CSV_DIR="SK_learn_models/Results_per_sk_model"
HEIGHT_COLUMN_TYPES=("height_1" "height_2" "height_3" "all")
LOCATIONS=("all" "camp_1_wood" "camp2_wood" "wood" "shed")

# Run the generation script
python sk_learn_generate_sbatch_cpu.py \
    --config_file_location "$CONFIG_FILE_LOCATION" \
    --model_type "$MODEL_TYPE" \
    --sk_model_params_location_name "$SK_MODEL_PARAMS_LOCATION_NAME" \
    --split_function "$SPLIT_FUNCTION" \
    --hyperparams_dir "$HYPERPARAMS_DIR" \
    --results_csv_dir "$RESULTS_CSV_DIR" \
    --height_column_type "${HEIGHT_COLUMN_TYPES[@]}" \
    --location "${LOCATIONS[@]}"

# Check if the generation script was successful
if [ $? -ne 0 ]; then
    echo "Generation script failed. Exiting."
    conda deactivate
    exit 1
fi

# Run the submission script
#python sk_learn_non_stop_sbatch.py --model_type "$MODEL_TYPE"
#
## Check if the submission script was successful
#if [ $? -ne 0 ]; then
#    echo "Submission script failed."
#else
#    echo "All jobs have been submitted successfully."
#fi

# Deactivate Conda environment
conda deactivate

echo "ENDING JOB $SLURM_JOB_ID"
