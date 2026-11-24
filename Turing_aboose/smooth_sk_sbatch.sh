#!/bin/bash
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -p long
#SBATCH --mail-type=BEGIN,END        # Send email when job begins and ends
#SBATCH --mail-user=vfilardi@wpi.edu
#SBATCH -J "sbatch_submission_manager"  # Job name
#SBATCH -o Turing_aboose/out/generate_submit_%j.out      # Standard output log
#SBATCH -e Turing_aboose/e/generate_submit_%j.err      # Standard error log
##SBATCH --dependency=afterok:799407   # Start after job 799407 finishes successfully

echo "STARTING JOB $SLURM_JOB_ID"

source /home/vfilardi/miniconda3/etc/profile.d/conda.sh
conda activate soil_2024 || exit 1
source "./scripts/install_akela_shared.sh" --persistent || exit 1

CONFIG_FILE_LOCATION="Final_data/freq_mean_sweep/freq_mean_sweep_configs_20251024.json"
SPLIT_FUNCTION="stratified_train_test_split_80_20"
HYPERPARAMS_DIR="SK_learn_models"
RESULTS_CSV_DIR="SK_learn_models/Results_per_sk_model"
#HEIGHT_COLUMN_TYPES=("height_1" "height_2" "height_3" "all")
#LOCATIONS=("all" "camp_1_wood" "camp2_wood" "wood")

HEIGHT_COLUMN_TYPES=("height_2")
LOCATIONS=("all")
MODEL_TYPES=("LinearRegression" "Lasso" "MultiTaskLasso" "RandomForestRegressor" "XGBRegressor")
PARAMS_LOCATIONS=("linearregression_params" "lasso_params" "multitasklasso_params" "randomforestregressor_params" "xgbregressor_params")

#MODEL_TYPES=("RandomForestRegressor")
#PARAMS_LOCATIONS=("randomforestregressor_params")


TRACKING_FILE="completed_models.txt"

if [ ! -f "$TRACKING_FILE" ]; then
    touch "$TRACKING_FILE"
    echo "Created tracking file: $TRACKING_FILE"
fi

is_model_completed() {
    local model="$1"
    grep -Fxq "$model" "$TRACKING_FILE"
}

run_model() {
    local model_type="$1"
    local params_location="$2"

    # Check if the model has already been processed
    #if is_model_completed "$model_type"; then
       # echo "---------------------------------------------"
        #echo "Model '$model_type' has already been processed. Skipping."
       # echo "---------------------------------------------"
       # return
    #fi

    echo "---------------------------------------------"
    echo "Processing Model Type: $model_type"
    echo "Parameters Location: $params_location"
    echo "HEIGHT_COLUMN_TYPES: ${HEIGHT_COLUMN_TYPES[@]}"
    echo "LOCATIONS: ${LOCATIONS[@]}"
    echo "---------------------------------------------"

    # Run the generation script
    python sk_learn_generate_sbatch_cpu.py \
        --config_file_location "$CONFIG_FILE_LOCATION" \
        --model_type "$model_type" \
        --sk_model_params_location_name "$params_location" \
        --split_function "$SPLIT_FUNCTION" \
        --hyperparams_dir "$HYPERPARAMS_DIR" \
        --results_csv_dir "$RESULTS_CSV_DIR" \
        --height_column_type "${HEIGHT_COLUMN_TYPES[@]}" \
        --location "${LOCATIONS[@]}" \
        --random_seed 42

    # Check if the generation script was successful
    if [ $? -ne 0 ]; then
        echo "Generation script failed for model_type '$model_type'. Exiting."
        conda deactivate
        exit 1
    fi

    # Run the submission script
    echo "# Run the submission script for $model_type"
    python sk_learn_non_stop_sbatch.py --model_type "$model_type"

    # Check if the submission script was successful
    if [ $? -ne 0 ]; then
        echo "Submission script failed for model_type '$model_type'."
    else
        echo "All jobs have been submitted successfully for model_type '$model_type'."
        # Append the model type to the tracking file
        echo "$model_type" >> "$TRACKING_FILE"
        echo "Marked '$model_type' as completed in '$TRACKING_FILE'."
    fi
}

for i in "${!MODEL_TYPES[@]}"; do
    run_model "${MODEL_TYPES[$i]}" "${PARAMS_LOCATIONS[$i]}"
done

# Deactivate Conda environment
conda deactivate

echo "ENDING JOB $SLURM_JOB_ID"
