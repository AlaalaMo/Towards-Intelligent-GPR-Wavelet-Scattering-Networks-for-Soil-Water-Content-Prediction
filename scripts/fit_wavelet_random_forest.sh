#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p short
#SBATCH --mem=32GB
set -eo pipefail

private_run=${1:-${AKELA_RF_PRIVATE_RUN:-}}
if [[ -z "$private_run" || ! -f "$private_run" || ! -r "$private_run" ]]; then
    echo "Supply a readable private run JSON as the first argument or AKELA_RF_PRIVATE_RUN." >&2
    exit 2
fi

source /home/vfilardi/miniconda3/etc/profile.d/conda.sh
conda activate soil_2024
exec python -u fit_wavelet_random_forest.py --private-run "$private_run"
