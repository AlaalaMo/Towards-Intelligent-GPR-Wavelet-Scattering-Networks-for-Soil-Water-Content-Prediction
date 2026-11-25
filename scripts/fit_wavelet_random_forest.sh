#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p short
#SBATCH --mem=32GB
set -eo pipefail

if [[ $# -ne 0 ]]; then
    echo "Usage: sbatch scripts/fit_wavelet_random_forest.sh" >&2
    exit 2
fi

source /home/vfilardi/miniconda3/etc/profile.d/conda.sh
conda activate soil_2024
exec python -u fit_wavelet_random_forest.py
