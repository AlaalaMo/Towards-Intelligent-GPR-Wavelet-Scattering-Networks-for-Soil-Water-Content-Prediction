#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p short
#SBATCH --mem=16GB
set -eo pipefail

if [[ $# -ne 1 || ! -f "$1" || ! -r "$1" ]]; then
    echo "Supply the path to a readable raw evaluation CSV." >&2
    exit 2
fi

source /home/vfilardi/miniconda3/etc/profile.d/conda.sh
conda activate soil_2024
exec python -u evaluate_wavelet_random_forest.py --eval-dataset "$1"
