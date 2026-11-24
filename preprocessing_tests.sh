#!/bin/bash
#SBATCH --job-name=preprocessing-tests
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p short
#SBATCH --mem=2GB
#SBATCH --time=00:10:00

set -eo pipefail

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
echo "STARTING TEST JOB ${SLURM_JOB_ID:-local}"

source /home/vfilardi/miniconda3/etc/profile.d/conda.sh
conda activate soil_2024 || exit 1
source "./scripts/install_akela_shared.sh" || exit 1

python -B -m unittest discover -s tests -v

echo "ENDING TEST JOB ${SLURM_JOB_ID:-local}"
