#!/bin/bash
#SBATCH -N 1
#SBATCH -n 8
#SBATCH -p short
#SBATCH --mem=32GB
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-user=vfilardi@wpi.edu
echo "STARTING JOB $SLURM_JOB_ID"

source /home/vfilardi/miniconda3/etc/profile.d/conda.sh
conda  activate soil_2024 || exit 1
source "./scripts/install_akela_shared.sh" --persistent || exit 1
python -u preprocessing_main.py
preprocessing_status=$?
conda deactivate

echo "ENDING JOB $SLURM_JOB_ID (preprocessing exit status: $preprocessing_status)"
exit "$preprocessing_status"
