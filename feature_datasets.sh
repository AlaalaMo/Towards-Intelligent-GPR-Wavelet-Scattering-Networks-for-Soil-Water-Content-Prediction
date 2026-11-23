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
conda  activate soil_2024
python preprocessing_main.py
conda deactivate

echo "ENDING JOB $SLURM_JOB_ID"