#!/bin/bash
#SBATCH -N 1
#SBATCH -n 10
#SBATCH -p short
##SBATCH -p long
#SBATCH --gres=gpu:1
#SBATCH --mem 8G
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-user=vfilardi@wpi.edu
##SBATCH -C "A100|A100-80G"
echo "STARTING JOB $SLURM_JOB_ID"
source /home/vfilardi/miniconda3/etc/profile.d/conda.sh
conda  activate soil_2024
python initial_DL_fit.py --train_function simple_train_model --search_space model/task_2_joru1_params.json --domain freq_mean_sweep --random_seed 42 --nn_model Stand_RNN
echo "ENDING JOB $SLURM_JOB_ID"

###