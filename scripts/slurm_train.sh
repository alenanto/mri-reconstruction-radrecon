#!/bin/bash -l
#SBATCH --export=NONE
#SBATCH --nodes=1
#SBATCH --clusters=tinyx
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --output logs/train/runs/%x-%j.out
#SBATCH --error logs/train/runs/%x-%j.out
#SBATCH --time=12:00:00
#SBATCH --job-name=dcerecon_train
#SBATCH --mail-user=alen.anto@fau.de
#SBATCH --mail-type=BEGIN,END,FAIL

# Export environment from this script to srun
unset SLURM_EXPORT_ENV             

# Activate environment and load modules
cd /home/hpc/iwbi/iwbi101h/Rad_Recon
source ~/miniconda3/etc/profile.d/conda.sh
conda activate racer
module load cuda/12.4.1



srun --unbuffered python src/train.py "$@"
