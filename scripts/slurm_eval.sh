#!/bin/bash -l
#SBATCH --export=NONE
#SBATCH --nodes=1
#SBATCH --clusters=alex
#SBATCH --partition=a100
##SBATCH --constraint=a100_80
#SBATCH --gres=gpu:a100:1
#SBATCH --output logs/eval/runs/%x-%j.out
#SBATCH --error logs/eval/runs/%x-%j.out
#SBATCH --time=01:00:00
#SBATCH --job-name=dcerecon

# Export environment from this script to srun
unset SLURM_EXPORT_ENV             

# Activate environment and load modules
module load python
module load cuda/12.3.0
conda activate dcerecon


srun --unbuffered python src/eval.py "$@"
