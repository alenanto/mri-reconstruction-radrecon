#!/bin/bash -l
#SBATCH --export=NONE
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --clusters=tinyx
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.outexit
#SBATCH --time=23:00:00
#SBATCH --job-name=dcerecon_pre
#SBATCH --mail-user=alen.anto@fau.de
#SBATCH --mail-type=BEGIN,END,FAIL

unset SLURM_EXPORT_ENV
cd /home/hpc/iwbi/iwbi101h/Rad_Recon
source ~/miniconda3/etc/profile.d/conda.sh
conda activate racer
module load cuda/12.4.1

srun --unbuffered python ./scripts/fastMRI_breast_preprocessing.py --skip-existing