#!/bin/bash
#SBATCH --job-name=WGAN_EEGImageRecon ### Job Name
#SBATCH --partition=gpu
### Job Partition
#SBATCH --nodes=1
### Number of Nodes
#SBATCH --ntasks-per-node=1
###SBATCH --ntasks-per-core=2
### Number of tasks (MPI processes)
#SBATCH --gres=gpu:a100:1
### Number of GPU Cards
#SBATCH --mem-per-cpu 16G
#SBATCH --time 5-0:00:00
#SBATCH --output=submited_log/%x-%j.out
#SBATCH --error=submited_log/%x-%j.err
### set module variable
export MODULEPATH=/archive/gpu/apps/gpumodules/modules/all
export PYTHONPATH="/archive/gpu/home/users/jakrapop.a/jupyter_work/WGAN-EEGImageRecon"
source /etc/profile.d/00-modulepath.sh
### load module
module purge
module load Anaconda3/2020.02
eval "$(conda shell.bash hook)"
source activate jakrapop_env39
srun python3 train_img_gen.py
