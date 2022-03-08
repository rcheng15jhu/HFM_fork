#!/bin/bash -l
#SBATCH --job-name=Vel3D_1
#SBATCH --time=35:00:00
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=48
#SBATCH --gres=gpu:1
#SBATCH -A rni2_gpu
#SBATCH --mail-type=end
#SBATCH --mail-user=rcheng15@jhu.edu

source /data/apps/go.sh
ml anaconda
conda activate HFMtf1.15
cd /home/rcheng15/HFM/HFM_fork/Source 
TF_CPP_MIN_LOG_LEVEL=2 python3.7 Vel3D_itBound_noNS.py gpu_attempt_20_130 50 20 130
