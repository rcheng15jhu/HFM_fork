#!/bin/bash -l
#SBATCH --job-name=Vel3D
#SBATCH --time=50:00:00
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=48
#SBATCH -A rni2_gpu
#SBATCH --mail-type=end
#SBATCH --mail-user=rcheng15@jhu.edu

source /data/apps/go.sh
ml anaconda
conda activate HFMtf1.15
cd /home/rcheng15/HFM/HFM_fork/Source 
python3.7 Vel3D_noGPU.py 10_150_10_100 10 150 10 100