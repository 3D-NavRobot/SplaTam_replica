#!/bin/bash
#SBATCH -p GPU-shared
#SBATCH -A cis250051p
#SBATCH --gres=gpu:v100-32:1
#SBATCH -t 00:10:00
#SBATCH --mem=15GB
#SBATCH -o replica_fast_2_run_splatam_%j.out
#SBATCH -e replica_fast_2_run_splatam_%j.err

# Load any necessary modules
module load cuda/12.4.0  
module load gcc/10.2.0

source ~/.bashrc
conda activate /jet/home/gnamomsa/.conda/envs/galane_splatam  

nvidia-smi
# Run your SplaTAM script
python scripts/splatam1.py configs/replica/replica_fast.py

