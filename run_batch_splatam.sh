#!/bin/bash
#SBATCH -p GPU-shared
#SBATCH -A cis250051p
#SBATCH --gres=gpu:v100-32:1
#SBATCH -t 24:00:00
#SBATCH --mem=32GB
#SBATCH -o adaptive_model_soup_splatam_%j.out
#SBATCH -e adaptive_model_soup_splatam_%j.err

# Load any necessary modules
module load cuda/12.4.0  
module load gcc/10.2.0

source ~/.bashrc
conda activate /jet/home/gnamomsa/.conda/envs/galane_splatam  

nvidia-smi
# Run your SplaTAM script

python scripts/splatam1_soup.py configs/replica/splatam.py
# python scripts/splatam1.py configs/replica/splatam.py
# python scripts/splatam_dynamic_lr.py configs/replica/splatam.py