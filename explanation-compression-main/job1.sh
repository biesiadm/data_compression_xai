#!/bin/bash
#SBATCH --job-name=kernel_thinning_nocat
#SBATCH --partition=long
#SBATCH --time=2-00:00:00
#SBATCH -w dgx-3
#SBATCH --mem-per-cpu=8G
#SBATCH --ntasks=1
#SBATCH --gpus=0
#SBATCH --output=./results/exp_knn_bank_nocat_check.txt
#SBATCH --mail-type=ALL
#SBATCH --mail-user=as406153@student.uw.edu.pl
source /home/as406153/.bashrc
source /home/as406153/anaconda3/etc/profile.d/conda.sh
conda activate exp7_env
python3 exp_bank_knn.py
