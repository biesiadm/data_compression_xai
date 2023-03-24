#!/bin/bash
#SBATCH --job-name=kernel_thinning_forest_shap
#SBATCH --partition=long
#SBATCH --time=3-00:00:00
#SBATCH --mem-per-cpu=50G
#SBATCH -w dgx-3
#SBATCH --ntasks=1
#SBATCH --gpus=0
#SBATCH --output=/home2/faculty/pkaczynska/exp_shap_big.txt
#SBATCH --mail-type=ALL
#SBATCH --mail-user=kaczynska.pm@gmail.com
source /home2/faculty/pkaczynska/.bashrc
source /raid/shared/mair/miniconda3/etc/profile.d/conda.sh
cd ./repo/data_compression_xai/explanation-compression-main
conda activate exp7_env
python3 exp_big.py
