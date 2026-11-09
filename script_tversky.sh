#!/bin/bash
#SBATCH --job-name=tversky_segformer
#SBATCH --partition=longq
#SBATCH --gres=gpu:h100_3g.47gb:1
#SBATCH --qos=longq
#SBATCH --output=tversky_%j.log
#SBATCH --error=tversky_%j.err
#SBATCH --cpus-per-task=1
#SBATCH --mem=10G
#SBATCH --time=03-00:00:00
# Load Conda environment
source ~/.bashrc
conda activate ahsan

cd /home/veningston/Aquib_Lossfunctions/SegFormer_Cityscapes_Project

# Run the test script
 python train_tversky.py

