#!/bin/bash
#SBATCH --job-name=combo_dice_focal
#SBATCH --partition=longq
#SBATCH --gres=gpu:h100_3g.47gb:1
#SBATCH --qos=longq
#SBATCH --output=combo_dice_focal_%j.log
#SBATCH --error=combo_dice_focal_%j.err
#SBATCH --cpus-per-task=1
#SBATCH --mem=10G
#SBATCH --time=03-00:00:00
# Load Conda environment
source ~/.bashrc
conda activate ahsan

cd /home/veningston/Aquib_Lossfunctions/SegFormer_Cityscapes_Project

# Run the test script
 python train_combo_dice_focal.py
