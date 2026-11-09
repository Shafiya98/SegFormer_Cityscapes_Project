#!/bin/bash
#SBATCH --job-name=compression
#SBATCH --partition=longq
#SBATCH --gres=gpu:h100_3g.47gb:1
#SBATCH --qos=longq
#SBATCH --output=compression_%j.log
#SBATCH --cpus-per-task=1
#SBATCH --mem=10G
#SBATCH --time=03-00:00:00
# Load Conda environment
source ~/.bashrc
conda activate ahsan

cd /home/veningston/Aquib_Lossfunctions/SegFormer_Cityscapes_Project

echo "Python Path:"
which python

python --version

python -c "import transformers; print(transformers.__version__)"
# Run the test script
 python train_focal.py

