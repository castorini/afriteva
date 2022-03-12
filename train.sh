#!/bin/bash
#SBATCH --mem=256G 
#SBATCH --cpus-per-task=10
#SBATCH --time=7-00:00:0
#SBATCH --gres=gpu:a100:2
#SBATCH --account=def-kshook
#SBATCH -o logs/train_afrit5_base_am_ha_sw.log

nvidia-smi

python3 src/trainer.py training_configs/t5_large.json
