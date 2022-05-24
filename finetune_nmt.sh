#!/bin/bash
#SBATCH --mem=128G 
#SBATCH --cpus-per-task=10
#SBATCH --time=48:00:0
#SBATCH --gres=gpu:a100:2
#SBATCH --account=def-jimmylin
#SBATCH -o logs/%j.log

export CUDA_AVAILABLE_DEVICES=0,1

python3 machine_translation/nmt_trainer.py \
    --model_name_or_path google/mt5-base \
    --do_train \
    --do_eval \
    --source_lang English \
    --target_lang Yoruba \
    --source_prefix "translate English to Yoruba: " \
    --train_file data/relnews/train.json \
    --validation_file data/relnews/dev.json \
    --test_file data/relnews/test.json \
    --output_dir experiments/mt5_base_en_yo \
    --max_source_length 200 \
    --max_target_length 200 \
    --per_device_train_batch_size=10 \
    --per_device_eval_batch_size=10 \
    --overwrite_output_dir \
    --predict_with_generate \
    --save_steps 5000 \
    --num_beams 10 \
    --do_predict \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 10 \
    --report_to wandb
