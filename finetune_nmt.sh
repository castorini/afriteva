#!/bin/bash
#SBATCH --mem=128G 
#SBATCH --cpus-per-task=10
#SBATCH --time=48:00:0
#SBATCH --gres=gpu:v100l:1
#SBATCH --account=rrg-jimmylin
#SBATCH -o logs/nmt_base_10.log

export CUDA_AVAILABLE_DEVICES=0

CUDA_VISIBLE_DEVICES=0 python3 machine_translation/nmt_trainer.py \
    --model_name_or_path google/mt5-base \
    --do_train \
    --do_eval \
    --source_lang English \
    --target_lang Yoruba \
    --source_prefix "translate English to Yoruba: " \
    --train_file data/menyo/train.json \
    --validation_file data/menyo/dev.json \
    --test_file data/menyo/test.json \
    --output_dir mt5_yo_en \
    --max_source_length 200 \
    --max_target_length 200 \
    --per_device_train_batch_size=2 \
    --per_device_eval_batch_size=2 \
    --overwrite_output_dir \
    --predict_with_generate \
    --save_steps 50000 \
    --num_beams 10 \
    --do_predict \
    --gradient_accumulation_steps 2 \
    --num_train_epochs 10
