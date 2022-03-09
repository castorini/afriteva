#!/bin/bash
#SBATCH --mem=128G 
#SBATCH --cpus-per-task=10
#SBATCH --time=1:00:0
#SBATCH --gres=gpu:a100:1
#SBATCH --account=def-kshook
#SBATCH -o logs/finetune_classification_am_ha_sw.log

export CUDA_AVAILABLE_DEVICES=0
export TOKENIZERS_PARALLELISM=true

CUDA_VISIBLE_DEVICES=0 python machine_translation/nmt_trainer.py \
    --model_name_or_path google/byt5-base \
    --do_train \
    --do_eval \
    --source_lang yo \
    --target_lang en \
    --source_prefix "translate Yoruba to English: " \
    --train_file data/yo_en/train.json \
    --validation_file data/yo_en/dev.json \
    --test_file data/yo_en/test.json \
    --output_dir byt5_yo_en \
    --max_source_length 200 \
    --max_target_length 200 \
    --per_device_train_batch_size=10 \
    --per_device_eval_batch_size=10 \
    --overwrite_output_dir \
    --predict_with_generate \
    --save_steps 50000 \
    --num_beams 10 \
    --do_predict