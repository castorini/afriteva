#!/bin/bash
#SBATCH --mem=128G 
#SBATCH --cpus-per-task=10
#SBATCH --time=1:00:0
#SBATCH --gres=gpu:a100:1
#SBATCH --account=def-kshook
#SBATCH -o logs/finetune_classification_am_ha_sw.log

export CUDA_AVAILABLE_DEVICES=0
export TOKENIZERS_PARALLELISM=true

python3 classification_scripts/classification_trainer.py --data_path="data/classification/hausa_newsclass" \
                       --model_name_or_path="afriT5_run/afriT5_small_am_ha_sw" \
                       --tokenizer_name_or_path="afriT5_run/afriT5_small_am_ha_sw" \
                       --output_dir="afriT5_run/afrit5_small_classification_run" \
                       --max_seq_length="512" \
                       --learning_rate="3e-4" \
                       --weight_decay="0.0" \
                       --adam_epsilon="1e-8" \
                       --warmup_steps="0" \
                       --train_batch_size="16" \
                       --eval_batch_size="16" \
                       --num_train_epochs="100" \
                       --gradient_accumulation_steps="16" \
                       --n_gpu="1" \
                       --fp_16="true" \
                       --max_grad_norm="1.0" \
                       --opt_level="O1" \
                       --seed="42" \
                       --lang="hausa"
