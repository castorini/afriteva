#!/bin/bash
#SBATCH --mem=64G
#SBATCH --cpus-per-task=10
#SBATCH --time=24:00:0
#SBATCH --gres=gpu:a100:1

#SBATCH -o logs/learn_tokenizer_am_ha_sw.log

# export CUDA_AVAILABLE_DEVICES=0
data_path=$1
vocab_size=$2

train_data_path=$data_path/train_afriberta/all_train.txt
output_path=tokenizers/all_lang_eng_tokenizer


mkdir -p $output_path

echo "Learning subword units..."
python3 src/subword.py --input=$train_data_path --output_path=$output_path --vocab_size=$vocab_size
echo "Done! All files below saved in ${output_path}"
ls $output_path
