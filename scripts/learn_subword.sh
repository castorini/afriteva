#!/bin/bash

export CUDA_AVAILABLE_DEVICES=0
data_path=$1
vocab_size=$2
output_path=$3

train_data_path=${data_path}/all_train.txt
output_path=${output_path}

mkdir -p $output_path

echo "Learning subword units..."
python3 src/subword.py --input=$train_data_path --output_path=$output_path --vocab_size=$vocab_size
echo "Done! All files below saved in ${output_path}"
ls $output_path
