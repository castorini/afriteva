#!/bin/bash

LANGS=("en")
DATA_DIR=data
MIN_TOKENS=6
SEED=42
N_TRAIN_SENTENCES=(1000000)
N_EVAL_SENTENCES=(100000)
mkdir -p $DATA_DIR


for i in "${!LANGS[@]}"
    do
        lang=${LANGS[i]}
        wget http://data.statmt.org/cc-100/$lang.txt.xz -O $DATA_DIR/$lang.txt.xz -o $DATA_DIR/$lang.txt.xz.out
        unxz $DATA_DIR/$lang.txt.xz

        # Remove lines containing only white space or few number of tokens
        cat $DATA_DIR/$lang.txt | sed '/^[[:space:]]*$/d' | awk -v n_tokens=$MIN_TOKENS 'NF>n_tokens'> $DATA_DIR/"$lang".new.txt
        mv $DATA_DIR/"$lang".new.txt $DATA_DIR/$lang.txt

        echo "Downloaded, unzipped & cleaned data for language: $lang"
        tail -n 3 $DATA_DIR/$lang.txt.xz.out
        n_sent=$(echo $(wc -l $DATA_DIR/$lang.txt) | cut -d ' ' -f 1) 
        echo "$n_sent sentences in cleaned data"

        python scripts/create_cc.py \
        --data-dir $DATA_DIR \
        --seed $SEED \
        --filename $lang.txt \
        --n-lines $n_sent \
        --n-lines-train ${N_TRAIN_SENTENCES[$i]} \
        --n-lines-eval ${N_EVAL_SENTENCES[$i]}
    done
