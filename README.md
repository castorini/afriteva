# AfriTeVa: Extending “Small Data” Pretraining Approaches to Sequence-to-Sequence Models 

This repo contains the code for the paper [AfriTeVa: Extending “Small Data” Pretraining Approaches to Sequence-to-Sequence Models ](#)
AfriTeVa is a sequence to 

## Languages Covered During Training

Afaan Oromoo(orm), Amharic(amh), Gahuza(gah), Hausa(hau), Igbo(igb), Nigerian Pidgin(pcm), Somali(som), Swahili(swa), Tigrinya(tig), Yoruba(yor)

## Reproducibility
 
The data for language modelling can be downloaded from [this URL](https://huggingface.co/datasets/castorini/afriberta-corpus)

### Tokenizer

We trained a Sentencepiece Unigram tokenizer for AfriTeVa, and it can be downloaded from [Here](#)
However, to train a custom tokenizer, run the command below with the following arguments

- data_path: Path to your training file/files
- vocab_size: Size of your learned vocabulary (number of tokens)
- output_path: Path to store learned tokenizer files

```bash
(virtual_env)$ bash learn_subword.sh ${data_path} ${vocab_size} ${output_path} 
```

## Citation
 xxx

