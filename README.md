# AfriTeVa: Extending “Small Data” Pretraining Approaches to Sequence-to-Sequence Models 

This repo contains the code for the paper [AfriTeVa: Extending “Small Data” Pretraining Approaches to Sequence-to-Sequence Models ](#)
AfriTeVa is a sequence to 

## Languages Covered During Pretraining

Afaan Oromoo(orm), Amharic(amh), Gahuza(gah), Hausa(hau), Igbo(igb), Nigerian Pidgin(pcm), Somali(som), Swahili(swa), Tigrinya(tig), Yoruba(yor)

**Models:**

We release the following pretrained models:

- [AfriTeVa Small](https://huggingface.co/castorini/afriteva_small) (64M params)
- [AfriTeVa Base](https://huggingface.co/castorini/afriteva_base) (229M params)
- [AfriTeVa Large](https://huggingface.co/castorini/afriteva_large) (745M params)

## Reproducibility

### Datasets

- **Language Modelling**: The data for language modelling can be downloaded from [this URL](https://huggingface.co/datasets/castorini/afriberta-corpus)
- **Machine Translation**: To obtain the Machine Translation dataset, please download it from [this repository](https://github.com/masakhane-io/lafand-mt)

- **Text Classification**: To obtain the topic classification dataset, please download it from [this repository](https://github.com/uds-lsv/transfer-distant-transformer-african)

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

```bibtex
@inproceedings{jude-ogundepo-etal-2022-afriteva,
    title = "{A}fri{T}e{VA}: Extending ?Small Data? Pretraining Approaches to Sequence-to-Sequence Models",
    author = "Jude Ogundepo, Odunayo  and
      Oladipo, Akintunde  and
      Adeyemi, Mofetoluwa  and
      Ogueji, Kelechi  and
      Lin, Jimmy",
    booktitle = "Proceedings of the Third Workshop on Deep Learning for Low-Resource Natural Language Processing",
    month = jul,
    year = "2022",
    address = "Hybrid",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.deeplo-1.14",
    doi = "10.18653/v1/2022.deeplo-1.14",
    pages = "126--135",
    abstract = "t",
}
```

