
from typing import Dict
import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class ClassificationDataset(Dataset):
    def __init__(self, tokenizer, data_path, type_path, class_map, max_len=512, sep: str = "\t"):
        self.path = os.path.join(data_path, type_path + '.tsv')
        self.class_map = class_map

        self.data_column = "news_title"
        self.class_column = "label"
        self.data = pd.read_csv(self.path, sep=sep)

        self.max_len = max_len
        self.tokenizer = tokenizer

        self.inputs = []
        self.targets = []

        self._build()

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        source_ids = self.inputs[index]["input_ids"].squeeze()
        target_ids = self.targets[index]["input_ids"].squeeze()

        # might need to squeeze
        src_mask = self.inputs[index]["attention_mask"].squeeze()
        # might need to squeeze
        target_mask = self.targets[index]["attention_mask"].squeeze()

        return {"source_ids": source_ids, "source_mask": src_mask, "target_ids": target_ids, "target_mask": target_mask}

    def _build(self):
        for idx in range(len(self.data)):
            input_, target = self.data.loc[idx,
                                           self.data_column], self.data.loc[idx, self.class_column]

            target = self.class_map[target.lower()]
            
            input_ = input_.lower() + ' </s>'
            target = target + " </s>"

            # tokenize inputs
            tokenized_inputs = self.tokenizer.batch_encode_plus(
                [input_], max_length=self.max_len, padding="max_length", truncation=True, return_tensors="pt"
            )
            # tokenize targets
            tokenized_targets = self.tokenizer.batch_encode_plus(
                [target], max_length=2, padding="max_length", truncation=True, return_tensors="pt"
            )

            # print(target)
            # print(tokenized_targets)

            self.inputs.append(tokenized_inputs)
            self.targets.append(tokenized_targets)


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("afriT5_run/afriT5_small")
    dataset = ClassificationDataset(tokenizer=tokenizer,
                                    data_path="data/classification/hausa_newsclass",
                                    type_path="test")
    print(tokenizer.decode(dataset[0]['source_ids'], skip_special_tokens=True))
    print(tokenizer.decode(
        dataset[0]['target_ids'], skip_special_tokens=True))
