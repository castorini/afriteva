import logging
import os
from typing import Dict

import numpy as np
import torch
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader
from utils import generate_label
from datasets import load_metric


logger = logging.getLogger(__name__)
mapper = {
    "O": 0,
    "B-DATE": 1,
    "I-DATE": 2,
    "B-PER": 3,
    "I-PER": 4,
    "B-ORG": 5,
    "I-ORG": 6,
    "B-LOC": 7,
    "I-LOC": 8,
}


class InputExample(object):
    """
    A single training/test example for token classification.
    """

    def __init__(self, guid, words, labels):
        """
        Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            words: list. The words of the sequence.
            labels: (Optional) list. The labels for each word of the sequence. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.words = words
        self.labels = labels


class InputFeatures(object):
    """
    A single set of features of data.
    """

    def __init__(self, input_ids, input_mask, segment_ids, label_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids


def read_examples_from_file(data_dir, mode):
    file_path = os.path.join(data_dir, "{}.txt".format(mode))
    guid_index = 1
    examples = []
    with open(file_path, encoding="utf-8") as f:
        words = []
        labels = []
        for line in f:
            line = line.strip()
            if len(line) < 2 or line == "\n":

                if words:
                    examples.append(
                        InputExample(
                            guid="{}-{}".format(mode, guid_index),
                            words=words,
                            labels=labels,
                        )
                    )
                    guid_index += 1
                    words = []
                    labels = []
            else:
                splits = line.split(" ")
                words.append(splits[0])
                if len(splits) > 1:
                    labels.append(splits[-1].replace("\n", ""))
                else:
                    # Examples could have no label for mode = "test"
                    labels.append("O")
        if words:
            examples.append(
                InputExample(
                    guid="{}-{}".format(mode, guid_index), words=words, labels=labels
                )
            )
    return examples


def generate_full_entity(target: str):
    new_target = []
    for i, entity in enumerate(target):
        if entity.split(":")[0] == "B-LOC":
            new_target.append(entity_loop(target, i, "LOC", entity))
        elif entity.split(":")[0] == "B-DATE":
            new_target.append(entity_loop(target, i, "DATE", entity))
        elif entity.split(":")[0] == "B-PER":
            new_target.append(entity_loop(target, i, "PER", entity))
        elif entity.split(":")[0] == "B-ORG":
            new_target.append(entity_loop(target, i, "ORG", entity))
    return new_target


def entity_loop(target: list, i: int, token_class: str, entity: str):
    entity_l = f"{token_class}: " + entity.split(":")[1]
    if len(target) == 1:
        return entity_l
    else:
        x = i + 1
        if x > len(target) - 1:
            return entity_l
        while target[x].split(":")[0] == f"I-{token_class}":
            entity_l += target[x].split(":")[1]
            if x == len(target) - 1:
                break
            x += 1
    return entity_l


class NERDataset(Dataset):
    def __init__(self, tokenizer, data_path, type_path, max_len=512, sep: str = "\t"):
        self.path = data_path
        self.type_path = type_path

        self.data = read_examples_from_file(data_path, type_path)

        self.max_len = max_len
        self.tokenizer = tokenizer

        self.inputs = []
        self.targets = []
        self.label = []

        self._build()
        self.label = set(self.label)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        source_ids = self.inputs[index]["input_ids"].squeeze()
        target_ids = self.targets[index]["input_ids"].squeeze()

        # might need to squeeze
        src_mask = self.inputs[index]["attention_mask"].squeeze()
        # might need to squeeze
        target_mask = self.targets[index]["attention_mask"].squeeze()

        return {
            "source_ids": source_ids,
            "source_mask": src_mask,
            "target_ids": target_ids,
            "target_mask": target_mask,
        }

    def _build(self):
        for idx in range(len(self.data)):
            input_, target = self.data[idx].words, self.data[idx].labels

            self.label.extend(target)

            target = [i + ": " + input_[a] for a, i in enumerate(target) if i != "O"]

            target = (
                generate_full_entity(target)
                if len(generate_full_entity(target)) > 0
                else ["None"]
            )

            input_ = " ".join(input_).lower() + " </s>"
            target = "; ".join(target).lower() + " </s>"

            tokenized_inputs = self.tokenizer.batch_encode_plus(
                [input_.strip()],
                max_length=self.max_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            # tokenize targets
            tokenized_targets = self.tokenizer.batch_encode_plus(
                [target.strip()],
                max_length=self.max_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )

            self.inputs.append(tokenized_inputs)
            self.targets.append(tokenized_targets)


def get_dataset(tokenizer, type_path, args):
    return NERDataset(
        tokenizer=tokenizer,
        data_path=args.data_path,
        type_path=type_path,
        max_len=args.max_seq_length,
    )


if __name__ == "__main__":
    x = read_examples_from_file("data/ner/yor", "dev")
    tokenizer = AutoTokenizer.from_pretrained("afriT5_run/afriT5_small")
    sample_dataset = NERDataset(
        tokenizer=tokenizer, data_path="data/ner/yor", type_path="dev"
    )

    input = tokenizer.decode(
        sample_dataset[1]["source_ids"],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    # print(len(sample_dataset[0]["source_ids"]))
    print(input)
    target = tokenizer.decode(
        sample_dataset[1]["target_ids"], skip_special_tokens=True
    ).strip()
    target += "; loc: amurka"
    print(target)
    labels = (
        generate_label(input.strip(), target.strip())
        if target != "none"
        else ["O"] * len(input.split())
    )
    print(labels)
    metric = load_metric("seqeval")
    print(metric.compute(predictions=[labels], references=[labels]))

    # metric = load_metric("seqeval")
    # loader = DataLoader(sample_dataset, batch_size=16,
    #                     num_workers=1, shuffle=True)

    # for batch in loader:
    #     target = [tokenizer.decode(ids, skip_special_tokens=True,  clean_up_tokenization_spaces=False).strip()
    #               for ids in batch["target_ids"]]
    #     texts = [tokenizer.decode(ids, skip_special_tokens=True,  clean_up_tokenization_spaces=False).strip()
    #              for ids in batch["source_ids"]]
    #     label = [generate_label(texts[i].strip(), target[i].strip()) if target[i].strip() != 'none' else [
    #         "O"]*len(texts[i].strip().split()) for i in range(len(texts))]

    #     # print(target)
    #     # print(texts)
    #     # print(label)
    #     print(metric.compute(predictions=label, references=label))
    #     break
