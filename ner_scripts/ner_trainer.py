from ner_datasets import NERDataset, get_dataset
from utils import LoggingCallback, generate_label
import pandas as pd
import numpy as np
import torch
import sys
import random
import argparse
import textwrap
from tqdm.auto import tqdm
from sklearn import metrics
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from model import T5FineTuner
import pytorch_lightning as pl
from datasets import load_metric
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

MODEL_MAX_LENGTH = 512


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Finetune T5 fle classification")
    parser.add_argument(
        "--data_path", type=str, required=True, help="Path of input training file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to store checkpoint",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Model name or path",
    )
    parser.add_argument(
        "--tokenizer_name_or_path",
        type=str,
        help="Tokenizer name or path",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=MODEL_MAX_LENGTH,
        help="Maimum sequence length",
    )
    parser.add_argument(
        "--learning_rate", type=float, default=3e-4, help="learning rate"
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="weight decay")
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--warmup_steps", type=float, default=0)
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--num_train_epochs", type=int, default=10)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--n_gpu", type=int, default=1)
    parser.add_argument("--early_stop_callback", type=bool, default=False)
    parser.add_argument("--fp_16", type=bool, default=False)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lang", type=str)
    parser.add_argument("--opt_level", type=str, default="O1")
    parser.add_argument("--use_fast_tokenizer", type=bool, default=True)

    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    print(f"Training Arguments {args}")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path)
    metric = load_metric("seqeval")

    train_dataset = NERDataset(
        tokenizer=tokenizer, data_path=args.data_path, type_path="train"
    )
    eval_dataset = NERDataset(
        tokenizer=tokenizer, data_path=args.data_path, type_path="dev"
    )
    test_dataset = NERDataset(
        tokenizer=tokenizer, data_path=args.data_path, type_path="test"
    )

    # save checkpoint during training
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        filename=args.output_dir + "/checkpoint.pth",
        monitor="val_loss",
        mode="min",
        save_top_k=5,
        verbose=True,
    )
    train_params = dict(
        accumulate_grad_batches=args.gradient_accumulation_steps,
        gpus=args.n_gpu,
        max_epochs=args.num_train_epochs,
        # early_stop_callback=False,
        precision=16 if args.fp_16 else 32,
        # amp_level=args.opt_level,
        gradient_clip_val=args.max_grad_norm,
        checkpoint_callback=checkpoint_callback,
        callbacks=[LoggingCallback()],
    )
    # Train
    model = T5FineTuner(args)
    trainer = pl.Trainer(**train_params)
    trainer.fit(model)

    # test evaluation
    test_loader = DataLoader(test_dataset, batch_size=32, num_workers=2, shuffle=True)
    model.model.eval()
    model = model.to("cpu")
    outputs = []
    targets = []
    all_text = []
    true_labels = []
    pred_labels = []
    for batch in tqdm(test_loader):

        outs = model.model.generate(
            input_ids=batch["source_ids"], attention_mask=batch["source_mask"]
        )
        dec = [
            tokenizer.decode(
                ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            ).strip()
            for ids in outs
        ]
        target = [
            tokenizer.decode(
                ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            ).strip()
            for ids in batch["target_ids"]
        ]
        texts = [
            tokenizer.decode(
                ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            ).strip()
            for ids in batch["source_ids"]
        ]
        true_label = [
            generate_label(texts[i].strip(), target[i].strip())
            if target[i].strip() != "none"
            else ["O"] * len(texts[i].strip().split())
            for i in range(len(texts))
        ]
        pred_label = [
            generate_label(texts[i].strip(), dec[i].strip())
            if dec[i].strip() != "none"
            else ["O"] * len(texts[i].strip().split())
            for i in range(len(texts))
        ]

        outputs.extend(dec)
        targets.extend(target)
        true_labels.extend(true_label)
        pred_labels.extend(pred_label)
        all_text.extend(texts)

    for i in range(10):
        print(f"Text:  {all_text[i]}")
        print(f"Predicted Token Class:  {pred_labels[i]}")
        print(f"True Token Class:  {true_labels[i]}")
        print("=====================================================================\n")

    print(metric.compute(predictions=pred_labels, references=true_labels))

    # random
    new_batch = next(iter(test_loader))
    new_batch["source_ids"].shape
    outs = model.model.generate(
        input_ids=new_batch["source_ids"], attention_mask=new_batch["source_mask"]
    )
    dec = [
        tokenizer.decode(
            ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        ).strip()
        for ids in outs
    ]
    target = [
        tokenizer.decode(
            ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        ).strip()
        for ids in new_batch["target_ids"]
    ]
    texts = [
        tokenizer.decode(
            ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        ).strip()
        for ids in new_batch["source_ids"]
    ]

    for i in range(32):
        c = texts[i]
        lines = textwrap.wrap("text:\n%s\n" % c, width=100)
        print("\n".join(lines))
        print("\nActual sentiment: %s" % target[i])
        print("predicted sentiment: %s" % dec[i])
        print("=====================================================================\n")


if __name__ == "__main__":
    main()
