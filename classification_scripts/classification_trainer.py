from classification_dataset import ClassificationDataset
from utils import LoggingCallback
import random
import argparse
import textwrap
import torch
from tqdm.auto import tqdm
from sklearn import metrics
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from model import T5FineTuner
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger

MODEL_MAX_LENGTH = 512


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Finetune T5 fle classification")
    parser.add_argument("--train_data_path", type=str, required=True, help="Path of input training file")
    parser.add_argument("--eval_data_path", type=str, required=False, help="Path of input eval file")
    parser.add_argument("--test_data_path", type=str, required=False, help="Path of input eval file")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to store checkpoint")
    parser.add_argument("--model_name_or_path", type=str, help="Model name or path", required=True)
    parser.add_argument("--tokenizer_name_or_path", type=str, help="Tokenizer name or path")
    parser.add_argument("--max_seq_length", type=int, default=MODEL_MAX_LENGTH, help="Maimum sequence length")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="learning rate")
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
    parser.add_argument("--class_labels", type=str, required=True)
    parser.add_argument("--data_column", type=str, required=True)
    parser.add_argument("--target_column", type=str, required=True)

    return parser


def generate_class_token(class_labels: list, tokenizer):
    class_map = {}
    for label in class_labels:
        if tokenizer.convert_tokens_to_ids(label) > 5:
            class_map[label] = label
        else:
            token = ""
            while not token.startswith("▁"):
                token = random.sample(list(tokenizer.vocab.keys()), 1)[0]
            class_map[label] = token.replace("▁", "")
    return class_map


def main():
    parser = get_parser()
    args = parser.parse_args()

    print(f"Training Arguments {args}")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path)

    label = args.class_labels.split(",") 
    class_map = generate_class_token(label, tokenizer)
    inv_class_map = {v: k for k, v in class_map.items()}

    dataset_kwargs = {
        "data_column": args.data_column,
        "target_column": args.target_column,
        "max_seq_length": args.max_seq_length,
        "class_map": class_map,
    }

    if args.train_data_path is not None:
        train_dataset = ClassificationDataset(
            tokenizer=tokenizer,
            data_path=args.train_data_path,
            **dataset_kwargs
        )
    
    if args.eval_data_path is not None:
        eval_dataset = ClassificationDataset(
            tokenizer=tokenizer,
            data_path=args.eval_data_path,
            **dataset_kwargs
        )

    if args.test_data_path is not None:
        test_dataset = ClassificationDataset(
            tokenizer=tokenizer,
            data_path=args.test_data_path,
            **dataset_kwargs
        )

    # save checkpoint during training
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        filename=args.output_dir + "/checkpoint.pth",
        monitor="val_loss",
        mode="min",
        save_last=True,
        every_n_epochs=1,
        verbose=True,
    )

    csv_logger = CSVLogger(
        save_dir=args.output_dir,
        name="",
        version=""
    )

    train_params = dict(
        accumulate_grad_batches=args.gradient_accumulation_steps,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=args.n_gpu if args.n_gpu else 1,
        max_epochs=args.num_train_epochs,
        precision=16 if args.fp_16 else 32,
        gradient_clip_val=args.max_grad_norm,
        callbacks=[LoggingCallback(), checkpoint_callback],
        logger=csv_logger
    )

    print("[INFO] Training model .........")
    t5_finetuner_module = T5FineTuner(args, train_dataset=train_dataset, eval_dataset=eval_dataset)
    trainer = pl.Trainer(**train_params)
    trainer.fit(t5_finetuner_module)


    print("[INFO] Evaluating model .........")
    test_loader = DataLoader(test_dataset, batch_size=64, num_workers=4, shuffle=True)
    t5_finetuner_module.model.eval()
    t5_finetuner_module = t5_finetuner_module.to("cpu")
    outputs = []
    targets = []
    for batch in tqdm(test_loader):

        outs = t5_finetuner_module.model.generate(
            input_ids=batch["source_ids"],
            attention_mask=batch["source_mask"],
            max_length=2,
        )
        dec = [tokenizer.decode(ids, skip_special_tokens=True) for ids in outs]
        target = [
            tokenizer.decode(ids, skip_special_tokens=True)
            for ids in batch["target_ids"]
        ]
        target = [inv_class_map[item] for item in target]
        dec = [inv_class_map[item] for item in dec]

        outputs.extend(dec)
        targets.extend(target)
    print(outputs)
    print(targets)
    print(metrics.accuracy_score(targets, outputs))
    print(metrics.classification_report(targets, outputs))

    print("[INFO] Generating predictions .........")
    new_batch = next(iter(test_loader))
    new_batch["source_ids"].shape
    outs = t5_finetuner_module.model.generate(
        input_ids=new_batch["source_ids"],
        attention_mask=new_batch["source_mask"],
        max_length=2,
    )

    dec = [tokenizer.decode(ids, skip_special_tokens=True) for ids in outs]

    texts = [
        tokenizer.decode(ids, skip_special_tokens=True)
        for ids in new_batch["source_ids"]
    ]
    targets = [
        tokenizer.decode(ids, skip_special_tokens=True)
        for ids in new_batch["target_ids"]
    ]

    targets = [inv_class_map[item] for item in targets]
    dec = [inv_class_map[item] for item in dec]

    for i in range(32):
        c = texts[i]
        lines = textwrap.wrap("text:\n%s\n" % c, width=100)
        print("\n".join(lines))
        print("\nActual sentiment: %s" % targets[i])
        print("predicted sentiment: %s" % dec[i])
        print("=====================================================================\n")


if __name__ == "__main__":
    main()
