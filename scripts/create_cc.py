import argparse
import os
import random

"""
Sample Train and Evaluation Sentences from a monolingual corpus.
"""

LANG_CODES = {
    "am": "amharic",
    "ha": "hausa",
    "ig": "igbo",
    "yo": "yoruba",
    "pcm": "pidgin",
    "ti": "tigrinya",
    "so": "somali",
    "sw": "swahili",
    "orm": "oromoo",
}


def main():
    parser = argparse.ArgumentParser(
        description="Subsample Sentences from Monolingual Corpus"
    )
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--filename")
    parser.add_argument("--data-dir")
    parser.add_argument("--n-lines", type=int)
    parser.add_argument("--n-lines-train", type=int)
    parser.add_argument("--n-lines-eval", type=int)
    args = parser.parse_args()

    random.seed(args.seed)

    n_lines_to_sample = args.n_lines_train + args.n_lines_eval

    lines = random.sample(range(args.n_lines), n_lines_to_sample)
    train_lines = sorted(lines[: args.n_lines_train])
    eval_lines = sorted(lines[args.n_lines_train : n_lines_to_sample])

    file = os.path.join(args.data_dir, args.filename)
    lang = LANG_CODES[args.filename.split(".")[0]]

    train = os.path.join(args.data_dir, f"train.{lang}")
    eval = os.path.join(args.data_dir, f"eval.{lang}")
    fp = open(file, "r")

    with open(train, "w") as train_out, open(eval, "w") as eval_out:
        j = 0
        k = 0
        for i, line in enumerate(fp):
            print(i)
            if (j < len(train_lines)) and (i == train_lines[j]):
                train_out.write(line)
                j += 1
            elif (k < len(eval_lines)) and (i == eval_lines[k]):
                eval_out.write(line)
                k += 1
            elif (j >= len(train_lines)) and (k >= len(eval_lines)):
                break


if __name__ == "__main__":
    main()
