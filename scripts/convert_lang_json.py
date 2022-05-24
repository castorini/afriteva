import os
import json
import jsonlines
import argparse


def generate_jsonl_dataset(
    input_directory, mode, lr_lang_code, lr_lang, target_lang_code="en", output_dir=None
):
    files = [file for file in os.listdir(input_directory) if mode in file]
    assert len(files) == 2, "The required amount of files not available"

    with open(
        os.path.join(input_directory, f"{mode}.{target_lang_code}"), "r"
    ) as eng_file, open(
        os.path.join(input_directory, f"{mode}.{lr_lang_code}"), "r"
    ) as lr_lang_file:
        eng_data = eng_file.read().split("\n")
        targ_lang_data = lr_lang_file.read().split("\n")
        assert len(eng_data) == len(targ_lang_data), " Data is not parallel"

    data = []
    for i in range(len(eng_data)):
        template = {"translate": {}}
        if eng_data[i].strip() != "":
            template["translate"]["English"] = eng_data[i]
            template["translate"][lr_lang] = targ_lang_data[i]
            data.append(template)

    with jsonlines.open(os.path.join(output_dir, f"{mode}.json"), "w") as writer:
        writer.write_all(data)


def main():
    parser = argparse.ArgumentParser(description="Convert dataset to jsonl format")
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--lr_lang_code", type=str, required=True)
    parser.add_argument("--lr_lang", type=str, required=True)
    args = parser.parse_args()

    modes = ["train", "test", "dev"]
    for mode in modes:
        generate_jsonl_dataset(
            input_directory=args.data_dir,
            mode=mode,
            lr_lang_code=args.lr_lang_code,
            lr_lang=args.lr_lang,
            target_lang_code="en",
            output_dir=args.data_dir,
        )


if __name__ == "__main__":
    main()
