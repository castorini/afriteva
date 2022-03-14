import os
import jsonlines
import pandas as pd


def generate_jsonl_dataset(input_directory, mode, output_dir=None):
    file = os.path.join(input_directory,f"{mode}.tsv")
    assert os.path.exists(file), "The required amount of files not available"

    print(file)
    data_df = pd.read_csv(file, delimiter="\t")

    try:
        eng_records = list(data_df.iloc[:, 0])
        yor_records = list(data_df.iloc[:, 1])
    except:
        eng_records = list(data_df.iloc[:, 0])
        yor_records = list(data_df.iloc[:, 1])

    eng_data = eng_records
    yor_data = yor_records
    assert len(eng_data) == len(yor_data), " Data is not parallel"

    data = []
    for i in range(len(eng_data)):
        template = {"translate": {}}
        template["translate"]["English"] = eng_data[i]
        template["translate"]["Yoruba"] = yor_data[i]
        data.append(template)

    with jsonlines.open(os.path.join(output_dir, f"{mode}.json"), "w") as writer:
        writer.write_all(data)


def main():
    file_path = "menyo-20k_MT-master/data"
    modes = os.listdir(file_path)

    for mode in modes:
        generate_jsonl_dataset(file_path, mode=mode.split('.')[0], output_dir="yoruba_json")


if __name__ == "__main__":
    main()
