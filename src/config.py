import argparse
import inspect
from transformers import AutoConfig, AutoTokenizer
from transformers import T5Config
from subword import SentencePieceUnigramTokenizer


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate training config")
    parser.add_argument(
        "--tokenizer_name", type=str, required=True, help="Path of token file"
    )
    parser.add_argument(
        "--config_name", type=str, required=True, help="Model size or type"
    )
    parser.add_argument(
        "--output_config_path", type=str, required=True, help="Path of config file"
    )

    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    config = AutoConfig.from_pretrained(args.tokenizer_name)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, config=config)

    config = T5Config.from_pretrained(
        args.tokenizer_name, vocab_size=len(tokenizer.vocab)
    )
    config.save_pretrained(args.output_config_path)


if __name__ == "__main__":
    main()
