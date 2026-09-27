"""
Train a BPE tokenizer (vocab size 5000) for Experiment 2.
Saves files into data_v2/ only.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from datasets import load_dataset
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def get_training_corpus():
    """Yield text from all four domains for BPE training."""
    logger.info("Loading datasets for tokenizer training...")

    # Domain A - TinyStories
    ts = load_dataset("roneneldan/TinyStories", split="train[:8000]")

    # Domain B - WikiText
    wt = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train[:6000]")

    # Domain C - Python code
    code = load_dataset("bigcode/the-stack-smol", data_dir="data/python", split="train[:6000]")

    # Domain D - Empathetic Dialogues (replacement for broken daily_dialog)
        # Domain D - OpenAssistant conversations
    dialog = load_dataset("OpenAssistant/oasst1", split="train[:6000]")

    def yield_texts():
        for ex in ts:
            if ex["text"] and len(ex["text"].strip()) > 20:
                yield ex["text"]

        for ex in wt:
            if ex["text"] and len(ex["text"].strip()) > 20:
                yield ex["text"]

        for ex in code:
            content = ex.get("content") or ""
            if content and len(content.strip()) > 20:
                yield content

        for ex in dialog:
            text = ex.get("text") or ""
            if text and len(text.strip()) > 20:
                yield text

    return yield_texts()


def train_bpe(vocab_size: int = 5000, output_dir: str = "data_v2") -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<pad>", "<unk>", "<s>", "</s>"],
        min_frequency=2,
        show_progress=True,
    )

    logger.info("Training BPE tokenizer (vocab_size=%d)...", vocab_size)
    tokenizer.train_from_iterator(get_training_corpus(), trainer=trainer)

    # Save tokenizer
    tokenizer_file = output_path / "tokenizer.json"
    tokenizer.save(str(tokenizer_file))
    logger.info("Saved tokenizer to %s", tokenizer_file)

    # Save config
    config = {
        "vocab_size": tokenizer.get_vocab_size(),
        "pad_token": "<pad>",
        "unk_token": "<unk>",
    }
    config_file = output_path / "tokenizer_config.json"
    with open(config_file, "w") as f:
        json.dump(config, f, indent=2)

    logger.info("Saved config to %s", config_file)
    logger.info("Final vocab size: %d", tokenizer.get_vocab_size())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab-size", type=int, default=5000)
    parser.add_argument("--output-dir", type=str, default="data_v2")
    args = parser.parse_args()
    train_bpe(vocab_size=args.vocab_size, output_dir=args.output_dir)