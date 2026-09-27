"""
Prepare Experiment 2 domains using the trained BPE tokenizer.
Writes only to data_v2/.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List

import torch
from datasets import load_dataset
from tokenizers import Tokenizer
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def load_bpe_tokenizer(tokenizer_dir: str = "data_v2") -> Tokenizer:
    path = Path(tokenizer_dir) / "tokenizer.json"
    if not path.exists():
        raise FileNotFoundError(f"Tokenizer not found at {path}. Run train_tokenizer_v2.py first.")
    return Tokenizer.from_file(str(path))


def texts_to_sequences(
    texts: List[str],
    tokenizer: Tokenizer,
    max_seq_len: int = 256,
    min_seq_len: int = 32,
) -> List[torch.Tensor]:
    sequences = []
    for text in tqdm(texts, desc="Tokenizing", leave=False):
        if not text or len(text.strip()) < 20:
            continue
        encoded = tokenizer.encode(text)
        ids = encoded.ids
        for i in range(0, len(ids) - 1, max_seq_len):
            chunk = ids[i : i + max_seq_len]
            if len(chunk) >= min_seq_len:
                sequences.append(torch.tensor(chunk, dtype=torch.long))
    return sequences


def prepare_domain(
    name: str,
    texts: List[str],
    tokenizer: Tokenizer,
    output_dir: Path,
    max_seq_len: int = 256,
) -> None:
    logger.info("Processing domain: %s (%d raw texts)", name, len(texts))
    sequences = texts_to_sequences(texts, tokenizer, max_seq_len=max_seq_len)
    save_path = output_dir / f"domain_{name}.pt"
    torch.save(sequences, save_path)
    logger.info("Saved %d sequences → %s", len(sequences), save_path)


def main(output_dir: str = "data_v2", max_seq_len: int = 256, samples_per_domain: int = 8000):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    tokenizer = load_bpe_tokenizer(output_dir)

    # Domain A - TinyStories
    ts = load_dataset("roneneldan/TinyStories", split=f"train[:{samples_per_domain}]")
    prepare_domain("A", [ex["text"] for ex in ts], tokenizer, output_path, max_seq_len)

    # Domain B - WikiText
    wt = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split=f"train[:{samples_per_domain}]")
    prepare_domain("B", [ex["text"] for ex in wt], tokenizer, output_path, max_seq_len)

    # Domain C - Python code
    code = load_dataset("bigcode/the-stack-smol", data_dir="data/python", split=f"train[:{samples_per_domain}]")
    prepare_domain("C", [ex.get("content", "") for ex in code], tokenizer, output_path, max_seq_len)

    # Domain D - Empathetic Dialogues
    dialog = load_dataset("empathetic_dialogues", split=f"train[:{samples_per_domain}]")
    dialog_texts = [
        (ex.get("context", "") + " " + ex.get("utterance", "")).strip()
        for ex in dialog
    ]
    prepare_domain("D", dialog_texts, tokenizer, output_path, max_seq_len)

    logger.info("All Experiment 2 domains prepared in %s", output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default="data_v2")
    parser.add_argument("--max-seq-len", type=int, default=256)
    parser.add_argument("--samples-per-domain", type=int, default=8000)
    args = parser.parse_args()
    main(
        output_dir=args.output_dir,
        max_seq_len=args.max_seq_len,
        samples_per_domain=args.samples_per_domain,
    )