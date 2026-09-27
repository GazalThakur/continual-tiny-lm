"""
Data preparation for StreamLM.
Downloads TinyStories, splits into sequential domains, builds a character-level tokenizer,
and saves tokenized sequences.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import torch
from datasets import load_dataset
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


@dataclass
class DataConfig:
    """Configuration for data preparation."""
    num_domains: int = 4
    samples_per_domain: int = 8000
    max_seq_len: int = 256
    min_seq_len: int = 32
    data_dir: str = "data"
    dataset_name: str = "roneneldan/TinyStories"
    seed: int = 42


def build_char_tokenizer(texts: List[str]) -> Dict:
    """Build a simple character-level tokenizer from a list of texts."""
    all_text = "".join(texts)
    chars = sorted(list(set(all_text)))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    return {
        "stoi": stoi,
        "itos": itos,
        "vocab_size": len(chars),
    }


def encode(text: str, stoi: Dict[str, int]) -> List[int]:
    """Convert text to list of token ids."""
    return [stoi[c] for c in text if c in stoi]


def tokenize_domain(
    texts: List[str],
    stoi: Dict[str, int],
    max_seq_len: int,
    min_seq_len: int,
) -> List[torch.Tensor]:
    """Tokenize a list of texts and split into fixed-length sequences."""
    tokenized: List[torch.Tensor] = []
    for text in tqdm(texts, desc="Tokenizing", leave=False):
        ids = encode(text, stoi)
        for start in range(0, len(ids) - 1, max_seq_len):
            chunk = ids[start : start + max_seq_len]
            if len(chunk) >= min_seq_len:
                tokenized.append(torch.tensor(chunk, dtype=torch.long))
    return tokenized


def prepare_data(config: DataConfig) -> None:
    """Main data preparation pipeline."""
    data_dir = Path(config.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading dataset: %s", config.dataset_name)
    dataset = load_dataset(config.dataset_name, split="train")

    total_needed = config.num_domains * config.samples_per_domain
    logger.info("Taking first %d stories...", total_needed)
    texts = [dataset[i]["text"] for i in tqdm(range(total_needed), desc="Loading")]

    # Split into sequential domains
    domains: List[List[str]] = []
    for i in range(config.num_domains):
        start = i * config.samples_per_domain
        end = start + config.samples_per_domain
        domains.append(texts[start:end])
        logger.info("Domain %d: %d stories", i + 1, len(domains[i]))

    # Build and save tokenizer
    logger.info("Building character-level vocabulary...")
    tokenizer = build_char_tokenizer(texts)
    logger.info("Vocab size: %d", tokenizer["vocab_size"])

    tokenizer_path = data_dir / "tokenizer.pt"
    torch.save(tokenizer, tokenizer_path)
    logger.info("Tokenizer saved to %s", tokenizer_path)

    # Tokenize and save each domain
    for i, domain_texts in enumerate(domains):
        logger.info("Processing Domain %d...", i + 1)
        tokenized = tokenize_domain(
            texts=domain_texts,
            stoi=tokenizer["stoi"],
            max_seq_len=config.max_seq_len,
            min_seq_len=config.min_seq_len,
        )
        save_path = data_dir / f"domain_{i + 1}.pt"
        torch.save(tokenized, save_path)
        logger.info("Saved %d sequences → %s", len(tokenized), save_path)

    logger.info("Data preparation complete.")


def main() -> None:
    config = DataConfig()
    prepare_data(config)


if __name__ == "__main__":
    main()