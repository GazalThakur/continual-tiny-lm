"""
Evaluation script for Experiment 2.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import DataLoader, Dataset, random_split
from tokenizers import Tokenizer

from src.model import TinyTransformer, count_parameters

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


class TextDataset(Dataset):
    def __init__(self, data, max_seq_len, pad_id):
        self.data = data
        self.max_seq_len = max_seq_len
        self.pad_id = pad_id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq = self.data[idx]
        if len(seq) > self.max_seq_len:
            seq = seq[:self.max_seq_len]
        x = seq[:-1].clone()
        y = seq[1:].clone()
        pad_len = (self.max_seq_len - 1) - len(x)
        if pad_len > 0:
            x = torch.cat([x, torch.full((pad_len,), self.pad_id, dtype=torch.long)])
            y = torch.cat([y, torch.full((pad_len,), -100, dtype=torch.long)])
        return x, y


def load_bpe_tokenizer(data_dir: str):
    tok = Tokenizer.from_file(str(Path(data_dir) / "tokenizer.json"))
    with open(Path(data_dir) / "tokenizer_config.json") as f:
        cfg = json.load(f)
    return tok, cfg


def get_pad_id(tokenizer: Tokenizer) -> int:
    """Look up the <pad> token id. Raises if the special token is missing,
    rather than silently falling back to 0 (which could mask a real bug)."""
    pad_id = tokenizer.token_to_id("<pad>")
    if pad_id is None:
        raise ValueError(
            "`<pad>` token not found in tokenizer vocabulary — "
            "check that train_tokenizer_v2.py registered it as a special token."
        )
    return pad_id


def load_domain_data(data_dir: str, domain: str):
    return torch.load(Path(data_dir) / f"domain_{domain}.pt", weights_only=False)


@torch.no_grad()
def compute_val_loss(model, loader, device):
    model.eval()
    total_loss = total_tokens = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        _, loss = model(x, targets=y)
        n = (y != -100).sum().item()
        total_loss += loss.item() * n
        total_tokens += n
    return total_loss / max(total_tokens, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", type=str, required=True, choices=["A", "B", "C", "D"])
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data-dir", type=str, default="data_v2")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--val-ratio", type=float, default=0.08)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    tokenizer, tok_cfg = load_bpe_tokenizer(args.data_dir)
    vocab_size = tok_cfg["vocab_size"]
    pad_id = get_pad_id(tokenizer)

    data = load_domain_data(args.data_dir, args.domain)
    logger.info("Domain %s | sequences: %d", args.domain, len(data))

    gen = torch.Generator().manual_seed(42)
    val_size = int(len(data) * args.val_ratio)
    _, val_subset = random_split(data, [len(data) - val_size, val_size], generator=gen)
    val_loader = DataLoader(TextDataset(list(val_subset), 256, pad_id), batch_size=args.batch_size)

    model = TinyTransformer(vocab_size=vocab_size, n_embd=256, n_head=8, n_layer=6, block_size=256).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    logger.info("Loaded %s", args.checkpoint)

    val_loss = compute_val_loss(model, val_loader, device)
    logger.info("Validation loss: %.4f", val_loss)


if __name__ == "__main__":
    main()