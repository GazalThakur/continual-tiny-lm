"""
Training script for Experiment 2 (BPE + new domains).
Uses data_v2/ and checkpoints_v2/ only.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, random_split
from tokenizers import Tokenizer

from src.model import TinyTransformer, count_parameters

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    domain: str = "A"                    # A / B / C / D
    batch_size: int = 32
    max_seq_len: int = 256
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    num_epochs: int = 3
    save_every: int = 500
    log_every: int = 50
    grad_clip: float = 1.0
    val_ratio: float = 0.08
    data_dir: str = "data_v2"
    checkpoint_dir: str = "checkpoints_v2"
    seed: int = 42
    init_from: Optional[str] = None
    load_optimizer: bool = False


class TextDataset(Dataset):
    def __init__(self, data: list[torch.Tensor], max_seq_len: int, pad_id: int) -> None:
        self.data = data
        self.max_seq_len = max_seq_len
        self.pad_id = pad_id

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        seq = self.data[idx]
        if len(seq) > self.max_seq_len:
            seq = seq[: self.max_seq_len]

        x = seq[:-1].clone()
        y = seq[1:].clone()

        pad_len = (self.max_seq_len - 1) - len(x)
        if pad_len > 0:
            x = torch.cat([x, torch.full((pad_len,), self.pad_id, dtype=torch.long)])
            y = torch.cat([y, torch.full((pad_len,), -100, dtype=torch.long)])  # ignore_index

        return x, y


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_bpe_tokenizer(data_dir: str) -> Tuple[Tokenizer, dict]:
    tok_path = Path(data_dir) / "tokenizer.json"
    cfg_path = Path(data_dir) / "tokenizer_config.json"
    if not tok_path.exists():
        raise FileNotFoundError(f"Tokenizer not found: {tok_path}")
    tokenizer = Tokenizer.from_file(str(tok_path))
    with open(cfg_path) as f:
        config = json.load(f)
    return tokenizer, config


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


def load_domain_data(data_dir: str, domain: str) -> list[torch.Tensor]:
    path = Path(data_dir) / f"domain_{domain}.pt"
    if not path.exists():
        raise FileNotFoundError(f"Domain data not found: {path}")
    return torch.load(path, weights_only=False)


def create_dataloaders(
    data: list[torch.Tensor],
    max_seq_len: int,
    batch_size: int,
    val_ratio: float,
    seed: int,
    pad_id: int,
) -> Tuple[DataLoader, DataLoader]:
    generator = torch.Generator().manual_seed(seed)
    val_size = int(len(data) * val_ratio)
    train_size = len(data) - val_size

    train_subset, val_subset = random_split(data, [train_size, val_size], generator=generator)

    train_ds = TextDataset(list(train_subset), max_seq_len, pad_id=pad_id)
    val_ds = TextDataset(list(val_subset), max_seq_len, pad_id=pad_id)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


@torch.no_grad()
def evaluate(model, dataloader, device) -> float:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        _, loss = model(x, targets=y)
        n_tokens = (y != -100).sum().item()
        total_loss += loss.item() * n_tokens
        total_tokens += n_tokens
    model.train()
    return total_loss / max(total_tokens, 1)


def save_checkpoint(model, optimizer, step, epoch, val_loss, path: Path) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "epoch": epoch,
            "val_loss": val_loss,
        },
        path,
    )
    logger.info("Saved checkpoint → %s", path)


def train(config: TrainConfig) -> None:
    torch.manual_seed(config.seed)
    device = get_device()
    logger.info("Using device: %s", device)

    # Tokenizer
    tokenizer, tok_config = load_bpe_tokenizer(config.data_dir)
    vocab_size = tok_config["vocab_size"]
    pad_id = get_pad_id(tokenizer)
    logger.info("Vocab size: %d | pad_id: %d", vocab_size, pad_id)

    # Data
    logger.info("Loading Domain %s ...", config.domain)
    raw_data = load_domain_data(config.data_dir, config.domain)
    logger.info("Total sequences: %d", len(raw_data))

    train_loader, val_loader = create_dataloaders(
        raw_data, config.max_seq_len, config.batch_size,
        config.val_ratio, config.seed, pad_id
    )
    logger.info("Train: %d | Val: %d", len(train_loader.dataset), len(val_loader.dataset))

    # Model
    model = TinyTransformer(
        vocab_size=vocab_size,
        n_embd=256,
        n_head=8,
        n_layer=6,
        block_size=config.max_seq_len,
    ).to(device)
    logger.info("Parameters: %s", f"{count_parameters(model):,}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )

    # Init from previous domain
    if config.init_from:
        ckpt = torch.load(config.init_from, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        prev_val = ckpt.get("val_loss", None)
        logger.info("Loaded weights from %s (prev val_loss=%s)", config.init_from, prev_val)
        if config.load_optimizer and "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
            logger.info("Loaded optimizer state")
        else:
            logger.info("Optimizer starts fresh")
    else:
        logger.info("Training from scratch")

    checkpoint_dir = Path(config.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    model.train()
    step = 0
    best_val_loss = float("inf")
    last_val_loss = None
    start_time = time.time()

    for epoch in range(1, config.num_epochs + 1):
        logger.info("===== Epoch %d/%d =====", epoch, config.num_epochs)
        epoch_loss = 0.0
        num_batches = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            _, loss = model(x, targets=y)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1
            step += 1

            if step % config.log_every == 0:
                avg = epoch_loss / num_batches
                elapsed = time.time() - start_time
                logger.info("Step %5d | Loss: %.4f | Avg: %.4f | Time: %.1fs", step, loss.item(), avg, elapsed)

            if step % config.save_every == 0:
                path = checkpoint_dir / f"domain{config.domain}_step{step}.pt"
                save_checkpoint(model, optimizer, step, epoch, None, path)

        avg_train = epoch_loss / num_batches
        val_loss = evaluate(model, val_loader, device)
        last_val_loss = val_loss
        logger.info("Epoch %d | Train: %.4f | Val: %.4f", epoch, avg_train, val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = checkpoint_dir / f"domain{config.domain}_best.pt"
            save_checkpoint(model, optimizer, step, epoch, val_loss, best_path)
            logger.info("New best val loss: %.4f", best_val_loss)

    final_path = checkpoint_dir / f"domain{config.domain}_final.pt"
    save_checkpoint(model, optimizer, step, config.num_epochs, last_val_loss, final_path)
    logger.info("Finished. Best val loss: %.4f", best_val_loss)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", type=str, required=True, choices=["A", "B", "C", "D"])
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--init-from", type=str, default=None)
    parser.add_argument("--load-optimizer", action="store_true")
    args = parser.parse_args()

    config = TrainConfig(
        domain=args.domain,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        init_from=args.init_from,
        load_optimizer=args.load_optimizer,
    )
    train(config)


if __name__ == "__main__":
    main()