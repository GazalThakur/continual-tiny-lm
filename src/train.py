"""
Single-domain training script for StreamLM.
Supports initialization from a previous checkpoint (for sequential / continual training).
Includes reproducible validation split and best-checkpoint selection by validation loss.
"""

from __future__ import annotations

import argparse
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, random_split

from src.model import TinyTransformer, count_parameters

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    """Configuration for single-domain training."""
    domain: int = 1
    batch_size: int = 8
    max_seq_len: int = 256
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    num_epochs: int = 3
    save_every: int = 500
    log_every: int = 50
    grad_clip: float = 1.0
    val_ratio: float = 0.08
    checkpoint_dir: str = "checkpoints"
    data_dir: str = "data"
    seed: int = 42
    init_from: Optional[str] = None
    load_optimizer: bool = False


class TextDataset(Dataset):
    """Next-token prediction dataset with proper padding and loss masking."""

    PAD_ID = -100

    def __init__(self, data: list[torch.Tensor], max_seq_len: int) -> None:
        self.data = data
        self.max_seq_len = max_seq_len

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
            x = torch.cat([x, torch.zeros(pad_len, dtype=torch.long)])
            y = torch.cat([y, torch.full((pad_len,), self.PAD_ID, dtype=torch.long)])

        return x, y


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_tokenizer(data_dir: str) -> dict:
    path = Path(data_dir) / "tokenizer.pt"
    if not path.exists():
        raise FileNotFoundError(f"Tokenizer not found at {path}. Run data.py first.")
    return torch.load(path, weights_only=False)


def load_domain_data(data_dir: str, domain: int) -> list[torch.Tensor]:
    path = Path(data_dir) / f"domain_{domain}.pt"
    if not path.exists():
        raise FileNotFoundError(f"Domain data not found at {path}. Run data.py first.")
    return torch.load(path, weights_only=False)


def create_dataloaders(
    data: list[torch.Tensor],
    max_seq_len: int,
    batch_size: int,
    val_ratio: float,
    seed: int,
) -> Tuple[DataLoader, DataLoader]:
    """Create reproducible train and validation DataLoaders."""
    generator = torch.Generator().manual_seed(seed)
    val_size = int(len(data) * val_ratio)
    train_size = len(data) - val_size

    train_subset, val_subset = random_split(
        data, [train_size, val_size], generator=generator
    )

    train_dataset = TextDataset(list(train_subset), max_seq_len=max_seq_len)
    val_dataset = TextDataset(list(val_subset), max_seq_len=max_seq_len)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    return train_loader, val_loader


@torch.no_grad()
def evaluate(
    model: TinyTransformer,
    dataloader: DataLoader,
    device: torch.device,
) -> float:
    """Compute token-weighted validation loss."""
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


def save_checkpoint(
    model: TinyTransformer,
    optimizer: torch.optim.Optimizer,
    step: int,
    epoch: int,
    val_loss: Optional[float],
    path: Path,
) -> None:
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

    # Data
    logger.info("Loading Domain %d...", config.domain)
    raw_data = load_domain_data(config.data_dir, config.domain)
    logger.info("Total sequences: %d", len(raw_data))

    train_loader, val_loader = create_dataloaders(
        data=raw_data,
        max_seq_len=config.max_seq_len,
        batch_size=config.batch_size,
        val_ratio=config.val_ratio,
        seed=config.seed,
    )
    logger.info(
        "Train sequences: %d | Val sequences: %d",
        len(train_loader.dataset),
        len(val_loader.dataset),
    )

    # Model
    tokenizer = load_tokenizer(config.data_dir)
    model = TinyTransformer(
        vocab_size=tokenizer["vocab_size"],
        n_embd=256,
        n_head=8,
        n_layer=6,
        block_size=config.max_seq_len,
    ).to(device)
    logger.info("Parameters: %s", f"{count_parameters(model):,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    # ---------- Initialize from previous checkpoint ----------
    # Design choice: we load model weights but start with a fresh optimizer.
    # This is the standard approach when moving to a new domain.
    if config.init_from is not None:
        init_path = Path(config.init_from)
        if not init_path.exists():
            raise FileNotFoundError(f"Init checkpoint not found: {init_path}")

        ckpt = torch.load(init_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])

        prev_epoch = ckpt.get("epoch", "?")
        prev_val_loss = ckpt.get("val_loss", None)
        if prev_val_loss is not None:
            logger.info(
                "Loaded model weights from %s (previous epoch=%s, val_loss=%.4f)",
                init_path, prev_epoch, prev_val_loss,
            )
        else:
            logger.info(
                "Loaded model weights from %s (previous epoch=%s)",
                init_path, prev_epoch,
            )

        if config.load_optimizer and "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
            logger.info("Also loaded optimizer state")
        else:
            logger.info("Optimizer starts fresh (recommended for new domain)")
    else:
        logger.info("Training from scratch (random initialization)")

    # ---------- Training ----------
    checkpoint_dir = Path(config.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    model.train()
    step = 0
    best_val_loss = float("inf")
    last_val_loss: Optional[float] = None
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
                logger.info(
                    "Step %5d | Train Loss: %.4f | Avg: %.4f | Time: %.1fs",
                    step, loss.item(), avg, elapsed,
                )

            if step % config.save_every == 0:
                ckpt_path = checkpoint_dir / f"domain{config.domain}_step{step}.pt"
                save_checkpoint(model, optimizer, step, epoch, None, ckpt_path)

        # End of epoch
        avg_train_loss = epoch_loss / num_batches
        val_loss = evaluate(model, val_loader, device)
        last_val_loss = val_loss

        logger.info(
            "Epoch %d | Train Loss: %.4f | Val Loss: %.4f",
            epoch, avg_train_loss, val_loss,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = checkpoint_dir / f"domain{config.domain}_best.pt"
            save_checkpoint(model, optimizer, step, epoch, val_loss, best_path)
            logger.info("New best validation loss: %.4f", best_val_loss)

    # Final save (uses last_val_loss to avoid relying on loop variable leakage)
    final_path = checkpoint_dir / f"domain{config.domain}_final.pt"
    save_checkpoint(model, optimizer, step, config.num_epochs, last_val_loss, final_path)
    logger.info("Training finished. Best validation loss: %.4f", best_val_loss)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train StreamLM on a single domain")
    parser.add_argument("--domain", type=int, default=1, help="Domain number to train on")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--val-ratio", type=float, default=0.08, help="Validation split ratio")
    parser.add_argument(
        "--init-from",
        type=str,
        default=None,
        help="Path to previous checkpoint to initialize from (e.g. checkpoints/domain1_best.pt)",
    )
    parser.add_argument(
        "--load-optimizer",
        action="store_true",
        help="Also load optimizer state from the init checkpoint",
    )
    args = parser.parse_args()

    config = TrainConfig(
        domain=args.domain,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        val_ratio=args.val_ratio,
        init_from=args.init_from,
        load_optimizer=args.load_optimizer,
    )
    train(config)


if __name__ == "__main__":
    main()