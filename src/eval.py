"""
Evaluation and text generation for StreamLM.
Supports any domain and checkpoint via command-line arguments.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

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
class EvalConfig:
    """Configuration for evaluation."""
    domain: int = 1
    checkpoint_path: str = "checkpoints/domain1_best.pt"
    data_dir: str = "data"
    max_seq_len: int = 256
    batch_size: int = 16
    val_ratio: float = 0.08
    num_generate: int = 5
    max_new_tokens: int = 200
    temperature: float = 0.8
    seed: int = 42


class TextDataset(Dataset):
    """Next-token prediction dataset with proper padding."""

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


def load_tokenizer(data_dir: str) -> dict:
    """Load the character-level tokenizer."""
    path = Path(data_dir) / "tokenizer.pt"
    if not path.exists():
        raise FileNotFoundError(f"Tokenizer not found: {path}")
    return torch.load(path, weights_only=False)


def load_domain_data(data_dir: str, domain: int) -> list[torch.Tensor]:
    """Load tokenized sequences for a given domain."""
    path = Path(data_dir) / f"domain_{domain}.pt"
    if not path.exists():
        raise FileNotFoundError(f"Domain data not found: {path}")
    return torch.load(path, weights_only=False)


def create_val_loader(
    data: list[torch.Tensor],
    max_seq_len: int,
    val_ratio: float,
    batch_size: int,
    seed: int,
) -> DataLoader:
    """Create a reproducible validation DataLoader using torch.random_split."""
    generator = torch.Generator().manual_seed(seed)
    val_size = int(len(data) * val_ratio)
    train_size = len(data) - val_size

    _, val_subset = random_split(data, [train_size, val_size], generator=generator)

    val_dataset = TextDataset(list(val_subset), max_seq_len=max_seq_len)
    return DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


@torch.no_grad()
def compute_val_loss(
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

    return total_loss / max(total_tokens, 1)


@torch.no_grad()
def generate_samples(
    model: TinyTransformer,
    tokenizer: dict,
    device: torch.device,
    num_samples: int,
    max_new_tokens: int,
    temperature: float,
) -> List[str]:
    """Generate text samples from fixed prompts."""
    model.eval()
    stoi, itos = tokenizer["stoi"], tokenizer["itos"]

    prompts = [
        "Once upon a time",
        "One day a little",
        "The girl found a",
        "In the forest there",
        "A small cat was",
    ]

    samples = []
    for i in range(num_samples):
        prompt = prompts[i % len(prompts)]
        idx = torch.tensor(
            [[stoi.get(c, 0) for c in prompt]],
            dtype=torch.long,
            device=device,
        )
        out = model.generate(idx, max_new_tokens=max_new_tokens, temperature=temperature)
        text = "".join(itos[int(t)] for t in out[0].tolist())
        samples.append(text)

    return samples


def evaluate(config: EvalConfig) -> None:
    """Run full evaluation pipeline."""
    torch.manual_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    # Load data & tokenizer
    tokenizer = load_tokenizer(config.data_dir)
    all_data = load_domain_data(config.data_dir, config.domain)
    logger.info("Domain %d | Total sequences: %d", config.domain, len(all_data))

    # Validation loader (reproducible split)
    val_loader = create_val_loader(
        data=all_data,
        max_seq_len=config.max_seq_len,
        val_ratio=config.val_ratio,
        batch_size=config.batch_size,
        seed=config.seed,
    )
    logger.info("Validation sequences: %d", len(val_loader.dataset))

    # Load model
    model = TinyTransformer(
        vocab_size=tokenizer["vocab_size"],
        n_embd=256,
        n_head=8,
        n_layer=6,
        block_size=config.max_seq_len,
    ).to(device)

    ckpt = torch.load(config.checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    logger.info("Loaded checkpoint: %s", config.checkpoint_path)
    logger.info("Parameters: %s", f"{count_parameters(model):,}")

    # Validation loss
    val_loss = compute_val_loss(model, val_loader, device)
    logger.info("Validation loss: %.4f", val_loss)

    # Generation
    logger.info("Generating %d samples...", config.num_generate)
    samples = generate_samples(
        model=model,
        tokenizer=tokenizer,
        device=device,
        num_samples=config.num_generate,
        max_new_tokens=config.max_new_tokens,
        temperature=config.temperature,
    )

    print("\n" + "=" * 60)
    print("GENERATED SAMPLES")
    print("=" * 60)
    for i, text in enumerate(samples, 1):
        print(f"\n--- Sample {i} ---")
        print(text)
        print("-" * 40)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate StreamLM")
    parser.add_argument("--domain", type=int, default=1, help="Domain number")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/domain1_best.pt",
        help="Path to checkpoint",
    )
    parser.add_argument("--val-ratio", type=float, default=0.08, help="Validation split ratio")
    parser.add_argument("--num-generate", type=int, default=5, help="Number of samples to generate")
    args = parser.parse_args()

    config = EvalConfig(
        domain=args.domain,
        checkpoint_path=args.checkpoint,
        val_ratio=args.val_ratio,
        num_generate=args.num_generate,
    )
    evaluate(config)


if __name__ == "__main__":
    main()