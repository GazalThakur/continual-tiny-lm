"""
Evaluation Matrix utility for Experiment 2 (BPE tokenizer, domains A-D).
Runs multiple checkpoints across multiple domains and builds a results table.
"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import DataLoader, random_split

from src.model import TinyTransformer
from src.eval_v2 import (
    TextDataset,
    load_bpe_tokenizer,
    get_pad_id,
    load_domain_data,
    compute_val_loss,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def evaluate_checkpoint_on_domain(
    checkpoint_path: str,
    domain: str,
    data_dir: str,
    max_seq_len: int,
    batch_size: int,
    val_ratio: float,
    seed: int,
    device: torch.device,
) -> float:
    tokenizer, tok_cfg = load_bpe_tokenizer(data_dir)
    vocab_size = tok_cfg["vocab_size"]
    pad_id = get_pad_id(tokenizer)

    data = load_domain_data(data_dir, domain)
    gen = torch.Generator().manual_seed(seed)
    val_size = int(len(data) * val_ratio)
    _, val_subset = random_split(data, [len(data) - val_size, val_size], generator=gen)
    val_loader = DataLoader(TextDataset(list(val_subset), max_seq_len, pad_id), batch_size=batch_size)

    model = TinyTransformer(
        vocab_size=vocab_size,
        n_embd=256,
        n_head=8,
        n_layer=6,
        block_size=max_seq_len,
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])

    return compute_val_loss(model, val_loader, device)


def run_matrix(
    checkpoints: List[str],
    domains: List[str],
    data_dir: str = "data_v2",
    max_seq_len: int = 256,
    batch_size: int = 32,
    val_ratio: float = 0.08,
    seed: int = 42,
    output_csv: str | None = None,
) -> Dict[str, Dict[str, float]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    results: Dict[str, Dict[str, float]] = {}

    for ckpt_path in checkpoints:
        ckpt_name = Path(ckpt_path).stem
        results[ckpt_name] = {}
        logger.info("Evaluating checkpoint: %s", ckpt_path)

        for domain in domains:
            logger.info("  → Domain %s ...", domain)
            try:
                loss = evaluate_checkpoint_on_domain(
                    checkpoint_path=ckpt_path,
                    domain=domain,
                    data_dir=data_dir,
                    max_seq_len=max_seq_len,
                    batch_size=batch_size,
                    val_ratio=val_ratio,
                    seed=seed,
                    device=device,
                )
                results[ckpt_name][domain] = loss
                logger.info("     Val Loss: %.4f", loss)
            except Exception as e:
                logger.error("     Failed: %s", str(e))
                results[ckpt_name][domain] = float("nan")

    print("\n" + "=" * 80)
    print("EVALUATION MATRIX (Experiment 2)")
    print("=" * 80)
    header = f"{'Checkpoint':<30}" + "".join([f"Domain {d:<10}" for d in domains])
    print(header)
    print("-" * 80)
    for ckpt_name, domain_scores in results.items():
        row = f"{ckpt_name:<30}"
        for d in domains:
            score = domain_scores.get(d, float("nan"))
            row += f"{score:<12.4f}"
        print(row)
    print("=" * 80)

    if output_csv:
        with open(output_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["checkpoint"] + [f"domain_{d}" for d in domains])
            for ckpt_name, domain_scores in results.items():
                row = [ckpt_name] + [domain_scores.get(d, "") for d in domains]
                writer.writerow(row)
        logger.info("Results saved to %s", output_csv)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run evaluation matrix (Experiment 2)")
    parser.add_argument("--checkpoints", type=str, nargs="+", required=True)
    parser.add_argument("--domains", type=str, nargs="+", default=["A", "B", "C", "D"])
    parser.add_argument("--data-dir", type=str, default="data_v2")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--val-ratio", type=float, default=0.08)
    parser.add_argument("--output-csv", type=str, default=None)
    args = parser.parse_args()

    run_matrix(
        checkpoints=args.checkpoints,
        domains=args.domains,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        val_ratio=args.val_ratio,
        output_csv=args.output_csv,
    )


if __name__ == "__main__":
    main()