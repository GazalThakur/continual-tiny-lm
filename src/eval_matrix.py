"""
Evaluation Matrix utility for StreamLM.
Runs multiple checkpoints across multiple domains and builds a results table.
"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import DataLoader

from src.model import TinyTransformer
from src.eval import (
    TextDataset,
    load_tokenizer,
    load_domain_data,
    create_val_loader,
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
    domain: int,
    data_dir: str,
    max_seq_len: int,
    batch_size: int,
    val_ratio: float,
    seed: int,
    device: torch.device,
) -> float:
    """Evaluate one checkpoint on one domain and return validation loss."""
    tokenizer = load_tokenizer(data_dir)
    data = load_domain_data(data_dir, domain)

    val_loader = create_val_loader(
        data=data,
        max_seq_len=max_seq_len,
        val_ratio=val_ratio,
        batch_size=batch_size,
        seed=seed,
    )

    model = TinyTransformer(
        vocab_size=tokenizer["vocab_size"],
        n_embd=256,
        n_head=8,
        n_layer=6,
        block_size=max_seq_len,
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])

    val_loss = compute_val_loss(model, val_loader, device)
    return val_loss


def run_matrix(
    checkpoints: List[str],
    domains: List[int],
    data_dir: str = "data",
    max_seq_len: int = 256,
    batch_size: int = 32,
    val_ratio: float = 0.08,
    seed: int = 42,
    output_csv: str | None = None,
) -> Dict[str, Dict[int, float]]:
    """
    Run full evaluation matrix.
    Returns: {checkpoint_name: {domain: val_loss}}
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    results: Dict[str, Dict[int, float]] = {}

    for ckpt_path in checkpoints:
        ckpt_name = Path(ckpt_path).stem
        results[ckpt_name] = {}
        logger.info("Evaluating checkpoint: %s", ckpt_path)

        for domain in domains:
            logger.info("  → Domain %d ...", domain)
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

    # Print nice table
    print("\n" + "=" * 80)
    print("EVALUATION MATRIX")
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

    # Optionally save to CSV
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
    parser = argparse.ArgumentParser(description="Run evaluation matrix across checkpoints and domains")
    parser.add_argument(
        "--checkpoints",
        type=str,
        nargs="+",
        required=True,
        help="List of checkpoint paths",
    )
    parser.add_argument(
        "--domains",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4],
        help="List of domain numbers to evaluate on",
    )
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--val-ratio", type=float, default=0.08)
    parser.add_argument("--output-csv", type=str, default=None, help="Optional path to save results as CSV")
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