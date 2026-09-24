#!/usr/bin/env python3
"""Plot the actual synthetic data-generating process used by the benchmark."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from core import synthetic_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    data = synthetic_dataset(args.seed, 0.10)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(data.time_h, data.latent_total, color="black", linewidth=2, label="Deterministic ODE total")
    ax.scatter(data.time_h, data.observed_total, s=22, color="#888888", alpha=0.7, label="Gaussian measurement error (10%)")
    ax.axvspan(2, 7, color="#8e44ad", alpha=0.12, label="Antibiotic exposure")
    ax.set_yscale("log")
    ax.set_xlabel("Time (h)")
    ax.set_ylabel(r"Total abundance $N(t)+P(t)$")
    ax.set_title("Synthetic biphasic trajectory with heteroscedastic measurement error")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
