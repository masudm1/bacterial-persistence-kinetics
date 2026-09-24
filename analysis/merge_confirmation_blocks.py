#!/usr/bin/env python3
"""Merge deterministic parallel confirmation blocks and regenerate summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from run_pipeline import paired_tests


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--blocks-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    files = [args.blocks_root / f"block_{index}" / "confirmation_trials.csv" for index in range(4)]
    if not all(path.exists() for path in files):
        raise FileNotFoundError("All four confirmation blocks are required")
    frame = pd.concat([pd.read_csv(path) for path in files], ignore_index=True)
    frame = frame.sort_values(["tier", "seed", "method"]).reset_index(drop=True)
    # Block-local trial numbers repeat. Re-index trials from the sorted,
    # globally unique seeds so the exported table has one unambiguous trial
    # identifier per dataset and noise tier.
    for tier, indices in frame.groupby("tier").groups.items():
        seeds = sorted(frame.loc[indices, "seed"].unique())
        seed_to_trial = {seed: trial for trial, seed in enumerate(seeds, start=1)}
        frame.loc[indices, "trial"] = frame.loc[indices, "seed"].map(seed_to_trial)
    frame["trial"] = frame["trial"].astype(int)
    if len(frame) != 180:
        raise ValueError(f"Expected 180 records, found {len(frame)}")
    if not frame.groupby(["tier", "seed"]).method.nunique().eq(3).all():
        raise ValueError("Every confirmation dataset must contain all three methods")
    if frame.groupby(["tier", "seed"])[["train_times_h", "test_times_h"]].nunique().to_numpy().max() != 1:
        raise ValueError("Methods do not share identical time splits")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_dir / "confirmation_trials.csv", index=False)
    metrics = [
        "alpha_relative_error_pct",
        "beta_relative_error_pct",
        "test_rmse_observed_log",
        "test_rmse_latent_log",
        "network_to_fitted_ode_rmse_log",
    ]
    frame.groupby(["tier", "method"])[metrics].agg(["median", "mean", "std", "count"]).to_csv(
        args.output_dir / "confirmation_summary.csv"
    )
    paired_tests(frame).to_csv(
        args.output_dir / "confirmation_paired_tests.csv", index=False
    )
    manifest = {
        "confirmation_seeds": sorted(frame.seed.unique().astype(int).tolist()),
        "trials_per_noise_tier": int(frame.groupby("tier").seed.nunique().min()),
        "completed_records": len(frame),
        "parallel_blocks": 4,
        "frozen_rule_sha256": frame.frozen_rule_sha256.unique().tolist(),
    }
    (args.output_dir / "confirmation_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    tier_order = ["low", "moderate", "high"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for method, color in (("NLLS", "#1f4e99"), ("PINN", "#9b1c1c")):
        subset = frame[frame.method.eq(method)]
        axes[0].plot(tier_order, [subset[subset.tier.eq(t)].alpha_relative_error_pct.median() for t in tier_order], "o-", label=method, color=color)
        axes[1].plot(tier_order, [subset[subset.tier.eq(t)].beta_relative_error_pct.median() for t in tier_order], "o-", label=method, color=color)
    for method, color in (("NLLS", "#1f4e99"), ("PINN", "#9b1c1c"), ("NN", "#4b8b3b")):
        subset = frame[frame.method.eq(method)]
        axes[2].plot(tier_order, [subset[subset.tier.eq(t)].test_rmse_observed_log.median() for t in tier_order], "o-", label=method, color=color)
    axes[0].set_ylabel(r"Median $\alpha$ relative error (%)")
    axes[1].set_ylabel(r"Median $\beta$ relative error (%)")
    axes[2].set_ylabel("Median held-out RMSE (natural-log units)")
    for ax in axes:
        ax.set_xlabel("Measurement-noise tier")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(args.output_dir / "fig_frozen_confirmation.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
