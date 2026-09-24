#!/usr/bin/env python3
"""Re-run parameter-separation sensitivity under the corrected noise model."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from core import SYNTHETIC_INITIAL, fit_nlls_synthetic, make_time_split, mask_for_times, solve_total
from neural import fitted_parameters, fit_neural


CONDITIONS = [
    (0.0001, 0.1),
    (0.001, 0.1),
    (0.01, 0.1),
    (0.05, 0.1),
    (0.1, 0.1),
    (0.1, 0.01),
]


def noisy_data(alpha: float, beta: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    times = np.linspace(0, 12, 100)
    clean = solve_total(times, SYNTHETIC_INITIAL, alpha, beta)
    rng = np.random.default_rng(50_000 + seed + int(round(1e5 * alpha)) + int(round(1e4 * beta)))
    observed = np.maximum(clean + rng.normal(0, 0.1 * clean), 1.0)
    return times, np.log(observed)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=2500)
    parser.add_argument("--lbfgs-steps", type=int, default=250)
    parser.add_argument("--data-weight", type=float, default=0.3)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for alpha_true, beta_true in CONDITIONS:
        ratio = beta_true / alpha_true
        for trial in range(1, 6):
            times, observed = noisy_data(alpha_true, beta_true, trial)
            split = make_time_split(times, 60_000 + trial)
            train = mask_for_times(times, split.train_times)
            result = fit_nlls_synthetic(times[train], observed[train])
            rows.append(
                {
                    "method": "NLLS",
                    "alpha_true": alpha_true,
                    "beta_true": beta_true,
                    "beta_alpha_ratio": ratio,
                    "trial": trial,
                    "alpha": result["alpha"],
                    "beta": result["beta"],
                    "alpha_relative_error_pct": 100 * abs(result["alpha"] / alpha_true - 1),
                    "beta_relative_error_pct": 100 * abs(result["beta"] / beta_true - 1),
                }
            )
            if ratio in (1.0, 1000.0) and trial <= 3:
                pinn = fit_neural(
                    times[train],
                    observed[train],
                    SYNTHETIC_INITIAL,
                    12.0,
                    trial,
                    physics=True,
                    real=False,
                    epochs=args.epochs,
                    lbfgs_steps=args.lbfgs_steps,
                    data_weight=args.data_weight,
                )
                fitted = fitted_parameters(pinn)
                rows.append(
                    {
                        "method": "PINN",
                        "alpha_true": alpha_true,
                        "beta_true": beta_true,
                        "beta_alpha_ratio": ratio,
                        "trial": trial,
                        "alpha": fitted["alpha"],
                        "beta": fitted["beta"],
                        "alpha_relative_error_pct": 100 * abs(fitted["alpha"] / alpha_true - 1),
                        "beta_relative_error_pct": 100 * abs(fitted["beta"] / beta_true - 1),
                    }
                )
        print(f"ratio {ratio:g}", flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(args.output_dir / "ratio_sweep_trials.csv", index=False)
    summary = frame.groupby(["method", "alpha_true", "beta_true", "beta_alpha_ratio"])[["alpha_relative_error_pct", "beta_relative_error_pct"]].agg(["median", "mean", "std", "count"])
    summary.to_csv(args.output_dir / "ratio_sweep_summary.csv")

    nlls = frame[frame.method.eq("NLLS")]
    grouped = nlls.groupby("beta_alpha_ratio")
    ratios = np.array(sorted(grouped.groups))
    alpha_median = np.array([grouped.get_group(x).alpha_relative_error_pct.median() for x in ratios])
    beta_median = np.array([grouped.get_group(x).beta_relative_error_pct.median() for x in ratios])
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ratios, alpha_median, "o-", label=r"$\alpha$ relative error")
    ax.plot(ratios, beta_median, "s-", label=r"$\beta$ relative error")
    ax.axvline(100, color="#777777", linestyle=":", label="Main synthetic condition")
    ax.set_xscale("log")
    ax.set_xlabel(r"True $\beta/\alpha$")
    ax.set_ylabel("Median relative error (%)")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(args.output_dir / "fig_ratio_sweep_corrected.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
