#!/usr/bin/env python3
"""Biological-replicate cluster bootstrap for corrected Umetani NLLS fits."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from core import fit_nlls_real
from run_pipeline import MEASURED_FRACTIONS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resamples", type=int, default=200)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(args.real_data)
    rng = np.random.default_rng(20250906)
    rows = []
    for condition in ("ciprofloxacin", "ampicillin"):
        subset = data[data.condition.eq(condition)]
        replicates = np.sort(subset.replicate.unique())
        for scenario, fraction in (
            ("assumed_0.1pct", 0.001),
            ("classification_anchor", MEASURED_FRACTIONS[condition]),
        ):
            point = fit_nlls_real(subset.time_h.to_numpy(), subset.log_relative_abundance.to_numpy(), fraction)
            starts = np.array(
                [
                    (point["D"], max(point["alpha"], 1e-7), point["beta"]),
                    (-4.0, 1e-3, 0.5),
                    (-8.0, 0.1, 2.0),
                ]
            )
            for bootstrap_id in range(1, args.resamples + 1):
                sampled = rng.choice(replicates, size=len(replicates), replace=True)
                pieces = [subset[subset.replicate.eq(rep)] for rep in sampled]
                sample = pd.concat(pieces, ignore_index=True)
                fit = fit_nlls_real(
                    sample.time_h.to_numpy(),
                    sample.log_relative_abundance.to_numpy(),
                    fraction,
                    starts=starts,
                )
                rows.append(
                    {
                        "condition": condition,
                        "initial_condition_scenario": scenario,
                        "initial_fraction": fraction,
                        "bootstrap_id": bootstrap_id,
                        "sampled_replicates": ";".join(str(int(x)) for x in sampled),
                        **fit,
                    }
                )
                if bootstrap_id % 20 == 0:
                    pd.DataFrame(rows).to_csv(args.output_dir / "cluster_bootstrap_trials.csv", index=False)
                    print(f"{condition} {scenario}: {bootstrap_id}/{args.resamples}", flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(args.output_dir / "cluster_bootstrap_trials.csv", index=False)
    summary_rows = []
    for keys, group in frame.groupby(["condition", "initial_condition_scenario", "initial_fraction"]):
        for parameter in ("D", "alpha", "beta"):
            summary_rows.append(
                {
                    "condition": keys[0],
                    "initial_condition_scenario": keys[1],
                    "initial_fraction": keys[2],
                    "parameter": parameter,
                    "median": group[parameter].median(),
                    "ci_2.5pct": group[parameter].quantile(0.025),
                    "ci_97.5pct": group[parameter].quantile(0.975),
                    "n_resamples": len(group),
                    "resampling_unit": "biological replicate (whole observed time course)",
                }
            )
    pd.DataFrame(summary_rows).to_csv(args.output_dir / "cluster_bootstrap_summary.csv", index=False)


if __name__ == "__main__":
    main()
