#!/usr/bin/env python3
"""Reconstruct post-washout division lags from the deposited tracking files."""

from __future__ import annotations

import argparse
import itertools
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import weibull_min


def acquisition_settings(folder: Path) -> dict[str, tuple[int, int, float]]:
    table = pd.read_csv(folder / "AcquisitionSetting.txt", sep="\t")
    return {
        str(int(row.ExperimentDate)): (
            int(row.DrugAddedFrame),
            int(row.DrugDurationFrameNumber),
            float(row.IntervalTime),
        )
        for row in table.itertuples(index=False)
    }


def extract_folder(folder: Path, condition: str) -> pd.DataFrame:
    settings = acquisition_settings(folder)
    rows = []
    for path in sorted(folder.glob("*Results*.xls")):
        match = re.match(r"(\d{6})_Results", path.name)
        if match is None:
            continue
        date = match.group(1)
        added, duration, interval_min = settings[date]
        washout_frame = added + duration
        tracking = pd.read_csv(path, sep="\t")
        duplicate_counts = tracking.PreviousCell.value_counts()
        parent_ids = duplicate_counts[duplicate_counts.eq(2)].index
        # The duplicated references identify one parent. Its own row is the
        # last intact-parent frame; the two child rows occur one frame later.
        parent_index_column = tracking.columns[0]
        division_frames = np.sort(
            tracking.loc[tracking[parent_index_column].isin(parent_ids), "Slice"].unique()
        )
        eligible = division_frames[division_frames > washout_frame]
        if len(eligible) == 0:
            rows.append({"condition": condition, "file": path.name, "washout_frame": washout_frame, "division_frame": np.nan, "lag_h": np.nan, "censored": True})
            continue
        first_division = int(eligible.min())
        rows.append(
            {
                "condition": condition,
                "file": path.name,
                "washout_frame": washout_frame,
                "division_frame": first_division,
                "interval_min": interval_min,
                "lag_h": (first_division - washout_frame) * interval_min / 60.0,
                "censored": False,
            }
        )
    return pd.DataFrame(rows)


def summarize(frame: pd.DataFrame, bootstrap_resamples: int = 5000) -> pd.DataFrame:
    rng = np.random.default_rng(20250905)
    rows = []
    for condition, group in frame.groupby("condition"):
        lag = group.loc[~group.censored, "lag_h"].to_numpy()
        if len(lag) <= 8:
            bootstrap_rates = np.fromiter(
                (1.0 / lag[list(indices)].mean() for indices in itertools.product(range(len(lag)), repeat=len(lag))),
                dtype=float,
                count=len(lag) ** len(lag),
            )
            bootstrap_method = "exact enumeration of all n^n resamples"
        else:
            bootstrap_rates = np.empty(max(bootstrap_resamples, 100_000))
            for index in range(len(bootstrap_rates)):
                sample = rng.choice(lag, size=len(lag), replace=True)
                bootstrap_rates[index] = 1.0 / sample.mean()
            bootstrap_method = f"Monte Carlo percentile bootstrap ({len(bootstrap_rates)} resamples)"
        weibull_shape, _, weibull_scale = weibull_min.fit(lag, floc=0)
        rows.append(
            {
                "condition": condition,
                "n_lineages": len(group),
                "n_events": len(lag),
                "n_censored": int(group.censored.sum()),
                "mean_lag_h": lag.mean(),
                "median_lag_h": np.median(lag),
                "inverse_mean_rate_h-1": 1.0 / lag.mean(),
                "bootstrap_rate_ci_lower": np.quantile(bootstrap_rates, 0.025),
                "bootstrap_rate_ci_upper": np.quantile(bootstrap_rates, 0.975),
                "bootstrap_method": bootstrap_method,
                "weibull_shape": weibull_shape,
                "weibull_scale_h": weibull_scale,
            }
        )
    return pd.DataFrame(rows)


def make_figure(frame: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, condition in zip(axes, ("ciprofloxacin", "ampicillin")):
        values = frame.loc[(frame.condition.eq(condition)) & (~frame.censored), "lag_h"]
        ax.hist(values, bins=max(4, int(np.sqrt(len(values)))), color="#4c78a8", edgecolor="white")
        ax.axvline(values.mean(), color="#9b1c1c", linestyle="--", label=f"Mean = {values.mean():.2f} h")
        ax.set_xlabel("First-division lag after washout (h)")
        ax.set_ylabel("Lineages")
        ax.set_title(condition.capitalize())
        ax.legend(frameon=False)
        ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dryad-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.dryad_dir / "Tracking_x40"
    frames = [
        extract_folder(root / "Data_MG1655_M9exp_CPFX1", "ciprofloxacin"),
        extract_folder(root / "Data_MG1655_LBexp_Amp200", "ampicillin"),
    ]
    combined = pd.concat(frames, ignore_index=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    combined.to_csv(args.output_dir / "single_cell_lags.csv", index=False)
    summarize(combined).to_csv(args.output_dir / "single_cell_lag_summary.csv", index=False)
    make_figure(combined, args.output_dir / "fig_single_cell_lags.png")


if __name__ == "__main__":
    main()
