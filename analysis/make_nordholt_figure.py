#!/usr/bin/env python3
"""Plot audited Nordholt replicate data and full-data fitted trajectories."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CONDITIONS = ("nordholt_h2o2", "nordholt_bac", "nordholt_ddac")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--curves", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = pd.read_csv(args.data)
    curves = pd.read_csv(args.curves)

    labels = {"nordholt_h2o2": r"H$_2$O$_2$", "nordholt_bac": "BAC", "nordholt_ddac": "DDAC"}
    styles = {
        "NLLS": ("#1f4e99", "-"),
        "NN": ("#4b8b3b", "--"),
        "PINN-network": ("#9b1c1c", "-"),
        "PINN-implied-ODE": ("#9b1c1c", ":"),
    }
    fig, axes = plt.subplots(1, 3, figsize=(15.6, 4.5))
    for axis, condition in zip(axes, CONDITIONS):
        condition_raw = raw[raw.condition.eq(condition)]
        observed = condition_raw[~condition_raw.is_zero_count]
        last_positive_min = float(observed.time_min.max())
        final_min = float(condition_raw.time_min.max())
        for replicate, group in observed.groupby("replicate"):
            axis.scatter(
                group.time_min,
                group.log_relative_abundance / np.log(10),
                s=18,
                alpha=0.55,
                label="Biological replicates" if replicate == 1 else None,
            )
        zero_times = condition_raw[condition_raw.is_zero_count].time_min
        if len(zero_times):
            floor = float(observed.log_relative_abundance.min() / np.log(10) - 0.35)
            axis.scatter(
                zero_times,
                np.full(len(zero_times), floor),
                marker="v",
                color="black",
                s=20,
                label="Zero count (not fitted)",
                zorder=4,
            )
        if last_positive_min < final_min:
            axis.axvspan(last_positive_min, final_min, color="#777777", alpha=0.08)
            axis.text(
                (last_positive_min + final_min) / 2,
                0.97,
                "extrapolation only",
                ha="center",
                va="top",
                transform=axis.get_xaxis_transform(),
                fontsize=8,
                color="#555555",
            )
        for method, (color, style) in styles.items():
            line = curves[(curves.condition.eq(condition)) & (curves.method.eq(method))].copy()
            line["time_min"] = line.time_h * 60.0
            fitted = line[line.time_min <= last_positive_min + 1e-10]
            extrapolated = line[line.time_min >= last_positive_min - 1e-10]
            axis.plot(
                fitted.time_min,
                fitted.log_relative_abundance / np.log(10),
                style,
                color=color,
                lw=2,
                label=method,
            )
            if last_positive_min < final_min:
                axis.plot(
                    extrapolated.time_min,
                    extrapolated.log_relative_abundance / np.log(10),
                    style,
                    color=color,
                    lw=1.4,
                    alpha=0.3,
                )
        axis.set_title(labels[condition])
        axis.set_xlabel("Time (min)")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel(r"$\log_{10}[Y(t)/Y(0)]$")
    axes[-1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
