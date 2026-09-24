#!/usr/bin/env python3
"""Pre-specify the PINN loss weight on simulations excluded from evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from core import SYNTHETIC_INITIAL, SYNTHETIC_TRUE, make_time_split, mask_for_times, synthetic_dataset
from neural import fitted_parameters, fit_neural


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=1800)
    parser.add_argument("--lbfgs-steps", type=int, default=180)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for weight in (0.03, 0.1, 0.3, 1.0, 3.0, 10.0):
        for seed in (901, 902, 903):
            data = synthetic_dataset(seed, 0.10)
            split = make_time_split(data.time_h, seed + 40_000)
            train = mask_for_times(data.time_h, split.train_times)
            result = fit_neural(
                data.time_h.to_numpy()[train],
                data.log_observed_total.to_numpy()[train],
                SYNTHETIC_INITIAL,
                12.0,
                seed,
                physics=True,
                real=False,
                epochs=args.epochs,
                lbfgs_steps=args.lbfgs_steps,
                data_weight=weight,
            )
            fitted = fitted_parameters(result)
            rows.append(
                {
                    "data_weight": weight,
                    "calibration_seed": seed,
                    "alpha": fitted["alpha"],
                    "beta": fitted["beta"],
                    "alpha_abs_log_error": abs(np.log(fitted["alpha"] / SYNTHETIC_TRUE["alpha"])),
                    "beta_abs_log_error": abs(np.log(fitted["beta"] / SYNTHETIC_TRUE["beta"])),
                    "joint_abs_log_error": abs(np.log(fitted["alpha"] / SYNTHETIC_TRUE["alpha"])) + abs(np.log(fitted["beta"] / SYNTHETIC_TRUE["beta"])),
                    "training_objective": result.final_loss,
                }
            )
            pd.DataFrame(rows).to_csv(args.output, index=False)
            print(f"weight={weight:g}, seed={seed}", flush=True)
    frame = pd.DataFrame(rows)
    summary = frame.groupby("data_weight").joint_abs_log_error.median().sort_values()
    summary.rename("median_joint_abs_log_error").to_csv(args.output.with_name("pinn_weight_calibration_summary.csv"))
    print(f"selected data weight: {summary.index[0]:g}")


if __name__ == "__main__":
    main()
