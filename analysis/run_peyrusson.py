#!/usr/bin/env python3
"""Matched replicate-level benchmark for Peyrusson extracellular oxacillin."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from core import Split, extract_peyrusson_oxacillin, fit_nlls_real, mask_for_times, predict_nlls, rmse
from neural import fitted_parameters, fit_neural, initial_condition_error, predict_neural


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workbook", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=2500)
    parser.add_argument("--lbfgs-steps", type=int, default=250)
    parser.add_argument("--data-weight", type=float, default=0.3)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = extract_peyrusson_oxacillin(args.workbook)
    data.to_csv(args.output_dir / "peyrusson_replicate_log_data.csv", index=False)
    fraction = 0.001
    initial = (1.0 - fraction, fraction)
    rows = []
    unique_times = np.sort(data.time_h.unique())
    held_out_times = unique_times[unique_times > 0]
    for trial, held_out in enumerate(held_out_times, start=1):
        split = Split(train_times=unique_times[unique_times != held_out], test_times=np.array([held_out]))
        train = mask_for_times(data.time_h, split.train_times)
        test = ~train
        t_train = data.time_h.to_numpy()[train]
        y_train = data.log_relative_abundance.to_numpy()[train]
        t_test = data.time_h.to_numpy()[test]
        y_test = data.log_relative_abundance.to_numpy()[test]
        common = {
            "condition": "peyrusson_extracellular_oxacillin",
            "initial_fraction": fraction,
            "trial": trial,
            "train_times_h": ";".join(str(x) for x in split.train_times),
            "test_times_h": ";".join(str(x) for x in split.test_times),
            "n_train_observations": int(train.sum()),
            "n_test_observations": int(test.sum()),
        }
        nlls = fit_nlls_real(t_train, y_train, fraction)
        pred = predict_nlls(t_test, nlls, initial, synthetic=False)
        rows.append(common | {"method": "NLLS", "test_rmse_observed_log": rmse(pred, y_test), **nlls, "initial_condition_max_abs_error": 0.0})
        nn = fit_neural(t_train, y_train, initial, 24.0, trial, False, True, epochs=args.epochs, lbfgs_steps=0)
        pred = predict_neural(nn, t_test)
        rows.append(common | {"method": "NN", "test_rmse_observed_log": rmse(pred, y_test), "D": np.nan, "alpha": np.nan, "beta": np.nan, "train_sse": nn.final_loss, "initial_condition_max_abs_error": initial_condition_error(nn)})
        pinn = fit_neural(t_train, y_train, initial, 24.0, trial, True, True, epochs=args.epochs, lbfgs_steps=args.lbfgs_steps, data_weight=args.data_weight)
        pred = predict_neural(pinn, t_test)
        rows.append(common | {"method": "PINN", "test_rmse_observed_log": rmse(pred, y_test), **fitted_parameters(pinn), "train_sse": pinn.final_loss, "initial_condition_max_abs_error": initial_condition_error(pinn)})
        pd.DataFrame(rows).to_csv(args.output_dir / "peyrusson_trials.csv", index=False)
        print(f"Peyrusson leave-one-time-out fold {trial}/{len(held_out_times)}", flush=True)
    summary = pd.DataFrame(rows).groupby("method")[["test_rmse_observed_log", "D", "alpha", "beta"]].agg(["median", "mean", "std", "count"])
    summary.to_csv(args.output_dir / "peyrusson_summary.csv")


if __name__ == "__main__":
    main()
