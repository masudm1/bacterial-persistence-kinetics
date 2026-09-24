#!/usr/bin/env python3
"""Calibration-only diagnosis of PINN parameter collapse.

This script uses seeds 911--915. These seeds are excluded from both the
original production experiment and the subsequent confirmation experiment.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from core import (
    SYNTHETIC_INITIAL,
    SYNTHETIC_TRUE,
    make_time_split,
    mask_for_times,
    rmse,
    solve_total,
    synthetic_dataset,
)
from neural import (
    fitted_parameters,
    fit_neural,
    physics_diagnostics,
    predict_neural,
)


CANDIDATES = {
    "baseline": {
        "architecture": "single",
        "physics_variant": "global",
        "collocation_strategy": "uniform",
        "data_weight": 0.3,
        "initial_rate_grid": [(0.01, 0.3)],
    },
    "boundary_aware": {
        "architecture": "single",
        "physics_variant": "phase_balanced",
        "collocation_strategy": "phase_stratified",
        "data_weight": 0.3,
        "initial_rate_grid": [(0.01, 0.3)],
    },
    "piecewise": {
        "architecture": "piecewise",
        "physics_variant": "phase_balanced",
        "collocation_strategy": "phase_stratified",
        "data_weight": 0.3,
        "initial_rate_grid": [(0.01, 0.3)],
    },
    "piecewise_multistart": {
        "architecture": "piecewise",
        "physics_variant": "phase_balanced",
        "collocation_strategy": "phase_stratified",
        "data_weight": 0.3,
        "initial_rate_grid": [(1e-4, 0.01), (0.003, 0.1), (0.03, 1.0)],
    },
    "piecewise_multistart_strong_physics": {
        "architecture": "piecewise",
        "physics_variant": "phase_balanced",
        "collocation_strategy": "phase_stratified",
        "data_weight": 0.1,
        "initial_rate_grid": [(1e-4, 0.01), (0.003, 0.1), (0.03, 1.0)],
    },
}


def fit_candidate(
    rule: dict,
    times: np.ndarray,
    observed: np.ndarray,
    seed: int,
    epochs: int,
    lbfgs_steps: int,
):
    fits = []
    for start_id, initial_rates in enumerate(rule["initial_rate_grid"], start=1):
        result = fit_neural(
            times,
            observed,
            SYNTHETIC_INITIAL,
            12.0,
            seed + 100 * start_id,
            physics=True,
            real=False,
            epochs=epochs,
            lbfgs_steps=lbfgs_steps,
            data_weight=rule["data_weight"],
            architecture=rule["architecture"],
            physics_variant=rule["physics_variant"],
            collocation_strategy=rule["collocation_strategy"],
            initial_rates=initial_rates,
        )
        fits.append((result.final_loss, start_id, initial_rates, result))
    return min(fits, key=lambda item: item[0]), fits


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=1800)
    parser.add_argument("--lbfgs-steps", type=int, default=180)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    start_rows = []
    dense_time = np.linspace(0, 12, 600)

    for candidate_name, rule in CANDIDATES.items():
        for calibration_seed in range(911, 916):
            data = synthetic_dataset(calibration_seed, 0.10)
            split = make_time_split(data.time_h, calibration_seed + 40_000)
            train = mask_for_times(data.time_h, split.train_times)
            test = ~train
            selected, all_fits = fit_candidate(
                rule,
                data.time_h.to_numpy()[train],
                data.log_observed_total.to_numpy()[train],
                calibration_seed,
                args.epochs,
                args.lbfgs_steps,
            )
            for objective, start_id, initial_rates, fitted in all_fits:
                fitted_rates = fitted_parameters(fitted)
                start_rows.append(
                    {
                        "candidate": candidate_name,
                        "calibration_seed": calibration_seed,
                        "start_id": start_id,
                        "initial_alpha": initial_rates[0],
                        "initial_beta": initial_rates[1],
                        "training_objective": objective,
                        "fitted_alpha": fitted_rates["alpha"],
                        "fitted_beta": fitted_rates["beta"],
                        "selected": start_id == selected[1],
                    }
                )
            objective, selected_start, initial_rates, result = selected
            rates = fitted_parameters(result)
            network_dense = predict_neural(result, dense_time)
            ode_dense = np.log(
                solve_total(dense_time, SYNTHETIC_INITIAL, rates["alpha"], rates["beta"])
            )
            ode_train = np.log(
                solve_total(
                    data.time_h.to_numpy()[train],
                    SYNTHETIC_INITIAL,
                    rates["alpha"],
                    rates["beta"],
                )
            )
            clean_dense = np.log(
                solve_total(
                    dense_time,
                    SYNTHETIC_INITIAL,
                    SYNTHETIC_TRUE["alpha"],
                    SYNTHETIC_TRUE["beta"],
                )
            )
            diagnostics = physics_diagnostics(result, 12.0)
            phase_masks = {
                "pre": dense_time < 2.0,
                "treatment": (dense_time > 2.0) & (dense_time < 7.0),
                "post": dense_time > 7.0,
            }
            row = {
                "candidate": candidate_name,
                "calibration_seed": calibration_seed,
                "selected_start_id": selected_start,
                "selected_initial_alpha": initial_rates[0],
                "selected_initial_beta": initial_rates[1],
                "alpha": rates["alpha"],
                "beta": rates["beta"],
                "alpha_abs_log_error": abs(np.log(rates["alpha"] / SYNTHETIC_TRUE["alpha"])),
                "beta_abs_log_error": abs(np.log(rates["beta"] / SYNTHETIC_TRUE["beta"])),
                "joint_abs_log_error": abs(np.log(rates["alpha"] / SYNTHETIC_TRUE["alpha"])) + abs(np.log(rates["beta"] / SYNTHETIC_TRUE["beta"])),
                "network_train_rmse_log": rmse(
                    predict_neural(result, data.time_h.to_numpy()[train]),
                    data.log_observed_total.to_numpy()[train],
                ),
                "network_test_rmse_log": rmse(
                    predict_neural(result, data.time_h.to_numpy()[test]),
                    data.log_observed_total.to_numpy()[test],
                ),
                "exact_ode_train_rmse_log": rmse(
                    ode_train, data.log_observed_total.to_numpy()[train]
                ),
                "network_to_fitted_ode_rmse_log": rmse(network_dense, ode_dense),
                "training_objective": objective,
                **diagnostics,
            }
            for phase, mask in phase_masks.items():
                row[f"{phase}_network_clean_rmse_log"] = rmse(network_dense[mask], clean_dense[mask])
                row[f"{phase}_fitted_ode_clean_rmse_log"] = rmse(ode_dense[mask], clean_dense[mask])
            rows.append(row)
            pd.DataFrame(rows).to_csv(args.output_dir / "pinn_collapse_diagnostics.csv", index=False)
            pd.DataFrame(start_rows).to_csv(args.output_dir / "pinn_multistart_diagnostics.csv", index=False)
            print(f"{candidate_name}: calibration seed {calibration_seed}", flush=True)

    frame = pd.DataFrame(rows)
    metrics = [
        "joint_abs_log_error",
        "beta_abs_log_error",
        "network_test_rmse_log",
        "exact_ode_train_rmse_log",
        "network_to_fitted_ode_rmse_log",
        "pre_RN_rmse_h-1",
        "treatment_RN_rmse_h-1",
        "post_RN_rmse_h-1",
        "pre_RP_rmse_h-1",
        "treatment_RP_rmse_h-1",
        "post_RP_rmse_h-1",
    ]
    summary = frame.groupby("candidate")[metrics].median().sort_values("joint_abs_log_error")
    summary.to_csv(args.output_dir / "pinn_collapse_diagnostics_summary.csv")
    selected_name = str(summary.index[0])
    frozen = {
        "selection_basis": "minimum median joint absolute log-parameter error on calibration seeds 911-915",
        "calibration_seeds": [911, 912, 913, 914, 915],
        "confirmation_seeds_reserved": list(range(2001, 2021)),
        "candidate": selected_name,
        "rule": CANDIDATES[selected_name],
        "epochs": args.epochs,
        "lbfgs_steps": args.lbfgs_steps,
        "confirmation_results_consulted": False,
    }
    (args.output_dir / "frozen_pinn_rule.json").write_text(json.dumps(frozen, indent=2) + "\n")

    plot_metrics = [
        ("joint_abs_log_error", "Joint absolute log-parameter error"),
        ("network_to_fitted_ode_rmse_log", "Network-to-fitted-ODE RMSE"),
        ("treatment_RP_rmse_h-1", r"Treatment $R_P$ RMSE ($h^{-1}$)"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    order = list(summary.index)
    short = [name.replace("piecewise_", "pw_").replace("boundary_aware", "boundary") for name in order]
    for ax, (metric, label) in zip(axes, plot_metrics):
        values = [frame[frame.candidate.eq(name)][metric].to_numpy() for name in order]
        ax.boxplot(values, tick_labels=short, showfliers=True)
        ax.set_ylabel(label)
        ax.tick_params(axis="x", rotation=30)
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(args.output_dir / "fig_pinn_collapse_diagnostics.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"frozen candidate: {selected_name}")


if __name__ == "__main__":
    main()
