#!/usr/bin/env python3
"""Matched replicate-level analysis of the Nordholt disinfectant data."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.linalg import expm
from scipy.optimize import differential_evolution, least_squares
from scipy.stats import wilcoxon

from core import Split, extract_nordholt_disinfectants, make_time_split, mask_for_times, rmse
from neural import (
    fitted_parameters,
    fit_neural,
    initial_condition_error,
    physics_diagnostics,
    predict_neural,
)


CONDITIONS = ("nordholt_h2o2", "nordholt_bac", "nordholt_ddac")
INITIAL_FRACTION = 0.001
PINN_STARTS = (
    (5.0, 1e-5, 0.01),
    (30.0, 1e-3, 1.0),
    (100.0, 0.1, 30.0),
    (300.0, 1.0, 300.0),
)
LOG_BOUNDS = (
    np.log(np.array([1e-3, 1e-8, 1e-8])),
    np.log(np.array([1000.0, 1000.0, 1000.0])),
)


def exact_log_total(times: np.ndarray, D: float, alpha: float, beta: float) -> np.ndarray:
    """Exact constant-coefficient treatment-phase solution in log-total space."""
    initial = np.array([1.0 - INITIAL_FRACTION, INITIAL_FRACTION])
    matrix = np.array([[D - alpha, beta], [alpha, -beta]], dtype=float)
    totals = np.array([(expm(matrix * float(time)) @ initial).sum() for time in times])
    return np.log(np.maximum(totals, np.finfo(float).tiny))


def nlls_residual(encoded: np.ndarray, times: np.ndarray, observed: np.ndarray) -> np.ndarray:
    D_abs, alpha, beta = np.exp(encoded)
    unique, inverse = np.unique(times, return_inverse=True)
    prediction = exact_log_total(unique, -D_abs, alpha, beta)[inverse]
    return prediction - observed


def fit_global_nlls(
    times: np.ndarray, observed: np.ndarray, seed: int
) -> tuple[dict[str, float], list[dict[str, float]]]:
    """Differential evolution followed by least-squares polishing.

    Local runs are returned as diagnostics only; they never override the global
    search result.
    """
    bounds = list(zip(LOG_BOUNDS[0], LOG_BOUNDS[1]))
    objective = lambda encoded: float(np.sum(nlls_residual(encoded, times, observed) ** 2))
    global_result = differential_evolution(
        objective,
        bounds,
        seed=seed,
        maxiter=350,
        popsize=15,
        tol=1e-9,
        polish=False,
        updating="immediate",
        workers=1,
    )
    polished = least_squares(
        nlls_residual,
        global_result.x,
        args=(times, observed),
        bounds=LOG_BOUNDS,
        max_nfev=10000,
    )
    D_abs, alpha, beta = np.exp(polished.x)
    best = {
        "D": -float(D_abs),
        "alpha": float(alpha),
        "beta": float(beta),
        "train_sse": float(np.sum(polished.fun**2)),
        "global_objective_before_polish": float(global_result.fun),
        "global_nfev": int(global_result.nfev),
    }

    diagnostics = []
    for start_id, (D0, alpha0, beta0) in enumerate(PINN_STARTS):
        local = least_squares(
            nlls_residual,
            np.log([D0, alpha0, beta0]),
            args=(times, observed),
            bounds=LOG_BOUNDS,
            max_nfev=10000,
        )
        local_D, local_alpha, local_beta = np.exp(local.x)
        diagnostics.append(
            {
                "start_id": start_id,
                "initial_D_abs": D0,
                "initial_alpha": alpha0,
                "initial_beta": beta0,
                "D": -float(local_D),
                "alpha": float(local_alpha),
                "beta": float(local_beta),
                "sse": float(np.sum(local.fun**2)),
                "converged": bool(local.success),
            }
        )
    return best, diagnostics


def fit_best_nn(
    times: np.ndarray,
    observed: np.ndarray,
    final_time: float,
    seed: int,
    epochs: int,
):
    initial = (1.0 - INITIAL_FRACTION, INITIAL_FRACTION)
    candidates = [
        fit_neural(times, observed, initial, final_time, seed * 10 + offset, False, True, epochs=epochs, lbfgs_steps=0)
        for offset in range(3)
    ]
    return min(candidates, key=lambda result: result.final_loss)


def fit_best_pinn(
    times: np.ndarray,
    observed: np.ndarray,
    final_time: float,
    seed: int,
    epochs: int,
    lbfgs_steps: int,
    data_weight: float,
    weight_seeds: int,
):
    """Use prespecified rate starts and weight seeds; select by training objective."""
    initial = (1.0 - INITIAL_FRACTION, INITIAL_FRACTION)
    candidates = []
    for start_id, (D_abs, alpha, beta) in enumerate(PINN_STARTS):
        for weight_seed_index in range(weight_seeds):
            result = fit_neural(
                times,
                observed,
                initial,
                final_time,
                seed * 100 + start_id * 10 + weight_seed_index,
                True,
                True,
                epochs=epochs,
                lbfgs_steps=lbfgs_steps,
                data_weight=data_weight,
                initial_rates=(alpha, beta),
                initial_D_abs=D_abs,
                residual_time_scale=final_time,
            )
            candidates.append((start_id, weight_seed_index, result))
    return min(candidates, key=lambda item: item[2].final_loss)


def holm_adjust(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    adjusted = np.empty_like(values, dtype=float)
    running = 0.0
    total = len(values)
    for rank, index in enumerate(order):
        running = max(running, (total - rank) * values[index])
        adjusted[index] = min(1.0, running)
    return adjusted


def paired_tests(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for condition in CONDITIONS:
        subset = frame[frame.condition.eq(condition)]
        pivot = subset.pivot(index="trial", columns="method", values="test_rmse_observed_log")
        for first, second in (("PINN", "NLLS"), ("PINN", "NN")):
            difference = pivot[first] - pivot[second]
            statistic, p_value = wilcoxon(difference, alternative="two-sided", method="exact")
            rows.append(
                {
                    "condition": condition,
                    "metric": "test_rmse_observed_log",
                    "method_1": first,
                    "method_2": second,
                    "W": statistic,
                    "p_two_sided": p_value,
                    "median_paired_difference_method1_minus_method2": float(np.median(difference)),
                    "n_pairs": len(difference),
                }
            )
    result = pd.DataFrame(rows)
    result["p_holm"] = holm_adjust(result.p_two_sided.to_numpy())
    return result


def r_squared(observed: np.ndarray, predicted: np.ndarray) -> float:
    denominator = float(np.sum((observed - np.mean(observed)) ** 2))
    return float(1.0 - np.sum((observed - predicted) ** 2) / denominator)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workbook", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--condition", choices=CONDITIONS)
    parser.add_argument("--epochs", type=int, default=2200)
    parser.add_argument("--lbfgs-steps", type=int, default=220)
    parser.add_argument("--data-weight", type=float, default=0.3)
    parser.add_argument("--trials", type=int, default=6)
    parser.add_argument("--pinn-weight-seeds", type=int, default=3)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    raw = extract_nordholt_disinfectants(args.workbook)
    selected_conditions = (args.condition,) if args.condition else CONDITIONS
    raw = raw[raw.condition.isin(selected_conditions)].copy()
    raw.to_csv(args.output_dir / "nordholt_replicate_data.csv", index=False)
    positive = raw[~raw.is_zero_count].copy()

    trial_rows: list[dict] = []
    full_rows: list[dict] = []
    local_rows: list[dict] = []
    curve_rows: list[dict] = []
    for condition_index, condition in enumerate(selected_conditions):
        data = positive[positive.condition.eq(condition)].copy()
        unique_times = np.sort(data.time_h.unique())
        final_time = float(raw.loc[raw.condition.eq(condition), "time_h"].max())
        for trial in range(1, args.trials + 1):
            split_seed = 5100 + CONDITIONS.index(condition) * 100 + trial
            split = make_time_split(unique_times, split_seed, test_fraction=0.20)
            train_mask = mask_for_times(data.time_h, split.train_times)
            test_mask = ~train_mask
            t_train = data.time_h.to_numpy()[train_mask]
            y_train = data.log_relative_abundance.to_numpy()[train_mask]
            t_test = data.time_h.to_numpy()[test_mask]
            y_test = data.log_relative_abundance.to_numpy()[test_mask]
            common = {
                "condition": condition,
                "trial": trial,
                "split_seed": split_seed,
                "initial_fraction": INITIAL_FRACTION,
                "train_times_h": ";".join(f"{value:.12g}" for value in split.train_times),
                "test_times_h": ";".join(f"{value:.12g}" for value in split.test_times),
                "n_train_observations": int(train_mask.sum()),
                "n_test_observations": int(test_mask.sum()),
            }

            nlls, _ = fit_global_nlls(t_train, y_train, split_seed)
            nlls_test = exact_log_total(t_test, nlls["D"], nlls["alpha"], nlls["beta"])
            trial_rows.append(
                common
                | {
                    "method": "NLLS",
                    **nlls,
                    "test_rmse_observed_log": rmse(nlls_test, y_test),
                    "ode_implied_test_rmse_log": rmse(nlls_test, y_test),
                    "network_to_fitted_ode_rmse_log": 0.0,
                    "initial_condition_max_abs_error": 0.0,
                    "selected_start_id": np.nan,
                    "all_RN_rmse_h-1": np.nan,
                    "all_RP_rmse_h-1": np.nan,
                }
            )

            nn = fit_best_nn(t_train, y_train, final_time, split_seed, args.epochs)
            nn_test = predict_neural(nn, t_test)
            trial_rows.append(
                common
                | {
                    "method": "NN",
                    "D": np.nan,
                    "alpha": np.nan,
                    "beta": np.nan,
                    "train_sse": nn.final_loss * len(y_train),
                    "global_objective_before_polish": np.nan,
                    "global_nfev": np.nan,
                    "test_rmse_observed_log": rmse(nn_test, y_test),
                    "ode_implied_test_rmse_log": np.nan,
                    "network_to_fitted_ode_rmse_log": np.nan,
                    "initial_condition_max_abs_error": initial_condition_error(nn),
                    "selected_start_id": np.nan,
                    "all_RN_rmse_h-1": np.nan,
                    "all_RP_rmse_h-1": np.nan,
                }
            )

            start_id, weight_seed_index, pinn = fit_best_pinn(
                t_train,
                y_train,
                final_time,
                split_seed,
                args.epochs,
                args.lbfgs_steps,
                args.data_weight,
                args.pinn_weight_seeds,
            )
            parameters = fitted_parameters(pinn)
            pinn_test = predict_neural(pinn, t_test)
            pinn_ode_test = exact_log_total(
                t_test, parameters["D"], parameters["alpha"], parameters["beta"]
            )
            dense = np.linspace(0.0, final_time, 500)
            gap = rmse(
                predict_neural(pinn, dense),
                exact_log_total(dense, parameters["D"], parameters["alpha"], parameters["beta"]),
            )
            trial_rows.append(
                common
                | {
                    "method": "PINN",
                    **parameters,
                    "train_sse": pinn.final_loss,
                    "global_objective_before_polish": np.nan,
                    "global_nfev": np.nan,
                    "test_rmse_observed_log": rmse(pinn_test, y_test),
                    "ode_implied_test_rmse_log": rmse(pinn_ode_test, y_test),
                    "network_to_fitted_ode_rmse_log": gap,
                    "initial_condition_max_abs_error": initial_condition_error(pinn),
                    "selected_start_id": start_id,
                    "selected_weight_seed_index": weight_seed_index,
                    **physics_diagnostics(pinn, final_time),
                }
            )
            pd.DataFrame(trial_rows).to_csv(args.output_dir / "nordholt_cv_trials.csv", index=False)
            print(f"{condition}: fold {trial}/{args.trials}", flush=True)

        times = data.time_h.to_numpy()
        observed = data.log_relative_abundance.to_numpy()
        full_seed = 6100 + CONDITIONS.index(condition)
        nlls, local = fit_global_nlls(times, observed, full_seed)
        for row in local:
            local_rows.append({"condition": condition, "method": "NLLS-local", **row})
        nlls_prediction = exact_log_total(times, nlls["D"], nlls["alpha"], nlls["beta"])
        full_rows.append(
            {
                "condition": condition,
                "method": "NLLS",
                **nlls,
                "network_observed_sse": float(np.sum((nlls_prediction - observed) ** 2)),
                "network_r2": r_squared(observed, nlls_prediction),
                "ode_implied_observed_sse": float(np.sum((nlls_prediction - observed) ** 2)),
                "ode_implied_r2": r_squared(observed, nlls_prediction),
                "network_to_fitted_ode_rmse_log": 0.0,
                "selected_start_id": np.nan,
                "initial_condition_max_abs_error": 0.0,
                "all_RN_rmse_h-1": np.nan,
                "all_RP_rmse_h-1": np.nan,
            }
        )

        nn = fit_best_nn(times, observed, final_time, full_seed, args.epochs)
        nn_prediction = predict_neural(nn, times)
        full_rows.append(
            {
                "condition": condition,
                "method": "NN",
                "D": np.nan,
                "alpha": np.nan,
                "beta": np.nan,
                "train_sse": nn.final_loss * len(observed),
                "global_objective_before_polish": np.nan,
                "global_nfev": np.nan,
                "network_observed_sse": float(np.sum((nn_prediction - observed) ** 2)),
                "network_r2": r_squared(observed, nn_prediction),
                "ode_implied_observed_sse": np.nan,
                "ode_implied_r2": np.nan,
                "network_to_fitted_ode_rmse_log": np.nan,
                "selected_start_id": np.nan,
                "initial_condition_max_abs_error": initial_condition_error(nn),
                "all_RN_rmse_h-1": np.nan,
                "all_RP_rmse_h-1": np.nan,
            }
        )

        start_id, weight_seed_index, pinn = fit_best_pinn(
            times,
            observed,
            final_time,
            full_seed,
            args.epochs,
            args.lbfgs_steps,
            args.data_weight,
            args.pinn_weight_seeds,
        )
        parameters = fitted_parameters(pinn)
        network_prediction = predict_neural(pinn, times)
        ode_prediction = exact_log_total(
            times, parameters["D"], parameters["alpha"], parameters["beta"]
        )
        dense = np.linspace(0.0, final_time, 500)
        full_rows.append(
            {
                "condition": condition,
                "method": "PINN",
                **parameters,
                "train_sse": pinn.final_loss,
                "global_objective_before_polish": np.nan,
                "global_nfev": np.nan,
                "network_observed_sse": float(np.sum((network_prediction - observed) ** 2)),
                "network_r2": r_squared(observed, network_prediction),
                "ode_implied_observed_sse": float(np.sum((ode_prediction - observed) ** 2)),
                "ode_implied_r2": r_squared(observed, ode_prediction),
                "network_to_fitted_ode_rmse_log": rmse(
                    predict_neural(pinn, dense),
                    exact_log_total(dense, parameters["D"], parameters["alpha"], parameters["beta"]),
                ),
                "selected_start_id": start_id,
                "selected_weight_seed_index": weight_seed_index,
                "initial_condition_max_abs_error": initial_condition_error(pinn),
                **physics_diagnostics(pinn, final_time),
            }
        )

        for method, predictor in (
            ("NLLS", lambda x: exact_log_total(x, nlls["D"], nlls["alpha"], nlls["beta"])),
            ("NN", lambda x: predict_neural(nn, x)),
            ("PINN-network", lambda x: predict_neural(pinn, x)),
            ("PINN-implied-ODE", lambda x: exact_log_total(x, parameters["D"], parameters["alpha"], parameters["beta"])),
        ):
            for time_h, value in zip(dense, predictor(dense)):
                curve_rows.append({"condition": condition, "method": method, "time_h": time_h, "log_relative_abundance": value})
        pd.DataFrame(full_rows).to_csv(args.output_dir / "nordholt_full_data_fits.csv", index=False)

    trials = pd.DataFrame(trial_rows)
    trials.to_csv(args.output_dir / "nordholt_cv_trials.csv", index=False)
    metrics = [
        "test_rmse_observed_log",
        "ode_implied_test_rmse_log",
        "D",
        "alpha",
        "beta",
        "network_to_fitted_ode_rmse_log",
    ]
    trials.groupby(["condition", "method"])[metrics].agg(["median", "mean", "std", "count"]).to_csv(
        args.output_dir / "nordholt_cv_summary.csv"
    )
    if set(selected_conditions) == set(CONDITIONS):
        paired_tests(trials).to_csv(args.output_dir / "nordholt_paired_tests.csv", index=False)
    pd.DataFrame(full_rows).to_csv(args.output_dir / "nordholt_full_data_fits.csv", index=False)
    pd.DataFrame(local_rows).to_csv(args.output_dir / "nordholt_nlls_local_diagnostics.csv", index=False)
    curves = pd.DataFrame(curve_rows)
    curves.to_csv(args.output_dir / "nordholt_full_fit_curves.csv", index=False)

    labels = {"nordholt_h2o2": r"H$_2$O$_2$", "nordholt_bac": "BAC", "nordholt_ddac": "DDAC"}
    fig, axes = plt.subplots(1, len(selected_conditions), figsize=(5.2 * len(selected_conditions), 4.5), squeeze=False)
    for axis, condition in zip(axes[0], selected_conditions):
        observed = raw[(raw.condition.eq(condition)) & (~raw.is_zero_count)]
        for replicate, group in observed.groupby("replicate"):
            axis.scatter(group.time_min, group.log_relative_abundance / np.log(10), s=18, alpha=0.55, label="Replicates" if replicate == 1 else None)
        zero_times = raw[(raw.condition.eq(condition)) & (raw.is_zero_count)].time_min
        if len(zero_times):
            floor = observed.log_relative_abundance.min() / np.log(10) - 0.35
            axis.scatter(zero_times, np.full(len(zero_times), floor), marker="v", color="black", s=20, label="Zero count (not fitted)")
        styles = {
            "NLLS": ("#1f4e99", "-"),
            "NN": ("#4b8b3b", "--"),
            "PINN-network": ("#9b1c1c", "-"),
            "PINN-implied-ODE": ("#9b1c1c", ":"),
        }
        for method, (color, style) in styles.items():
            line = curves[(curves.condition.eq(condition)) & (curves.method.eq(method))]
            axis.plot(line.time_h * 60.0, line.log_relative_abundance / np.log(10), style, color=color, lw=2, label=method)
        axis.set_title(labels[condition])
        axis.set_xlabel("Time (min)")
        axis.grid(alpha=0.25)
    axes[0, 0].set_ylabel(r"$\log_{10}[Y(t)/Y(0)]$")
    axes[0, -1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(args.output_dir / "fig_nordholt_repaired_fits.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    protocol = {
        "source_file": args.workbook.name,
        "source_sha256": hashlib.sha256(args.workbook.read_bytes()).hexdigest(),
        "conditions": list(selected_conditions),
        "replicate_definition": "biological replicates, as specified by the source publication",
        "normalization": "each replicate divided by its own time-zero CFU/mL",
        "fit_scale": "natural log relative abundance",
        "zero_count_rule": "retained and flagged in audit table; excluded from log-scale fits; no imputation",
        "split_rule": "held out by time point; all replicate rows at a time stay in one fold; time zero stays in training",
        "initial_persister_fraction": INITIAL_FRACTION,
        "nlls": "differential evolution followed by bounded least-squares polishing",
        "pinn": {
            "same_training_rows_as_other_methods": True,
            "exact_initial_condition": True,
            "rate_multistart_grid": PINN_STARTS,
            "selection": "minimum training objective",
            "network_weight_seeds_per_rate_start": args.pinn_weight_seeds,
            "dimensionless_physics_residual": "physical residual multiplied by final time in hours",
            "data_weight": args.data_weight,
            "epochs": args.epochs,
            "lbfgs_steps": args.lbfgs_steps,
        },
    }
    (args.output_dir / "nordholt_protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")


if __name__ == "__main__":
    main()
