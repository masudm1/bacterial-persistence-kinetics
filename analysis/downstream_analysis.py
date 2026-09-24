#!/usr/bin/env python3
"""Regenerate model-dependent real-data diagnostics after the pipeline repair."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares, minimize_scalar

from core import (
    SYNTHETIC_INITIAL,
    SYNTHETIC_TRUE,
    extract_real_data,
    fit_nlls_real,
    fit_nlls_synthetic,
    predict_nlls,
    solve_total,
    synthetic_dataset,
)
from neural import fitted_parameters, fit_neural, initial_condition_error, predict_neural
from run_pipeline import MEASURED_FRACTIONS


def full_real_fits(data: pd.DataFrame, epochs: int, lbfgs_steps: int, data_weight: float) -> pd.DataFrame:
    rows = []
    for condition in ("ciprofloxacin", "ampicillin"):
        subset = data[data.condition.eq(condition)]
        for scenario, fraction in (
            ("assumed_0.1pct", 0.001),
            ("classification_anchor", MEASURED_FRACTIONS[condition]),
        ):
            initial = (1.0 - fraction, fraction)
            times = subset.time_h.to_numpy()
            observed = subset.log_relative_abundance.to_numpy()
            nlls = fit_nlls_real(times, observed, fraction)
            prediction = predict_nlls(times, nlls, initial, synthetic=False)
            rows.append(
                {
                    "condition": condition,
                    "initial_condition_scenario": scenario,
                    "initial_fraction": fraction,
                    "method": "NLLS",
                    "seed": np.nan,
                    "D": nlls["D"],
                    "alpha": nlls["alpha"],
                    "beta": nlls["beta"],
                    "data_sse_log": float(np.sum((prediction - observed) ** 2)),
                    "training_objective": nlls["train_sse"],
                    "initial_condition_max_abs_error": 0.0,
                }
            )
            pinn_candidates = []
            for seed in (101, 102, 103):
                result = fit_neural(
                    times,
                    observed,
                    initial,
                    float(times.max()),
                    seed,
                    physics=True,
                    real=True,
                    epochs=epochs,
                    lbfgs_steps=lbfgs_steps,
                    data_weight=data_weight,
                )
                parameters = fitted_parameters(result)
                prediction = predict_neural(result, times)
                pinn_candidates.append((result.final_loss, seed, result, parameters, prediction))
            objective, seed, result, parameters, prediction = min(pinn_candidates, key=lambda x: x[0])
            rows.append(
                {
                    "condition": condition,
                    "initial_condition_scenario": scenario,
                    "initial_fraction": fraction,
                    "method": "PINN_best_of_3",
                    "seed": seed,
                    "D": parameters["D"],
                    "alpha": parameters["alpha"],
                    "beta": parameters["beta"],
                    "data_sse_log": float(np.sum((prediction - observed) ** 2)),
                    "training_objective": objective,
                    "initial_condition_max_abs_error": initial_condition_error(result),
                }
            )
            print(f"full fit: {condition}, {scenario}", flush=True)
    return pd.DataFrame(rows)


def ic_sensitivity(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    base_grid = np.logspace(-6, -1, 16)
    for condition in ("ciprofloxacin", "ampicillin"):
        subset = data[data.condition.eq(condition)]
        fractions = np.unique(np.append(base_grid, [0.001, MEASURED_FRACTIONS[condition]]))
        for fraction in np.sort(fractions):
            result = fit_nlls_real(
                subset.time_h.to_numpy(), subset.log_relative_abundance.to_numpy(), float(fraction)
            )
            rows.append({"condition": condition, "initial_fraction": fraction, **result})
    return pd.DataFrame(rows)


def plot_ic(frame: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, condition in zip(axes, ("ciprofloxacin", "ampicillin")):
        subset = frame[frame.condition.eq(condition)]
        ax.plot(100 * subset.initial_fraction, subset.beta, "o-", color="#9b1c1c", label=r"NLLS $\beta$")
        ax.axvline(0.1, color="#777777", linestyle=":", label="0.1% assumption")
        anchor = 100 * MEASURED_FRACTIONS[condition]
        ax.axvline(anchor, color="#1f77b4", linestyle="-", label="classification anchor")
        ax.set_xscale("log")
        ax.set_xlabel("Initial tolerant fraction (%)")
        ax.set_ylabel(r"Recovered $\beta$ ($\mathrm{h}^{-1}$)")
        ax.set_title(condition.capitalize())
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _profile_synthetic(data: pd.DataFrame, which: str, grid: np.ndarray) -> pd.DataFrame:
    times = data.time_h.to_numpy()
    observed = data.log_observed_total.to_numpy()
    rows = []
    for fixed in grid:
        def objective(log_other: float) -> float:
            other = np.exp(log_other)
            alpha, beta = (fixed, other) if which == "alpha" else (other, fixed)
            prediction = np.log(solve_total(times, SYNTHETIC_INITIAL, alpha, beta))
            return float(np.sum((prediction - observed) ** 2))
        fit = minimize_scalar(objective, bounds=(np.log(1e-8), np.log(10.0)), method="bounded")
        rows.append({"condition": "synthetic_moderate_trial_1", "parameter": which, "fixed_value": fixed, "profile_sse": fit.fun})
    return pd.DataFrame(rows)


def _profile_real(data: pd.DataFrame, condition: str, which: str, grid: np.ndarray) -> pd.DataFrame:
    subset = data[data.condition.eq(condition)]
    times = subset.time_h.to_numpy()
    observed = subset.log_relative_abundance.to_numpy()
    fraction = 0.001
    initial = (1.0 - fraction, fraction)
    full = fit_nlls_real(times, observed, fraction)
    rows = []
    starts = [
        np.log([-full["D"], max(full["beta" if which == "alpha" else "alpha"], 1e-7)]),
        np.log([2.0, 1e-3]),
        np.log([5.0, 0.1]),
        np.log([10.0, 1.0]),
    ]
    unique, inverse = np.unique(times, return_inverse=True)
    for fixed in grid:
        def residual(encoded: np.ndarray) -> np.ndarray:
            D = -np.exp(encoded[0])
            other = np.exp(encoded[1])
            alpha, beta = (fixed, other) if which == "alpha" else (other, fixed)
            prediction = np.log(solve_total(unique, initial, alpha, beta, D=D))[inverse]
            return prediction - observed
        best_sse = np.inf
        for start in starts:
            fit = least_squares(
                residual,
                start,
                bounds=(np.log([1e-3, 1e-8]), np.log([30.0, 20.0])),
                max_nfev=2500,
            )
            best_sse = min(best_sse, float(2 * fit.cost))
        rows.append({"condition": condition, "parameter": which, "fixed_value": fixed, "profile_sse": best_sse})
    return pd.DataFrame(rows)


def profile_likelihood(data: pd.DataFrame) -> pd.DataFrame:
    synthetic = synthetic_dataset(1, 0.10)
    synthetic_fit = fit_nlls_synthetic(
        synthetic.time_h.to_numpy(), synthetic.log_observed_total.to_numpy()
    )
    reference_sse = {"synthetic_moderate_trial_1": synthetic_fit["train_sse"]}
    frames = [
        _profile_synthetic(synthetic, "alpha", np.logspace(-5, -1, 25)),
        _profile_synthetic(synthetic, "beta", np.logspace(-3, 0.5, 25)),
    ]
    for condition in ("ciprofloxacin", "ampicillin"):
        frames.append(_profile_real(data, condition, "alpha", np.logspace(-8, 0, 25)))
        frames.append(_profile_real(data, condition, "beta", np.logspace(-2, 1, 25)))
        subset = data[data.condition.eq(condition)]
        reference_sse[condition] = fit_nlls_real(
            subset.time_h.to_numpy(), subset.log_relative_abundance.to_numpy(), 0.001
        )["train_sse"]
    result = pd.concat(frames, ignore_index=True)
    sample_sizes = {
        "synthetic_moderate_trial_1": len(synthetic),
        "ciprofloxacin": int(data.condition.eq("ciprofloxacin").sum()),
        "ampicillin": int(data.condition.eq("ampicillin").sum()),
    }
    result["likelihood_ratio_statistic"] = np.nan
    for (condition, parameter), idx in result.groupby(["condition", "parameter"]).groups.items():
        sse = result.loc[idx, "profile_sse"]
        ratio = np.maximum(sse / reference_sse[condition], 1.0)
        result.loc[idx, "likelihood_ratio_statistic"] = sample_sizes[condition] * np.log(ratio)
    return result


def plot_profiles(frame: pd.DataFrame, output: Path) -> None:
    conditions = ["synthetic_moderate_trial_1", "ciprofloxacin", "ampicillin"]
    labels = ["Synthetic (10% noise)", "Ciprofloxacin", "Ampicillin"]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for column, (condition, label) in enumerate(zip(conditions, labels)):
        for row, parameter in enumerate(("alpha", "beta")):
            subset = frame[(frame.condition.eq(condition)) & (frame.parameter.eq(parameter))]
            axes[row, column].plot(subset.fixed_value, subset.likelihood_ratio_statistic, "o-", markersize=3)
            axes[row, column].axhline(3.841, color="#9b1c1c", linestyle="--", label="95% LR threshold")
            axes[row, column].set_xscale("log")
            axes[row, column].set_xlabel(fr"Fixed $\{parameter}$ ($\mathrm{{h}}^{{-1}}$)")
            axes[row, column].set_ylabel("Likelihood-ratio statistic")
            axes[row, column].set_title(f"{label}: {parameter}")
            axes[row, column].grid(alpha=0.25)
    axes[0, 0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def fim_log_parameters(data: pd.DataFrame, full_fits: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for condition in ("ciprofloxacin", "ampicillin"):
        subset = data[data.condition.eq(condition)]
        best = full_fits[(full_fits.condition.eq(condition)) & (full_fits.initial_condition_scenario.eq("assumed_0.1pct")) & (full_fits.method.eq("NLLS"))].iloc[0]
        theta = np.log([-best.D, best.alpha, best.beta])
        times = subset.time_h.to_numpy()
        observed = subset.log_relative_abundance.to_numpy()
        unique, inverse = np.unique(times, return_inverse=True)
        initial = (0.999, 0.001)
        def prediction(encoded: np.ndarray) -> np.ndarray:
            D, alpha, beta = -np.exp(encoded[0]), np.exp(encoded[1]), np.exp(encoded[2])
            return np.log(solve_total(unique, initial, alpha, beta, D=D))[inverse]
        base = prediction(theta)
        jacobian = np.empty((len(base), 3))
        step = 1e-4
        for j in range(3):
            plus, minus = theta.copy(), theta.copy()
            plus[j] += step
            minus[j] -= step
            jacobian[:, j] = (prediction(plus) - prediction(minus)) / (2 * step)
        sigma2 = float(np.sum((base - observed) ** 2) / (len(observed) - 3))
        covariance = sigma2 * np.linalg.pinv(jacobian.T @ jacobian)
        correlation = covariance / np.sqrt(np.outer(np.diag(covariance), np.diag(covariance)))
        rows.append(
            {
                "condition": condition,
                "parameterization": "log(-D), log(alpha), log(beta)",
                "condition_number": np.linalg.cond(jacobian.T @ jacobian),
                "corr_logalpha_logbeta": correlation[1, 2],
                "sd_logminusD": np.sqrt(covariance[0, 0]),
                "sd_logalpha": np.sqrt(covariance[1, 1]),
                "sd_logbeta": np.sqrt(covariance[2, 2]),
            }
        )
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=2500)
    parser.add_argument("--lbfgs-steps", type=int, default=250)
    parser.add_argument("--data-weight", type=float, default=0.3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(args.real_data)
    fits = full_real_fits(data, args.epochs, args.lbfgs_steps, args.data_weight)
    fits.to_csv(args.output_dir / "real_full_data_fits.csv", index=False)
    sensitivity = ic_sensitivity(data)
    sensitivity.to_csv(args.output_dir / "initial_fraction_sensitivity.csv", index=False)
    plot_ic(sensitivity, args.output_dir / "fig_initial_fraction_sensitivity.png")
    profiles = profile_likelihood(data)
    profiles.to_csv(args.output_dir / "profile_likelihood.csv", index=False)
    plot_profiles(profiles, args.output_dir / "fig_profile_likelihood_lr.png")
    fim_log_parameters(data, fits).to_csv(args.output_dir / "fim_log_parameterization.csv", index=False)


if __name__ == "__main__":
    main()
