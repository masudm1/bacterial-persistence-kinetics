#!/usr/bin/env python3
"""Run the corrected matched-estimator benchmarks and write auditable outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from core import (
    NOISE_TIERS,
    SYNTHETIC_INITIAL,
    SYNTHETIC_TRUE,
    extract_real_data,
    fit_nlls_real,
    fit_nlls_synthetic,
    make_time_split,
    mask_for_times,
    predict_nlls,
    rmse,
    synthetic_dataset,
)
from neural import fitted_parameters, fit_neural, initial_condition_error, predict_neural


MEASURED_FRACTIONS = {
    "ciprofloxacin": 32.0 / 197_954.0,
    "ampicillin": 6.0 / 377_167.8211,
}


def split_id(train_times: np.ndarray, test_times: np.ndarray) -> str:
    payload = np.concatenate([train_times, [-999.0], test_times]).tobytes()
    return hashlib.sha256(payload).hexdigest()[:12]


def _score_all(
    method: str,
    prediction_test: np.ndarray,
    noisy_test: np.ndarray,
    latent_test: np.ndarray | None,
) -> dict[str, float | str]:
    result: dict[str, float | str] = {
        "method": method,
        "test_rmse_observed_log": rmse(prediction_test, noisy_test),
    }
    if latent_test is not None:
        result["test_rmse_latent_log"] = rmse(prediction_test, latent_test)
    return result


def run_synthetic(args: argparse.Namespace, out_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for tier, noise_fraction in NOISE_TIERS.items():
        for trial in range(1, args.synthetic_trials + 1):
            data = synthetic_dataset(trial, noise_fraction)
            split = make_time_split(data.time_h, seed=10_000 + trial)
            train = mask_for_times(data.time_h, split.train_times)
            test = ~train
            t_train = data.time_h.to_numpy()[train]
            y_train = data.log_observed_total.to_numpy()[train]
            t_test = data.time_h.to_numpy()[test]
            y_test = data.log_observed_total.to_numpy()[test]
            latent_test = data.log_latent_total.to_numpy()[test]
            common = {
                "tier": tier,
                "noise_fraction": noise_fraction,
                "trial": trial,
                "split_id": split_id(split.train_times, split.test_times),
                "n_train_observations": int(train.sum()),
                "n_test_observations": int(test.sum()),
                "train_times_h": ";".join(f"{x:.8g}" for x in split.train_times),
                "test_times_h": ";".join(f"{x:.8g}" for x in split.test_times),
            }

            nlls = fit_nlls_synthetic(t_train, y_train)
            prediction = predict_nlls(t_test, nlls, SYNTHETIC_INITIAL, synthetic=True)
            rows.append(
                common
                | _score_all("NLLS", prediction, y_test, latent_test)
                | {
                    "alpha": nlls["alpha"],
                    "beta": nlls["beta"],
                    "alpha_relative_error_pct": 100 * abs(nlls["alpha"] / SYNTHETIC_TRUE["alpha"] - 1),
                    "beta_relative_error_pct": 100 * abs(nlls["beta"] / SYNTHETIC_TRUE["beta"] - 1),
                    "training_objective": nlls["train_sse"],
                    "initial_condition_max_abs_error": 0.0,
                }
            )

            nn_result = fit_neural(
                t_train,
                y_train,
                SYNTHETIC_INITIAL,
                12.0,
                trial,
                physics=False,
                real=False,
                epochs=args.epochs,
                lbfgs_steps=0,
            )
            prediction = predict_neural(nn_result, t_test)
            rows.append(
                common
                | _score_all("NN", prediction, y_test, latent_test)
                | {
                    "alpha": np.nan,
                    "beta": np.nan,
                    "alpha_relative_error_pct": np.nan,
                    "beta_relative_error_pct": np.nan,
                    "training_objective": nn_result.final_loss,
                    "initial_condition_max_abs_error": initial_condition_error(nn_result),
                }
            )

            pinn_result = fit_neural(
                t_train,
                y_train,
                SYNTHETIC_INITIAL,
                12.0,
                trial,
                physics=True,
                real=False,
                epochs=args.epochs,
                lbfgs_steps=args.lbfgs_steps,
                data_weight=args.data_weight,
            )
            parameters = fitted_parameters(pinn_result)
            prediction = predict_neural(pinn_result, t_test)
            rows.append(
                common
                | _score_all("PINN", prediction, y_test, latent_test)
                | {
                    "alpha": parameters["alpha"],
                    "beta": parameters["beta"],
                    "alpha_relative_error_pct": 100 * abs(parameters["alpha"] / SYNTHETIC_TRUE["alpha"] - 1),
                    "beta_relative_error_pct": 100 * abs(parameters["beta"] / SYNTHETIC_TRUE["beta"] - 1),
                    "training_objective": pinn_result.final_loss,
                    "initial_condition_max_abs_error": initial_condition_error(pinn_result),
                }
            )
            pd.DataFrame(rows).to_csv(out_dir / "synthetic_trials.csv", index=False)
            print(f"synthetic {tier} trial {trial}/{args.synthetic_trials}", flush=True)
    return pd.DataFrame(rows)


def run_real(args: argparse.Namespace, real_data: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    fraction_scenarios = {"assumed_0.1pct": 0.001, "classification_anchor": None}
    for condition in ("ciprofloxacin", "ampicillin"):
        data = real_data[real_data.condition.eq(condition)].reset_index(drop=True)
        for scenario, scenario_fraction in fraction_scenarios.items():
            initial_fraction = MEASURED_FRACTIONS[condition] if scenario_fraction is None else scenario_fraction
            initial = (1.0 - initial_fraction, initial_fraction)
            for trial in range(1, args.real_splits + 1):
                split = make_time_split(data.time_h, seed=20_000 + trial)
                train = mask_for_times(data.time_h, split.train_times)
                test = ~train
                t_train = data.time_h.to_numpy()[train]
                y_train = data.log_relative_abundance.to_numpy()[train]
                t_test = data.time_h.to_numpy()[test]
                y_test = data.log_relative_abundance.to_numpy()[test]
                common = {
                    "condition": condition,
                    "initial_condition_scenario": scenario,
                    "initial_fraction": initial_fraction,
                    "trial": trial,
                    "split_id": split_id(split.train_times, split.test_times),
                    "n_train_observations": int(train.sum()),
                    "n_test_observations": int(test.sum()),
                    "train_times_h": ";".join(f"{x:.8g}" for x in split.train_times),
                    "test_times_h": ";".join(f"{x:.8g}" for x in split.test_times),
                }
                nlls = fit_nlls_real(t_train, y_train, initial_fraction)
                prediction = predict_nlls(t_test, nlls, initial, synthetic=False)
                rows.append(
                    common
                    | _score_all("NLLS", prediction, y_test, None)
                    | nlls
                    | {"initial_condition_max_abs_error": 0.0}
                )
                nn_result = fit_neural(
                    t_train,
                    y_train,
                    initial,
                    7.0,
                    trial,
                    physics=False,
                    real=True,
                    epochs=args.epochs,
                    lbfgs_steps=0,
                )
                prediction = predict_neural(nn_result, t_test)
                rows.append(
                    common
                    | _score_all("NN", prediction, y_test, None)
                    | {"D": np.nan, "alpha": np.nan, "beta": np.nan, "train_sse": nn_result.final_loss}
                    | {"initial_condition_max_abs_error": initial_condition_error(nn_result)}
                )
                pinn_result = fit_neural(
                    t_train,
                    y_train,
                    initial,
                    7.0,
                    trial,
                    physics=True,
                    real=True,
                    epochs=args.epochs,
                    lbfgs_steps=args.lbfgs_steps,
                    data_weight=args.data_weight,
                )
                parameters = fitted_parameters(pinn_result)
                prediction = predict_neural(pinn_result, t_test)
                rows.append(
                    common
                    | _score_all("PINN", prediction, y_test, None)
                    | parameters
                    | {"train_sse": pinn_result.final_loss}
                    | {"initial_condition_max_abs_error": initial_condition_error(pinn_result)}
                )
                pd.DataFrame(rows).to_csv(out_dir / "real_trials.csv", index=False)
                print(f"real {condition} {scenario} split {trial}/{args.real_splits}", flush=True)
    return pd.DataFrame(rows)


def summarize_metric(frame: pd.DataFrame, group: list[str], metrics: list[str]) -> pd.DataFrame:
    return frame.groupby(group, dropna=False)[metrics].agg(["median", "mean", "std", "count"]).reset_index()


def paired_tests(synthetic: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for tier, group in synthetic.groupby("tier"):
        pivot = group.pivot(index="trial", columns="method")
        for metric, first, second in [
            ("alpha_relative_error_pct", "PINN", "NLLS"),
            ("beta_relative_error_pct", "PINN", "NLLS"),
            ("test_rmse_observed_log", "PINN", "NLLS"),
            ("test_rmse_observed_log", "PINN", "NN"),
        ]:
            x = pivot[metric][first].to_numpy()
            y = pivot[metric][second].to_numpy()
            try:
                stat, p = wilcoxon(x, y, alternative="two-sided")
            except ValueError:
                stat, p = np.nan, np.nan
            rows.append({"tier": tier, "metric": metric, "method_1": first, "method_2": second, "W": stat, "p_two_sided": p, "median_paired_difference_method1_minus_method2": np.median(x - y), "n_pairs": len(x)})
    result = pd.DataFrame(rows)
    valid = result.p_two_sided.notna()
    p_values = result.loc[valid, "p_two_sided"].to_numpy()
    order = np.argsort(p_values)
    adjusted_sorted = np.maximum.accumulate((len(p_values) - np.arange(len(p_values))) * p_values[order])
    adjusted = np.empty_like(adjusted_sorted)
    adjusted[order] = np.minimum(adjusted_sorted, 1.0)
    result.loc[valid, "p_holm"] = adjusted
    return result


def make_figures(synthetic: pd.DataFrame, real: pd.DataFrame, out_dir: Path) -> None:
    colors = {"NLLS": "#1f4e99", "PINN": "#9b1c1c", "NN": "#4b8b3b"}
    tier_order = ["low", "moderate", "high"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for method in ("NLLS", "PINN"):
        grouped = synthetic[synthetic.method.eq(method)].groupby("tier")
        axes[0].plot(tier_order, [grouped.get_group(x).alpha_relative_error_pct.median() for x in tier_order], "o-", label=method, color=colors[method])
        axes[1].plot(tier_order, [grouped.get_group(x).beta_relative_error_pct.median() for x in tier_order], "o-", label=method, color=colors[method])
    axes[0].set_ylabel(r"Median relative error in $\alpha$ (%)")
    axes[1].set_ylabel(r"Median relative error in $\beta$ (%)")
    for ax in axes:
        ax.set_xlabel("Measurement-noise tier")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_parameter_recovery.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    positions = np.arange(3)
    width = 0.23
    for offset, method in enumerate(("NLLS", "PINN", "NN")):
        medians = [synthetic[(synthetic.tier.eq(t)) & (synthetic.method.eq(method))].test_rmse_observed_log.median() for t in tier_order]
        axes[0].bar(positions + (offset - 1) * width, medians, width, label=method, color=colors[method])
    axes[0].set_xticks(positions, tier_order)
    axes[0].set_ylabel("Median held-out RMSE (natural-log units)")
    axes[0].set_title("Synthetic observations")
    axes[0].legend(frameon=False)
    real_groups = list(real.groupby(["condition", "initial_condition_scenario"], sort=False))
    condition_labels = {"ciprofloxacin": "Ciprofloxacin", "ampicillin": "Ampicillin"}
    scenario_labels = {"assumed_0.1pct": "0.1% assumption", "classification_anchor": "classification anchor"}
    labels = [f"{condition_labels[c]}\n{scenario_labels[s]}" for (c, s), _ in real_groups]
    positions_real = np.arange(len(labels))
    for offset, method in enumerate(("NLLS", "PINN", "NN")):
        medians = [g[g.method.eq(method)].test_rmse_observed_log.median() for _, g in real_groups]
        axes[1].bar(positions_real + (offset - 1) * width, medians, width, label=method, color=colors[method])
    axes[1].set_xticks(positions_real, labels, rotation=15, ha="right")
    axes[1].set_ylabel("Median held-out RMSE (natural-log units)")
    axes[1].set_title("Replicate-level real data")
    axes[1].legend(frameon=False)
    for ax in axes:
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_matched_heldout_rmse.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_protocol(out_dir: Path, args: argparse.Namespace, real_data: pd.DataFrame) -> None:
    audit = {
        "synthetic_noise_model": "Y_i = max(X_i + epsilon_i, 1), epsilon_i independently Normal(0, (noise_fraction * X_i)^2); X_i is the deterministic ODE total",
        "synthetic_process_noise": False,
        "split_unit": "unique time point",
        "time_zero_forced_into_training": True,
        "real_objective": "unweighted sum of squared natural-log residuals over observed replicate-level relative abundances",
        "ampicillin_source_columns": "B:G (six biological replicate columns)",
        "missing_value_policy": "retain as missing; no imputation",
        "ampicillin_counts_by_time": real_data[real_data.condition.eq("ampicillin")].groupby("time_h").size().tolist(),
        "ciprofloxacin_counts_by_time": real_data[real_data.condition.eq("ciprofloxacin")].groupby("time_h").size().tolist(),
        "neural_epochs": args.epochs,
        "pinn_lbfgs_steps": args.lbfgs_steps,
        "pinn_data_weight": args.data_weight,
        "measured_fraction_calculations": {
            "ciprofloxacin": "32 / 197954",
            "ampicillin": "6 / 377167.8211",
        },
        "measured_fractions": MEASURED_FRACTIONS,
    }
    (out_dir / "analysis_protocol.json").write_text(json.dumps(audit, indent=2) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dryad-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--synthetic-trials", type=int, default=10)
    parser.add_argument("--real-splits", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=2500)
    parser.add_argument("--lbfgs-steps", type=int, default=250)
    parser.add_argument("--data-weight", type=float, default=0.1)
    parser.add_argument("--skip-synthetic", action="store_true")
    parser.add_argument("--skip-real", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    real_data = extract_real_data(args.dryad_dir)
    real_data.to_csv(args.output_dir / "real_replicate_log_data.csv", index=False)
    write_protocol(args.output_dir, args, real_data)
    synthetic_path = args.output_dir / "synthetic_trials.csv"
    real_path = args.output_dir / "real_trials.csv"
    synthetic = pd.read_csv(synthetic_path) if args.skip_synthetic else run_synthetic(args, args.output_dir)
    real = pd.read_csv(real_path) if args.skip_real else run_real(args, real_data, args.output_dir)
    summarize_metric(
        synthetic,
        ["tier", "method"],
        ["alpha_relative_error_pct", "beta_relative_error_pct", "test_rmse_observed_log", "test_rmse_latent_log"],
    ).to_csv(args.output_dir / "synthetic_summary.csv", index=False)
    summarize_metric(
        real,
        ["condition", "initial_condition_scenario", "method"],
        ["test_rmse_observed_log", "D", "alpha", "beta"],
    ).to_csv(args.output_dir / "real_summary.csv", index=False)
    paired_tests(synthetic).to_csv(args.output_dir / "paired_wilcoxon_tests.csv", index=False)
    make_figures(synthetic, real, args.output_dir)
    print(f"completed; outputs written to {args.output_dir}")


if __name__ == "__main__":
    main()
