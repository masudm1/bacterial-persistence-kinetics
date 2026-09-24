#!/usr/bin/env python3
"""Parameter-ratio study under the unchanged frozen Phase-II PINN rule."""

from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from core import SYNTHETIC_INITIAL, fit_nlls_synthetic, make_time_split, mask_for_times, predict_nlls, rmse, solve_total
from neural import fitted_parameters, initial_condition_error, physics_diagnostics, predict_neural
from run_frozen_confirmation import fit_frozen_pinn


CONDITIONS = (
    (0.0001, 0.1),
    (0.001, 0.1),
    (0.01, 0.1),
    (0.05, 0.1),
    (0.1, 0.1),
    (0.1, 0.01),
)


def make_dataset(alpha: float, beta: float, seed: int) -> pd.DataFrame:
    times = np.linspace(0.0, 12.0, 100)
    latent = solve_total(times, SYNTHETIC_INITIAL, alpha, beta)
    rng = np.random.default_rng(seed)
    observed = np.maximum(latent + rng.normal(0.0, 0.10 * latent), 1.0)
    return pd.DataFrame(
        {
            "time_h": times,
            "log_observed": np.log(observed),
            "log_latent": np.log(latent),
        }
    )


def analyze_task(task: dict) -> dict:
    alpha_true = task["alpha_true"]
    beta_true = task["beta_true"]
    seed = task["seed"]
    data = make_dataset(alpha_true, beta_true, seed)
    split = make_time_split(data.time_h, seed + 50_000)
    train = mask_for_times(data.time_h, split.train_times)
    test = ~train
    t_train = data.time_h.to_numpy()[train]
    y_train = data.log_observed.to_numpy()[train]
    t_test = data.time_h.to_numpy()[test]
    y_test = data.log_observed.to_numpy()[test]
    latent_test = data.log_latent.to_numpy()[test]
    common = {
        "alpha_true": alpha_true,
        "beta_true": beta_true,
        "beta_alpha_ratio": beta_true / alpha_true,
        "trial": task["trial"],
        "seed": seed,
        "noise_fraction": 0.10,
        "train_times_h": ";".join(f"{value:.8g}" for value in split.train_times),
        "test_times_h": ";".join(f"{value:.8g}" for value in split.test_times),
        "n_train_observations": int(train.sum()),
        "n_test_observations": int(test.sum()),
    }
    nlls = fit_nlls_synthetic(t_train, y_train)
    nlls_test = predict_nlls(t_test, nlls, SYNTHETIC_INITIAL, synthetic=True)
    rows = [
        common
        | {
            "method": "NLLS",
            "alpha": nlls["alpha"],
            "beta": nlls["beta"],
            "alpha_relative_error_pct": 100.0 * abs(nlls["alpha"] / alpha_true - 1.0),
            "beta_relative_error_pct": 100.0 * abs(nlls["beta"] / beta_true - 1.0),
            "test_rmse_observed_log": rmse(nlls_test, y_test),
            "test_rmse_latent_log": rmse(nlls_test, latent_test),
            "training_objective": nlls["train_sse"],
            "selected_start_id": np.nan,
            "network_to_fitted_ode_rmse_log": 0.0,
            "initial_condition_max_abs_error": 0.0,
        }
    ]
    objective, start_id, _initial_rates, pinn = fit_frozen_pinn(
        task["rule"],
        t_train,
        y_train,
        seed,
        task["epochs"],
        task["lbfgs_steps"],
    )
    fitted = fitted_parameters(pinn)
    pinn_test = predict_neural(pinn, t_test)
    dense = np.linspace(0.0, 12.0, 600)
    ode_dense = np.log(solve_total(dense, SYNTHETIC_INITIAL, fitted["alpha"], fitted["beta"]))
    rows.append(
        common
        | {
            "method": "PINN",
            "alpha": fitted["alpha"],
            "beta": fitted["beta"],
            "alpha_relative_error_pct": 100.0 * abs(fitted["alpha"] / alpha_true - 1.0),
            "beta_relative_error_pct": 100.0 * abs(fitted["beta"] / beta_true - 1.0),
            "test_rmse_observed_log": rmse(pinn_test, y_test),
            "test_rmse_latent_log": rmse(pinn_test, latent_test),
            "training_objective": objective,
            "selected_start_id": start_id,
            "network_to_fitted_ode_rmse_log": rmse(predict_neural(pinn, dense), ode_dense),
            "initial_condition_max_abs_error": initial_condition_error(pinn),
            **physics_diagnostics(pinn, 12.0),
        }
    )
    return {"rows": rows}


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metrics = ["alpha_relative_error_pct", "beta_relative_error_pct", "test_rmse_observed_log", "test_rmse_latent_log", "network_to_fitted_ode_rmse_log"]
    for keys, group in frame.groupby(["alpha_true", "beta_true", "beta_alpha_ratio", "method"]):
        for metric in metrics:
            values = group[metric].dropna().to_numpy()
            rows.append(
                {
                    "alpha_true": keys[0],
                    "beta_true": keys[1],
                    "beta_alpha_ratio": keys[2],
                    "method": keys[3],
                    "metric": metric,
                    "median": float(np.median(values)),
                    "q1": float(np.quantile(values, 0.25)),
                    "q3": float(np.quantile(values, 0.75)),
                    "mean": float(np.mean(values)),
                    "sd": float(np.std(values, ddof=1)),
                    "n": len(values),
                }
            )
    return pd.DataFrame(rows)


def paired_tests(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ratio, group in frame.groupby("beta_alpha_ratio"):
        pivot = group.pivot(index="trial", columns="method")
        for metric in ("alpha_relative_error_pct", "beta_relative_error_pct"):
            difference = pivot[metric]["PINN"] - pivot[metric]["NLLS"]
            statistic, p_value = wilcoxon(difference, alternative="two-sided", method="exact")
            rows.append(
                {
                    "beta_alpha_ratio": ratio,
                    "metric": metric,
                    "W": statistic,
                    "p_two_sided": p_value,
                    "median_pinn_minus_nlls": float(np.median(difference)),
                    "n_pairs": len(difference),
                }
            )
    result = pd.DataFrame(rows)
    order = np.argsort(result.p_two_sided.to_numpy())
    adjusted = np.empty(len(result))
    running = 0.0
    values = result.p_two_sided.to_numpy()
    for rank, index in enumerate(order):
        running = max(running, (len(values) - rank) * values[index])
        adjusted[index] = min(running, 1.0)
    result["p_holm"] = adjusted
    return result


def make_figure(summary: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7))
    colors = {"NLLS": "#2457a6", "PINN": "#a32121"}
    for axis, metric, title in zip(
        axes,
        ("alpha_relative_error_pct", "beta_relative_error_pct"),
        (r"$\alpha$ relative error", r"$\beta$ relative error"),
    ):
        for method in ("NLLS", "PINN"):
            table = summary[(summary.metric.eq(metric)) & (summary.method.eq(method))].sort_values("beta_alpha_ratio")
            median = table["median"].to_numpy()
            axis.errorbar(
                table.beta_alpha_ratio,
                median,
                yerr=[median - table.q1.to_numpy(), table.q3.to_numpy() - median],
                fmt="o-",
                capsize=3,
                color=colors[method],
                label=method,
            )
        axis.axvline(100.0, color="#777777", ls=":", label="Main synthetic condition")
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlabel(r"True $\beta/\alpha$")
        axis.set_ylabel("Relative error (%)")
        axis.set_title(title)
        axis.grid(alpha=0.25)
    axes[1].legend(frameon=False)
    fig.suptitle("Frozen-rule parameter-ratio study: median and interquartile range (5 datasets)")
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen-rule", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frozen_bytes = args.frozen_rule.read_bytes()
    frozen = json.loads(frozen_bytes)
    rule = frozen["rule"]
    tasks = []
    for condition_index, (alpha, beta) in enumerate(CONDITIONS):
        for trial in range(1, args.trials + 1):
            tasks.append(
                {
                    "alpha_true": alpha,
                    "beta_true": beta,
                    "trial": trial,
                    "seed": 120_000 + 100 * condition_index + trial,
                    "rule": rule,
                    "epochs": frozen["epochs"],
                    "lbfgs_steps": frozen["lbfgs_steps"],
                }
            )
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(analyze_task, task): task for task in tasks}
        for completed, future in enumerate(as_completed(futures), start=1):
            task = futures[future]
            rows.extend(future.result()["rows"])
            pd.DataFrame(rows).to_csv(args.output_dir / "phase6_ratio_trials.csv", index=False)
            print(f"ratio {task['beta_true']/task['alpha_true']:g}, trial {task['trial']} complete ({completed}/{len(tasks)})", flush=True)
    frame = pd.DataFrame(rows).sort_values(["beta_alpha_ratio", "trial", "method"])
    frame.to_csv(args.output_dir / "phase6_ratio_trials.csv", index=False)
    summary = summarize(frame)
    summary.to_csv(args.output_dir / "phase6_ratio_summary.csv", index=False)
    paired_tests(frame).to_csv(args.output_dir / "phase6_ratio_paired_tests.csv", index=False)
    make_figure(summary, args.output_dir / "fig_phase6_frozen_ratio_sweep.png")
    protocol = {
        "frozen_rule_sha256": hashlib.sha256(frozen_bytes).hexdigest(),
        "frozen_candidate": frozen["candidate"],
        "noise_model": "deterministic ODE total plus independent Gaussian measurement error with SD 10% of latent total",
        "conditions": CONDITIONS,
        "trials_per_condition": args.trials,
        "selection": "minimum composite training objective; held-out observations never consulted",
        "seeds": [task["seed"] for task in tasks],
    }
    (args.output_dir / "phase6_ratio_protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")


if __name__ == "__main__":
    main()
