#!/usr/bin/env python3
"""Phase-VI robust multistart PINN analysis for Umetani and Peyrusson data."""

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

from core import fit_nlls_real, make_time_split, mask_for_times, predict_nlls, rmse, solve_total
from neural import fitted_parameters, fit_neural, initial_condition_error, physics_diagnostics, predict_neural
from run_pipeline import MEASURED_FRACTIONS


PINN_STARTS = (
    (2.0, 1e-5, 0.05),
    (5.0, 1e-3, 0.5),
    (10.0, 0.05, 2.0),
    (20.0, 0.5, 10.0),
)


def r_squared(observed: np.ndarray, predicted: np.ndarray) -> float:
    denominator = float(np.sum((observed - np.mean(observed)) ** 2))
    return float(1.0 - np.sum((observed - predicted) ** 2) / denominator)


def ode_prediction(times: np.ndarray, fraction: float, parameters: dict[str, float]) -> np.ndarray:
    return np.log(
        solve_total(
            np.asarray(times, dtype=float),
            (1.0 - fraction, fraction),
            parameters["alpha"],
            parameters["beta"],
            D=parameters["D"],
        )
    )


def fit_best_nn(times: np.ndarray, observed: np.ndarray, fraction: float, final_time: float, base_seed: int, epochs: int):
    initial = (1.0 - fraction, fraction)
    candidates = [
        fit_neural(times, observed, initial, final_time, base_seed + index, False, True, epochs=epochs, lbfgs_steps=0)
        for index in range(3)
    ]
    return min(candidates, key=lambda result: result.final_loss)


def fit_best_pinn(
    times: np.ndarray,
    observed: np.ndarray,
    fraction: float,
    final_time: float,
    base_seed: int,
    epochs: int,
    lbfgs_steps: int,
    data_weight: float,
    weight_seeds: int,
):
    initial = (1.0 - fraction, fraction)
    candidates = []
    for start_id, (D_abs, alpha, beta) in enumerate(PINN_STARTS):
        for weight_seed_index in range(weight_seeds):
            seed = base_seed + 100 * start_id + weight_seed_index
            result = fit_neural(
                times,
                observed,
                initial,
                final_time,
                seed,
                True,
                True,
                epochs=epochs,
                lbfgs_steps=lbfgs_steps,
                data_weight=data_weight,
                initial_rates=(alpha, beta),
                initial_D_abs=D_abs,
                residual_time_scale=final_time,
            )
            candidates.append((start_id, weight_seed_index, seed, result))
    return min(candidates, key=lambda item: item[3].final_loss), candidates


def analyze_task(task: dict) -> dict:
    times = np.asarray(task["times"], dtype=float)
    observed = np.asarray(task["observed"], dtype=float)
    train_mask = np.asarray(task["train_mask"], dtype=bool)
    test_mask = ~train_mask
    fraction = float(task["initial_fraction"])
    final_time = float(task["final_time"])
    t_train, y_train = times[train_mask], observed[train_mask]
    t_test, y_test = times[test_mask], observed[test_mask]
    dense = np.linspace(0.0, final_time, 600)
    common = {
        "dataset": task["dataset"],
        "condition": task["condition"],
        "initial_condition_scenario": task["scenario"],
        "initial_fraction": fraction,
        "fold": task["fold"],
        "selection_seed": task["seed"],
        "train_times_h": ";".join(f"{value:.12g}" for value in np.unique(t_train)),
        "test_times_h": ";".join(f"{value:.12g}" for value in np.unique(t_test)),
        "n_train_observations": len(y_train),
        "n_test_observations": len(y_test),
    }
    rows = []

    nlls = fit_nlls_real(t_train, y_train, fraction)
    nlls_train = predict_nlls(t_train, nlls, (1.0 - fraction, fraction), synthetic=False)
    nlls_test = predict_nlls(t_test, nlls, (1.0 - fraction, fraction), synthetic=False) if len(t_test) else np.array([])
    rows.append(
        common
        | {
            "method": "NLLS",
            **nlls,
            "training_objective": nlls["train_sse"],
            "network_train_sse": float(np.sum((nlls_train - y_train) ** 2)),
            "ode_implied_train_sse": float(np.sum((nlls_train - y_train) ** 2)),
            "test_rmse_observed_log": rmse(nlls_test, y_test) if len(y_test) else np.nan,
            "ode_implied_test_rmse_log": rmse(nlls_test, y_test) if len(y_test) else np.nan,
            "network_to_fitted_ode_rmse_log": 0.0,
            "initial_condition_max_abs_error": 0.0,
            "selected_start_id": np.nan,
            "selected_weight_seed_index": np.nan,
            "all_RN_rmse_h-1": np.nan,
            "all_RP_rmse_h-1": np.nan,
        }
    )

    nn = fit_best_nn(t_train, y_train, fraction, final_time, task["seed"] * 10, task["epochs"])
    nn_train = predict_neural(nn, t_train)
    nn_test = predict_neural(nn, t_test) if len(t_test) else np.array([])
    rows.append(
        common
        | {
            "method": "NN",
            "D": np.nan,
            "alpha": np.nan,
            "beta": np.nan,
            "train_sse": float(np.sum((nn_train - y_train) ** 2)),
            "training_objective": nn.final_loss,
            "network_train_sse": float(np.sum((nn_train - y_train) ** 2)),
            "ode_implied_train_sse": np.nan,
            "test_rmse_observed_log": rmse(nn_test, y_test) if len(y_test) else np.nan,
            "ode_implied_test_rmse_log": np.nan,
            "network_to_fitted_ode_rmse_log": np.nan,
            "initial_condition_max_abs_error": initial_condition_error(nn),
            "selected_start_id": np.nan,
            "selected_weight_seed_index": np.nan,
            "all_RN_rmse_h-1": np.nan,
            "all_RP_rmse_h-1": np.nan,
        }
    )

    selected, candidates = fit_best_pinn(
        t_train,
        y_train,
        fraction,
        final_time,
        task["seed"] * 1000,
        task["epochs"],
        task["lbfgs_steps"],
        task["data_weight"],
        task["weight_seeds"],
    )
    start_id, weight_seed_index, selected_seed, pinn = selected
    parameters = fitted_parameters(pinn)
    pinn_train = predict_neural(pinn, t_train)
    pinn_test = predict_neural(pinn, t_test) if len(t_test) else np.array([])
    ode_train = ode_prediction(t_train, fraction, parameters)
    ode_test = ode_prediction(t_test, fraction, parameters) if len(t_test) else np.array([])
    gap = rmse(predict_neural(pinn, dense), ode_prediction(dense, fraction, parameters))
    rows.append(
        common
        | {
            "method": "PINN",
            **parameters,
            "train_sse": float(np.sum((pinn_train - y_train) ** 2)),
            "training_objective": pinn.final_loss,
            "network_train_sse": float(np.sum((pinn_train - y_train) ** 2)),
            "ode_implied_train_sse": float(np.sum((ode_train - y_train) ** 2)),
            "test_rmse_observed_log": rmse(pinn_test, y_test) if len(y_test) else np.nan,
            "ode_implied_test_rmse_log": rmse(ode_test, y_test) if len(y_test) else np.nan,
            "network_to_fitted_ode_rmse_log": gap,
            "initial_condition_max_abs_error": initial_condition_error(pinn),
            "selected_start_id": start_id,
            "selected_weight_seed_index": weight_seed_index,
            "selected_network_seed": selected_seed,
            **physics_diagnostics(pinn, final_time),
        }
    )

    full_diagnostics = []
    if task["fold"] == 0:
        for candidate_start, candidate_weight, candidate_seed, candidate in candidates:
            fitted = fitted_parameters(candidate)
            network = predict_neural(candidate, dense)
            ode = ode_prediction(dense, fraction, fitted)
            full_diagnostics.append(
                common
                | {
                    "start_id": candidate_start,
                    "weight_seed_index": candidate_weight,
                    "network_seed": candidate_seed,
                    "initial_D_abs": PINN_STARTS[candidate_start][0],
                    "initial_alpha": PINN_STARTS[candidate_start][1],
                    "initial_beta": PINN_STARTS[candidate_start][2],
                    **fitted,
                    "training_objective": candidate.final_loss,
                    "network_to_fitted_ode_rmse_log": rmse(network, ode),
                    **physics_diagnostics(candidate, final_time),
                    "selected": candidate is pinn,
                }
            )

    curves = []
    if task["fold"] == 0:
        full_observed = observed
        for row in rows:
            if row["method"] == "NLLS":
                prediction = ode_prediction(dense, fraction, row)
                observed_prediction = ode_prediction(times, fraction, row)
            elif row["method"] == "NN":
                prediction = predict_neural(nn, dense)
                observed_prediction = predict_neural(nn, times)
            else:
                prediction = predict_neural(pinn, dense)
                observed_prediction = predict_neural(pinn, times)
            row["network_observed_sse"] = float(np.sum((observed_prediction - full_observed) ** 2))
            row["network_r2"] = r_squared(full_observed, observed_prediction)
            if row["method"] in {"NLLS", "PINN"}:
                ode_observed = ode_prediction(times, fraction, row)
                row["ode_implied_observed_sse"] = float(np.sum((ode_observed - full_observed) ** 2))
                row["ode_implied_r2"] = r_squared(full_observed, ode_observed)
            else:
                row["ode_implied_observed_sse"] = np.nan
                row["ode_implied_r2"] = np.nan
            for time_h, value in zip(dense, prediction):
                curves.append(common | {"method": row["method"] if row["method"] != "PINN" else "PINN-network", "time_h": time_h, "log_relative_abundance": value})
            if row["method"] == "PINN":
                for time_h, value in zip(dense, ode_prediction(dense, fraction, row)):
                    curves.append(common | {"method": "PINN-implied-ODE", "time_h": time_h, "log_relative_abundance": value})
    return {"rows": rows, "diagnostics": full_diagnostics, "curves": curves}


def holm(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    result = np.empty_like(values, dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(values) - rank) * values[index])
        result[index] = min(running, 1.0)
    return result


def paired_tests(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in frame.groupby(["dataset", "condition", "initial_condition_scenario"]):
        pivot = group.pivot(index="fold", columns="method", values="test_rmse_observed_log")
        for first, second in (("PINN", "NLLS"), ("PINN", "NN")):
            difference = (pivot[first] - pivot[second]).dropna()
            statistic, p_value = wilcoxon(difference, alternative="two-sided", method="auto")
            rows.append(
                {
                    "dataset": keys[0],
                    "condition": keys[1],
                    "initial_condition_scenario": keys[2],
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
    result["p_holm"] = holm(result.p_two_sided.to_numpy())
    return result


def make_tasks(umetani: pd.DataFrame, peyrusson: pd.DataFrame, args: argparse.Namespace) -> list[dict]:
    tasks = []
    if args.datasets in {"all", "umetani"}:
        for condition_index, condition in enumerate(("ciprofloxacin", "ampicillin")):
            subset = umetani[umetani.condition.eq(condition)].reset_index(drop=True)
            for scenario_index, (scenario, fraction) in enumerate(
                (("assumed_0.1pct", 0.001), ("classification_anchor", MEASURED_FRACTIONS[condition]))
            ):
                splits = [(0, np.ones(len(subset), dtype=bool))]
                for fold in range(1, args.umetani_folds + 1):
                    split = make_time_split(subset.time_h, 20_000 + fold)
                    splits.append((fold, mask_for_times(subset.time_h, split.train_times)))
                for fold, train_mask in splits:
                    tasks.append(
                        {
                            "dataset": "Umetani",
                            "condition": condition,
                            "scenario": scenario,
                            "initial_fraction": fraction,
                            "fold": fold,
                            "seed": 70_000 + condition_index * 10_000 + scenario_index * 1_000 + fold,
                            "times": subset.time_h.to_numpy(),
                            "observed": subset.log_relative_abundance.to_numpy(),
                            "train_mask": train_mask,
                            "final_time": float(subset.time_h.max()),
                            "epochs": args.epochs,
                            "lbfgs_steps": args.lbfgs_steps,
                            "data_weight": args.data_weight,
                            "weight_seeds": args.weight_seeds,
                        }
                    )
    if args.datasets in {"all", "peyrusson"}:
        subset = peyrusson.reset_index(drop=True)
        unique_times = np.sort(subset.time_h.unique())
        splits = [(0, np.ones(len(subset), dtype=bool))]
        for fold, held_out in enumerate(unique_times[unique_times > 0], start=1):
            splits.append((fold, subset.time_h.to_numpy() != held_out))
        for fold, train_mask in splits:
            tasks.append(
                {
                    "dataset": "Peyrusson",
                    "condition": "peyrusson_extracellular_oxacillin",
                    "scenario": "assumed_0.1pct",
                    "initial_fraction": 0.001,
                    "fold": fold,
                    "seed": 100_000 + fold,
                    "times": subset.time_h.to_numpy(),
                    "observed": subset.log_relative_abundance.to_numpy(),
                    "train_mask": train_mask,
                    "final_time": float(subset.time_h.max()),
                    "epochs": args.epochs,
                    "lbfgs_steps": args.lbfgs_steps,
                    "data_weight": args.data_weight,
                    "weight_seeds": args.weight_seeds,
                }
            )
    return tasks


def plot_full_fits(data_frames: dict[str, pd.DataFrame], curves: pd.DataFrame, output: Path) -> None:
    panels = [
        ("Umetani", "ciprofloxacin", "assumed_0.1pct", "Ciprofloxacin: 0.1%"),
        ("Umetani", "ciprofloxacin", "classification_anchor", "Ciprofloxacin: anchor"),
        ("Umetani", "ampicillin", "assumed_0.1pct", "Ampicillin: 0.1%"),
        ("Umetani", "ampicillin", "classification_anchor", "Ampicillin: anchor"),
        ("Peyrusson", "peyrusson_extracellular_oxacillin", "assumed_0.1pct", "Oxacillin: 0.1%"),
    ]
    available = [item for item in panels if not curves[(curves.dataset.eq(item[0])) & (curves.condition.eq(item[1])) & (curves.initial_condition_scenario.eq(item[2]))].empty]
    fig, axes = plt.subplots(1, len(available), figsize=(5.0 * len(available), 4.5), squeeze=False)
    styles = {"NLLS": ("#2457a6", "-"), "NN": ("#3c873a", "--"), "PINN-network": ("#a32121", "-"), "PINN-implied-ODE": ("#a32121", ":")}
    for axis, (dataset, condition, scenario, title) in zip(axes[0], available):
        observed = data_frames[dataset]
        observed = observed[observed.condition.eq(condition)] if "condition" in observed else observed
        for replicate, group in observed.groupby("replicate"):
            axis.scatter(group.time_h, group.log_relative_abundance, s=17, alpha=0.45, color="#555555", label="Replicates" if replicate == observed.replicate.min() else None)
        subset = curves[(curves.dataset.eq(dataset)) & (curves.condition.eq(condition)) & (curves.initial_condition_scenario.eq(scenario))]
        for method, (color, linestyle) in styles.items():
            line = subset[subset.method.eq(method)]
            axis.plot(line.time_h, line.log_relative_abundance, color=color, ls=linestyle, lw=2, label=method)
        axis.set_title(title)
        axis.set_xlabel("Time (h)")
        axis.grid(alpha=0.22)
    axes[0, 0].set_ylabel(r"$\log[Y(t)/Y(0)]$")
    axes[0, -1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--umetani-data", type=Path, required=True)
    parser.add_argument("--peyrusson-data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--datasets", choices=("all", "umetani", "peyrusson"), default="all")
    parser.add_argument("--epochs", type=int, default=1800)
    parser.add_argument("--lbfgs-steps", type=int, default=180)
    parser.add_argument("--data-weight", type=float, default=0.3)
    parser.add_argument("--weight-seeds", type=int, default=2)
    parser.add_argument("--umetani-folds", type=int, default=10)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    umetani = pd.read_csv(args.umetani_data)
    peyrusson = pd.read_csv(args.peyrusson_data)
    tasks = make_tasks(umetani, peyrusson, args)
    all_rows, all_diagnostics, all_curves = [], [], []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(analyze_task, task): task for task in tasks}
        for completed, future in enumerate(as_completed(futures), start=1):
            task = futures[future]
            result = future.result()
            all_rows.extend(result["rows"])
            all_diagnostics.extend(result["diagnostics"])
            all_curves.extend(result["curves"])
            pd.DataFrame(all_rows).to_csv(args.output_dir / "phase6_real_fit_records.csv", index=False)
            print(f"{task['dataset']} {task['condition']} {task['scenario']} fold {task['fold']} complete ({completed}/{len(tasks)})", flush=True)

    records = pd.DataFrame(all_rows).sort_values(["dataset", "condition", "initial_condition_scenario", "fold", "method"])
    records.to_csv(args.output_dir / "phase6_real_fit_records.csv", index=False)
    diagnostics = pd.DataFrame(all_diagnostics).sort_values(["dataset", "condition", "initial_condition_scenario", "start_id", "weight_seed_index"])
    diagnostics.to_csv(args.output_dir / "phase6_full_pinn_candidate_diagnostics.csv", index=False)
    curves = pd.DataFrame(all_curves).sort_values(["dataset", "condition", "initial_condition_scenario", "method", "time_h"])
    curves.to_csv(args.output_dir / "phase6_full_fit_curves.csv", index=False)
    full = records[records.fold.eq(0)].copy()
    full.to_csv(args.output_dir / "phase6_full_data_fits.csv", index=False)
    cv = records[records.fold.gt(0)].copy()
    cv.groupby(["dataset", "condition", "initial_condition_scenario", "method"])[
        ["test_rmse_observed_log", "ode_implied_test_rmse_log", "D", "alpha", "beta", "network_to_fitted_ode_rmse_log"]
    ].agg(["median", "mean", "std", "count"]).to_csv(args.output_dir / "phase6_real_cv_summary.csv")
    paired_tests(cv).to_csv(args.output_dir / "phase6_real_paired_tests.csv", index=False)
    plot_full_fits(
        {"Umetani": umetani, "Peyrusson": peyrusson.assign(condition="peyrusson_extracellular_oxacillin")},
        curves,
        args.output_dir / "fig_phase6_real_full_fits.png",
    )
    protocol = {
        "umetani_data_sha256": hashlib.sha256(args.umetani_data.read_bytes()).hexdigest(),
        "peyrusson_data_sha256": hashlib.sha256(args.peyrusson_data.read_bytes()).hexdigest(),
        "selection": "minimum composite training objective; held-out observations never consulted",
        "pinn_rate_starts": PINN_STARTS,
        "network_weight_seeds_per_rate_start": args.weight_seeds,
        "dimensionless_physics_residual": "physical residual multiplied by assay duration in hours",
        "epochs": args.epochs,
        "lbfgs_steps": args.lbfgs_steps,
        "data_weight": args.data_weight,
        "umetani_folds": args.umetani_folds,
        "peyrusson_folds": 4,
        "initial_conditions": "identical within each method comparison and imposed exactly for neural models",
    }
    (args.output_dir / "phase6_real_protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")


if __name__ == "__main__":
    main()
