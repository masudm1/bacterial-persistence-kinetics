#!/usr/bin/env python3
"""Profile likelihood and exact replicate-cluster bootstrap for Nordholt data."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.linalg import expm
from scipy.optimize import least_squares
from scipy.stats import chi2

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
PARAMETERS = ("D_abs", "alpha", "beta")
CONDITION_LABELS = {
    "nordholt_h2o2": r"H$_2$O$_2$",
    "nordholt_bac": "BAC",
    "nordholt_ddac": "DDAC",
}


def exact_log_total(times: np.ndarray, D: float, alpha: float, beta: float) -> np.ndarray:
    """Closed-form total for the constant two-state system.

    The matrix-exponential fallback is used only for the degenerate repeated-
    eigenvalue case. This vectorized expression is algebraically identical but
    substantially faster for profile and bootstrap refitting.
    """
    times = np.asarray(times, dtype=float)
    trace = D - alpha - beta
    discriminant = max(trace * trace + 4.0 * D * beta, 0.0)
    root = math.sqrt(discriminant)
    if root <= 1e-10 * max(1.0, abs(trace)):
        initial = np.array([1.0 - INITIAL_FRACTION, INITIAL_FRACTION])
        matrix = np.array([[D - alpha, beta], [alpha, -beta]], dtype=float)
        totals = np.array([(expm(matrix * float(time)) @ initial).sum() for time in times])
    else:
        slow = 0.5 * (trace + root)
        fast = 0.5 * (trace - root)
        initial_slope = D * (1.0 - INITIAL_FRACTION)
        slow_weight = (initial_slope - fast) / (slow - fast)
        totals = slow_weight * np.exp(slow * times) + (1.0 - slow_weight) * np.exp(fast * times)
    return np.log(np.maximum(totals, np.finfo(float).tiny))


def nlls_residual(encoded: np.ndarray, times: np.ndarray, observed: np.ndarray) -> np.ndarray:
    D_abs, alpha, beta = np.exp(encoded)
    unique, inverse = np.unique(times, return_inverse=True)
    return exact_log_total(unique, -D_abs, alpha, beta)[inverse] - observed


def fit_encoded(
    times: np.ndarray,
    observed: np.ndarray,
    starts: Iterable[np.ndarray],
    row_weights: np.ndarray | None = None,
) -> tuple[np.ndarray, float, bool]:
    if row_weights is None:
        row_weights = np.ones(len(observed))
    sqrt_weights = np.sqrt(np.asarray(row_weights, dtype=float))

    def residual(encoded: np.ndarray) -> np.ndarray:
        return sqrt_weights * nlls_residual(encoded, times, observed)

    best = None
    for start in starts:
        clipped = np.clip(np.asarray(start, dtype=float), LOG_BOUNDS[0] + 1e-12, LOG_BOUNDS[1] - 1e-12)
        result = least_squares(
            residual,
            clipped,
            bounds=LOG_BOUNDS,
            max_nfev=10000,
            xtol=1e-11,
            ftol=1e-11,
            gtol=1e-11,
        )
        sse = float(np.sum(result.fun**2))
        if best is None or sse < best[1]:
            best = (result.x.copy(), sse, bool(result.success))
    assert best is not None
    return best


def profile_point(
    fixed_index: int,
    fixed_log_value: float,
    times: np.ndarray,
    observed: np.ndarray,
    starts: list[np.ndarray],
) -> tuple[np.ndarray, float, bool]:
    nuisance_indices = [index for index in range(3) if index != fixed_index]
    lower = LOG_BOUNDS[0][nuisance_indices]
    upper = LOG_BOUNDS[1][nuisance_indices]

    def residual(nuisance: np.ndarray) -> np.ndarray:
        encoded = np.empty(3)
        encoded[fixed_index] = fixed_log_value
        encoded[nuisance_indices] = nuisance
        return nlls_residual(encoded, times, observed)

    best = None
    for full_start in starts:
        nuisance_start = np.clip(full_start[nuisance_indices], lower + 1e-12, upper - 1e-12)
        result = least_squares(
            residual,
            nuisance_start,
            bounds=(lower, upper),
            max_nfev=10000,
            xtol=1e-11,
            ftol=1e-11,
            gtol=1e-11,
        )
        sse = float(np.sum(result.fun**2))
        if best is None or sse < best[1]:
            encoded = np.empty(3)
            encoded[fixed_index] = fixed_log_value
            encoded[nuisance_indices] = result.x
            best = (encoded, sse, bool(result.success))
    assert best is not None
    return best


def weak_compositions(total: int, parts: int):
    if parts == 1:
        yield (total,)
        return
    for first in range(total + 1):
        for remainder in weak_compositions(total - first, parts - 1):
            yield (first,) + remainder


def multinomial_probability(counts: tuple[int, ...]) -> float:
    total = sum(counts)
    log_probability = math.lgamma(total + 1) - sum(math.lgamma(count + 1) for count in counts)
    log_probability -= total * math.log(len(counts))
    return float(math.exp(log_probability))


def weighted_quantile(values: np.ndarray, weights: np.ndarray, probability: float) -> float:
    order = np.argsort(values)
    sorted_values = values[order]
    sorted_weights = weights[order]
    cumulative = np.cumsum(sorted_weights) / np.sum(sorted_weights)
    return float(sorted_values[np.searchsorted(cumulative, probability, side="left")])


def interval_from_profile(group: pd.DataFrame, threshold: float) -> dict[str, float | str | bool]:
    ordered = group.sort_values("fixed_value")
    values = ordered.fixed_value.to_numpy()
    lr = ordered.conditional_gaussian_lr.to_numpy()
    minimum_index = int(np.argmin(lr))

    def interpolate(first: int, second: int) -> float:
        y1, y2 = lr[first], lr[second]
        if np.isclose(y1, y2):
            return float(np.sqrt(values[first] * values[second]))
        fraction = (threshold - y1) / (y2 - y1)
        return float(np.exp(np.log(values[first]) + fraction * (np.log(values[second]) - np.log(values[first]))))

    lower_crossed = False
    lower = float(values[0])
    for index in range(minimum_index - 1, -1, -1):
        if lr[index] > threshold and lr[index + 1] <= threshold:
            lower = interpolate(index, index + 1)
            lower_crossed = True
            break

    upper_crossed = False
    upper = float(values[-1])
    for index in range(minimum_index + 1, len(values)):
        if lr[index] > threshold and lr[index - 1] <= threshold:
            upper = interpolate(index - 1, index)
            upper_crossed = True
            break
    if lower_crossed and upper_crossed:
        status = "two_sided_on_grid"
    elif lower_crossed:
        status = "upper_not_crossed"
    elif upper_crossed:
        status = "lower_not_crossed"
    else:
        status = "neither_side_crossed"
    return {"lower": lower, "upper": upper, "lower_crossed": lower_crossed, "upper_crossed": upper_crossed, "status": status}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--full-fits", type=Path, required=True)
    parser.add_argument("--source-workbook", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--profile-points", type=int, default=61)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(args.data)
    data = raw[~raw.is_zero_count].copy()
    full = pd.read_csv(args.full_fits)
    nlls_full = full[full.method.eq("NLLS")].set_index("condition")
    broad_starts = [np.log(np.asarray(start, dtype=float)) for start in PINN_STARTS]

    profile_rows = []
    bootstrap_rows = []
    for condition in CONDITIONS:
        subset = data[data.condition.eq(condition)].copy()
        times = subset.time_h.to_numpy()
        observed = subset.log_relative_abundance.to_numpy()
        mle = np.array(
            [
                -float(nlls_full.loc[condition, "D"]),
                float(nlls_full.loc[condition, "alpha"]),
                float(nlls_full.loc[condition, "beta"]),
            ]
        )
        mle_encoded = np.log(np.maximum(mle, np.exp(LOG_BOUNDS[0])))
        reference_sse = float(nlls_full.loc[condition, "train_sse"])
        n_observations = len(observed)

        for fixed_index, parameter in enumerate(PARAMETERS):
            broad_grid = np.geomspace(
                np.exp(LOG_BOUNDS[0][fixed_index]),
                np.exp(LOG_BOUNDS[1][fixed_index]),
                args.profile_points,
            )
            local_grid = mle[fixed_index] * np.geomspace(0.2, 5.0, args.profile_points)
            local_grid = np.clip(
                local_grid,
                np.exp(LOG_BOUNDS[0][fixed_index]),
                np.exp(LOG_BOUNDS[1][fixed_index]),
            )
            grid = np.unique(np.sort(np.concatenate([broad_grid, local_grid, [mle[fixed_index]]])))
            previous = mle_encoded.copy()
            for fixed_value in grid:
                starts = [mle_encoded, previous, *broad_starts]
                encoded, sse, converged = profile_point(
                    fixed_index, math.log(fixed_value), times, observed, starts
                )
                previous = encoded
                ratio = max(sse / reference_sse, 1.0)
                profile_rows.append(
                    {
                        "condition": condition,
                        "parameter": parameter,
                        "fixed_value": fixed_value,
                        "profile_sse": sse,
                        "reference_sse": reference_sse,
                        "relative_sse": sse / reference_sse,
                        "delta_sse": sse - reference_sse,
                        "conditional_gaussian_lr": n_observations * math.log(ratio),
                        "n_observations": n_observations,
                        "converged": converged,
                        "optimized_D": -math.exp(encoded[0]),
                        "optimized_alpha": math.exp(encoded[1]),
                        "optimized_beta": math.exp(encoded[2]),
                    }
                )
            pd.DataFrame(profile_rows).to_csv(args.output_dir / "nordholt_profile_likelihood.csv", index=False)
            print(f"{condition}: {parameter} profile complete", flush=True)

        replicates = sorted(subset.replicate.unique().astype(int))
        start_vectors = [mle_encoded, *broad_starts]
        for bootstrap_id, counts in enumerate(weak_compositions(len(replicates), len(replicates)), start=1):
            count_map = dict(zip(replicates, counts))
            weights = subset.replicate.map(count_map).to_numpy(dtype=float)
            encoded, sse, converged = fit_encoded(times, observed, start_vectors, weights)
            D_abs, alpha, beta = np.exp(encoded)
            bootstrap_rows.append(
                {
                    "condition": condition,
                    "bootstrap_configuration": bootstrap_id,
                    "cluster_counts": ";".join(str(value) for value in counts),
                    "probability": multinomial_probability(counts),
                    "D": -D_abs,
                    "alpha": alpha,
                    "beta": beta,
                    "weighted_sse": sse,
                    "converged": converged,
                    "alpha_at_lower_bound": alpha <= np.exp(LOG_BOUNDS[0][1]) * 1.01,
                    "beta_at_lower_bound": beta <= np.exp(LOG_BOUNDS[0][2]) * 1.01,
                    "any_upper_bound": bool(np.any(np.exp(encoded) >= np.exp(LOG_BOUNDS[1]) / 1.01)),
                }
            )
            if bootstrap_id % 50 == 0:
                pd.DataFrame(bootstrap_rows).to_csv(args.output_dir / "nordholt_exact_cluster_bootstrap.csv", index=False)
        pd.DataFrame(bootstrap_rows).to_csv(args.output_dir / "nordholt_exact_cluster_bootstrap.csv", index=False)
        print(f"{condition}: exact cluster bootstrap complete ({bootstrap_id} configurations)", flush=True)

    profiles = pd.DataFrame(profile_rows)
    bootstrap = pd.DataFrame(bootstrap_rows)
    profiles.to_csv(args.output_dir / "nordholt_profile_likelihood.csv", index=False)
    bootstrap.to_csv(args.output_dir / "nordholt_exact_cluster_bootstrap.csv", index=False)

    threshold = float(chi2.ppf(0.95, 1))
    interval_rows = []
    for (condition, parameter), group in profiles.groupby(["condition", "parameter"]):
        interval_rows.append(
            {
                "condition": condition,
                "parameter": parameter,
                "threshold": threshold,
                "interpretation": "conditional iid-Gaussian diagnostic only",
                **interval_from_profile(group, threshold),
            }
        )
    profile_intervals = pd.DataFrame(interval_rows)
    profile_intervals.to_csv(args.output_dir / "nordholt_profile_intervals_diagnostic.csv", index=False)

    summary_rows = []
    for condition, group in bootstrap.groupby("condition"):
        weights = group.probability.to_numpy()
        for parameter in ("D", "alpha", "beta"):
            values = group[parameter].to_numpy()
            profile_parameter = "D_abs" if parameter == "D" else parameter
            profile_status = str(
                profile_intervals.loc[
                    profile_intervals.condition.eq(condition)
                    & profile_intervals.parameter.eq(profile_parameter),
                    "status",
                ].iloc[0]
            )
            profile_two_sided = profile_status == "two_sided_on_grid"
            model_adequate_for_inference = condition != "nordholt_h2o2"
            interval_interpretable = profile_two_sided and model_adequate_for_inference
            if not model_adequate_for_inference:
                interpretation_note = "not interpreted: early non-monotone data are inconsistent with this constant-rate ODE"
            elif not profile_two_sided:
                interpretation_note = f"not interpreted: diagnostic profile status is {profile_status}"
            else:
                interpretation_note = "interpretable conditionally on the fixed initial persister fraction"
            summary_rows.append(
                {
                    "condition": condition,
                    "parameter": parameter,
                    "point_estimate": float(nlls_full.loc[condition, parameter]),
                    "bootstrap_median": weighted_quantile(values, weights, 0.5),
                    "percentile_2.5": weighted_quantile(values, weights, 0.025),
                    "percentile_97.5": weighted_quantile(values, weights, 0.975),
                    "probability_at_lower_bound": float(
                        np.sum(weights[group[f"{parameter}_at_lower_bound"]])
                    ) if parameter in {"alpha", "beta"} else np.nan,
                    "probability_any_upper_bound": float(np.sum(weights[group.any_upper_bound])),
                    "n_exact_configurations": len(group),
                    "total_probability": float(weights.sum()),
                    "profile_status": profile_status,
                    "profile_two_sided": profile_two_sided,
                    "model_adequate_for_inference": model_adequate_for_inference,
                    "interval_interpretable": interval_interpretable,
                    "interpretation_note": interpretation_note,
                }
            )
    bootstrap_summary = pd.DataFrame(summary_rows)
    bootstrap_summary.to_csv(args.output_dir / "nordholt_cluster_bootstrap_summary.csv", index=False)

    fig, axes = plt.subplots(3, 3, figsize=(14, 11), sharey=True)
    colors = {"D_abs": "#1f4e99", "alpha": "#4b8b3b", "beta": "#9b1c1c"}
    for row, condition in enumerate(CONDITIONS):
        for column, parameter in enumerate(PARAMETERS):
            axis = axes[row, column]
            group = profiles[(profiles.condition.eq(condition)) & (profiles.parameter.eq(parameter))]
            axis.semilogx(group.fixed_value, np.minimum(group.conditional_gaussian_lr, 20.0), color=colors[parameter], lw=2)
            axis.axhline(threshold, color="#555555", ls="--", lw=1)
            point = -float(nlls_full.loc[condition, "D"]) if parameter == "D_abs" else float(nlls_full.loc[condition, parameter])
            axis.axvline(point, color="black", ls=":", lw=1)
            if row == 0:
                axis.set_title({"D_abs": r"$-D$", "alpha": r"$\alpha$", "beta": r"$\beta$"}[parameter])
            if column == 0:
                axis.set_ylabel(f"{CONDITION_LABELS[condition]}\nprofile LR")
            if row == 2:
                axis.set_xlabel(r"Rate ($h^{-1}$)")
            axis.set_ylim(0, 20)
            axis.grid(alpha=0.2)
    fig.suptitle("Nordholt replicate-level ODE profile likelihoods", y=1.01)
    fig.tight_layout()
    fig.savefig(args.output_dir / "fig_nordholt_profile_likelihood.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for axis, parameter in zip(axes, ("D", "alpha", "beta")):
        table = bootstrap_summary[bootstrap_summary.parameter.eq(parameter)].set_index("condition")
        for y, condition in enumerate(CONDITIONS):
            point = float(table.loc[condition, "point_estimate"])
            lower = float(table.loc[condition, "percentile_2.5"])
            upper = float(table.loc[condition, "percentile_97.5"])
            plot_point, plot_lower, plot_upper = (-point, -upper, -lower) if parameter == "D" else (point, lower, upper)
            if bool(table.loc[condition, "interval_interpretable"]):
                axis.errorbar(
                    plot_point,
                    y,
                    xerr=[[plot_point - plot_lower], [plot_upper - plot_point]],
                    fmt="o",
                    color="#1f4e99",
                    capsize=4,
                )
            else:
                axis.plot(plot_point, y, marker="x", ms=8, mew=2, color="#777777")
        axis.set_yticks(range(len(CONDITIONS)), [CONDITION_LABELS[value] for value in CONDITIONS])
        axis.set_xscale("log")
        axis.set_xlabel({"D": r"$-D$ ($h^{-1}$)", "alpha": r"$\alpha$ ($h^{-1}$)", "beta": r"$\beta$ ($h^{-1}$)"}[parameter])
        axis.grid(alpha=0.2)
    fig.suptitle(
        "Exact biological-replicate cluster-bootstrap 95% percentile intervals\n"
        "Gray crosses: interval not interpreted because the profile is unbounded or the ODE is inadequate"
    )
    fig.tight_layout()
    fig.savefig(args.output_dir / "fig_nordholt_cluster_bootstrap.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    protocol = {
        "source_workbook": args.source_workbook.name,
        "source_sha256": hashlib.sha256(args.source_workbook.read_bytes()).hexdigest(),
        "analysis_data": args.data.name,
        "initial_persister_fraction": 0.001,
        "profile": {
            "parameterization": "natural logarithm of -D, alpha, and beta",
            "bounds_h-1": {"D_abs": [1e-3, 1000.0], "alpha": [1e-8, 1000.0], "beta": [1e-8, 1000.0]},
            "broad_points_per_parameter": args.profile_points,
            "local_points_per_parameter": args.profile_points,
            "local_grid_relative_to_mle": [0.2, 5.0],
            "threshold_crossing": "linear interpolation of LR against log(parameter)",
            "nuisance_optimization": "bounded least squares from the MLE and four broad starts",
            "diagnostic_lr": "n*log(SSE/SSE_min), conditional on iid homoscedastic Gaussian log residuals",
            "diagnostic_threshold": threshold,
            "warning": "LR intervals are diagnostic because repeated observations within biological replicates are correlated; H2O2 is also model-misspecified",
        },
        "cluster_bootstrap": {
            "unit": "biological replicate",
            "method": "all unordered cluster-count configurations enumerated exactly with multinomial probabilities",
            "configuration_counts": {"nordholt_h2o2": 126, "nordholt_bac": 10, "nordholt_ddac": 462},
            "interval": "weighted 2.5th and 97.5th percentiles",
            "monte_carlo_error": "none",
            "conditional_on_initial_fraction": True,
            "interpretation_rule": "show an interval only when its profile is two-sided and the constant-rate ODE is adequate for inference",
        },
    }
    (args.output_dir / "nordholt_identifiability_protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")


if __name__ == "__main__":
    main()
