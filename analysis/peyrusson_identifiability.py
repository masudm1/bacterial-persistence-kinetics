#!/usr/bin/env python3
"""Profile likelihood and exact replicate-cluster bootstrap for Peyrusson data."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.linalg import expm
from scipy.optimize import least_squares
from scipy.stats import chi2


INITIAL_FRACTION = 0.001
PARAMETERS = ("D_abs", "alpha", "beta")
LOG_BOUNDS = (
    np.log(np.array([1e-3, 1e-8, 1e-8])),
    np.log(np.array([30.0, 20.0, 20.0])),
)
BROAD_STARTS = (
    (1.0, 1e-5, 0.01),
    (3.0, 1e-3, 0.1),
    (8.0, 0.05, 1.0),
    (20.0, 1.0, 10.0),
)


def exact_log_total(times: np.ndarray, D: float, alpha: float, beta: float) -> np.ndarray:
    initial = np.array([1.0 - INITIAL_FRACTION, INITIAL_FRACTION])
    matrix = np.array([[D - alpha, beta], [alpha, -beta]], dtype=float)
    totals = np.array([(expm(matrix * float(time)) @ initial).sum() for time in np.asarray(times, dtype=float)])
    return np.log(np.maximum(totals, np.finfo(float).tiny))


def residual(encoded: np.ndarray, times: np.ndarray, observed: np.ndarray) -> np.ndarray:
    D_abs, alpha, beta = np.exp(encoded)
    unique, inverse = np.unique(times, return_inverse=True)
    return exact_log_total(unique, -D_abs, alpha, beta)[inverse] - observed


def fit_weighted(times: np.ndarray, observed: np.ndarray, starts: list[np.ndarray], weights: np.ndarray | None = None):
    sqrt_weights = np.sqrt(np.ones(len(observed)) if weights is None else weights)
    best = None
    for start in starts:
        result = least_squares(
            lambda encoded: sqrt_weights * residual(encoded, times, observed),
            np.clip(start, LOG_BOUNDS[0] + 1e-12, LOG_BOUNDS[1] - 1e-12),
            bounds=LOG_BOUNDS,
            max_nfev=10000,
            xtol=1e-11,
            ftol=1e-11,
            gtol=1e-11,
        )
        sse = float(np.sum(result.fun**2))
        if best is None or sse < best[1]:
            best = (result.x.copy(), sse, bool(result.success))
    return best


def profile_point(fixed_index: int, fixed_value: float, times: np.ndarray, observed: np.ndarray, starts: list[np.ndarray]):
    nuisance = [index for index in range(3) if index != fixed_index]
    best = None
    for start in starts:
        def fixed_residual(values: np.ndarray) -> np.ndarray:
            encoded = np.empty(3)
            encoded[fixed_index] = math.log(fixed_value)
            encoded[nuisance] = values
            return residual(encoded, times, observed)

        result = least_squares(
            fixed_residual,
            np.clip(start[nuisance], LOG_BOUNDS[0][nuisance] + 1e-12, LOG_BOUNDS[1][nuisance] - 1e-12),
            bounds=(LOG_BOUNDS[0][nuisance], LOG_BOUNDS[1][nuisance]),
            max_nfev=10000,
            xtol=1e-11,
            ftol=1e-11,
            gtol=1e-11,
        )
        sse = float(np.sum(result.fun**2))
        if best is None or sse < best[1]:
            encoded = np.empty(3)
            encoded[fixed_index] = math.log(fixed_value)
            encoded[nuisance] = result.x
            best = (encoded, sse, bool(result.success))
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
    value = math.lgamma(total + 1) - sum(math.lgamma(count + 1) for count in counts)
    return float(math.exp(value - total * math.log(len(counts))))


def weighted_quantile(values: np.ndarray, weights: np.ndarray, probability: float) -> float:
    order = np.argsort(values)
    cumulative = np.cumsum(weights[order]) / np.sum(weights)
    return float(values[order][np.searchsorted(cumulative, probability, side="left")])


def profile_interval(group: pd.DataFrame, threshold: float) -> dict:
    ordered = group.sort_values("fixed_value")
    values = ordered.fixed_value.to_numpy()
    lr = ordered.conditional_gaussian_lr.to_numpy()
    minimum = int(np.argmin(lr))

    def crossing(first: int, second: int) -> float:
        fraction = (threshold - lr[first]) / (lr[second] - lr[first])
        return float(np.exp(np.log(values[first]) + fraction * (np.log(values[second]) - np.log(values[first]))))

    lower, lower_crossed = float(values[0]), False
    for index in range(minimum - 1, -1, -1):
        if lr[index] > threshold and lr[index + 1] <= threshold:
            lower, lower_crossed = crossing(index, index + 1), True
            break
    upper, upper_crossed = float(values[-1]), False
    for index in range(minimum + 1, len(values)):
        if lr[index] > threshold and lr[index - 1] <= threshold:
            upper, upper_crossed = crossing(index - 1, index), True
            break
    status = "two_sided_on_grid" if lower_crossed and upper_crossed else "lower_not_crossed" if upper_crossed else "upper_not_crossed" if lower_crossed else "neither_side_crossed"
    return {"lower": lower, "upper": upper, "lower_crossed": lower_crossed, "upper_crossed": upper_crossed, "status": status}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--phase6-fits", type=Path, required=True)
    parser.add_argument("--source-workbook", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--profile-points", type=int, default=61)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(args.data)
    fits = pd.read_csv(args.phase6_fits)
    nlls = fits[(fits.dataset.eq("Peyrusson")) & (fits.fold.eq(0)) & (fits.method.eq("NLLS"))].iloc[0]
    times = data.time_h.to_numpy(dtype=float)
    observed = data.log_relative_abundance.to_numpy(dtype=float)
    mle = np.array([-nlls.D, nlls.alpha, nlls.beta], dtype=float)
    mle_encoded = np.log(mle)
    reference_sse = float(nlls.ode_implied_observed_sse)
    broad = [np.log(np.asarray(start)) for start in BROAD_STARTS]
    profile_rows = []
    for fixed_index, parameter in enumerate(PARAMETERS):
        broad_grid = np.geomspace(np.exp(LOG_BOUNDS[0][fixed_index]), np.exp(LOG_BOUNDS[1][fixed_index]), args.profile_points)
        local_grid = np.clip(mle[fixed_index] * np.geomspace(0.2, 5.0, args.profile_points), np.exp(LOG_BOUNDS[0][fixed_index]), np.exp(LOG_BOUNDS[1][fixed_index]))
        grid = np.unique(np.sort(np.concatenate([broad_grid, local_grid, [mle[fixed_index]]])))
        previous = mle_encoded.copy()
        for fixed_value in grid:
            encoded, sse, converged = profile_point(fixed_index, float(fixed_value), times, observed, [mle_encoded, previous, *broad])
            previous = encoded
            ratio = max(sse / reference_sse, 1.0)
            profile_rows.append(
                {
                    "condition": "peyrusson_extracellular_oxacillin",
                    "parameter": parameter,
                    "fixed_value": fixed_value,
                    "profile_sse": sse,
                    "reference_sse": reference_sse,
                    "relative_sse": sse / reference_sse,
                    "conditional_gaussian_lr": len(observed) * math.log(ratio),
                    "converged": converged,
                    "optimized_D": -math.exp(encoded[0]),
                    "optimized_alpha": math.exp(encoded[1]),
                    "optimized_beta": math.exp(encoded[2]),
                }
            )
        print(f"Peyrusson {parameter} profile complete", flush=True)
    profiles = pd.DataFrame(profile_rows)
    profiles.to_csv(args.output_dir / "peyrusson_profile_likelihood.csv", index=False)

    threshold = float(chi2.ppf(0.95, 1))
    intervals = []
    for parameter, group in profiles.groupby("parameter"):
        intervals.append({"condition": "peyrusson_extracellular_oxacillin", "parameter": parameter, "threshold": threshold, "interpretation": "conditional iid-Gaussian diagnostic only", **profile_interval(group, threshold)})
    interval_frame = pd.DataFrame(intervals)
    interval_frame.to_csv(args.output_dir / "peyrusson_profile_intervals_diagnostic.csv", index=False)

    replicates = sorted(data.replicate.unique().astype(int))
    bootstrap_rows = []
    for configuration, counts in enumerate(weak_compositions(len(replicates), len(replicates)), start=1):
        count_map = dict(zip(replicates, counts))
        weights = data.replicate.map(count_map).to_numpy(dtype=float)
        encoded, sse, converged = fit_weighted(times, observed, [mle_encoded, *broad], weights)
        D_abs, alpha, beta = np.exp(encoded)
        bootstrap_rows.append(
            {
                "condition": "peyrusson_extracellular_oxacillin",
                "configuration": configuration,
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
    bootstrap = pd.DataFrame(bootstrap_rows)
    bootstrap.to_csv(args.output_dir / "peyrusson_exact_cluster_bootstrap.csv", index=False)
    weights = bootstrap.probability.to_numpy()
    summaries = []
    for parameter, profile_parameter in (("D", "D_abs"), ("alpha", "alpha"), ("beta", "beta")):
        values = bootstrap[parameter].to_numpy()
        status = interval_frame.loc[interval_frame.parameter.eq(profile_parameter), "status"].iloc[0]
        summaries.append(
            {
                "condition": "peyrusson_extracellular_oxacillin",
                "parameter": parameter,
                "point_estimate": float(nlls[parameter]),
                "bootstrap_median": weighted_quantile(values, weights, 0.5),
                "percentile_2.5": weighted_quantile(values, weights, 0.025),
                "percentile_97.5": weighted_quantile(values, weights, 0.975),
                "profile_status": status,
                "interval_interpretable": status == "two_sided_on_grid",
                "n_exact_configurations": len(bootstrap),
                "total_probability": float(weights.sum()),
                "interpretation_note": "conditional on the fixed 0.1% initial fraction and three empirical biological replicates" if status == "two_sided_on_grid" else f"not interpreted: diagnostic profile status is {status}",
            }
        )
    summary = pd.DataFrame(summaries)
    summary.to_csv(args.output_dir / "peyrusson_cluster_bootstrap_summary.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3))
    colors = {"D_abs": "#2457a6", "alpha": "#3c873a", "beta": "#a32121"}
    for axis, parameter in zip(axes, PARAMETERS):
        group = profiles[profiles.parameter.eq(parameter)]
        axis.semilogx(group.fixed_value, np.minimum(group.conditional_gaussian_lr, 20.0), color=colors[parameter], lw=2)
        axis.axhline(threshold, color="#555555", ls="--", lw=1)
        point = -float(nlls.D) if parameter == "D_abs" else float(nlls[parameter])
        axis.axvline(point, color="black", ls=":", lw=1)
        axis.set_title({"D_abs": r"$-D$", "alpha": r"$\alpha$", "beta": r"$\beta$"}[parameter])
        axis.set_xlabel(r"Rate ($h^{-1}$)")
        axis.set_ylabel("Profile LR")
        axis.set_ylim(0, 20)
        axis.grid(alpha=0.22)
    fig.suptitle("Peyrusson extracellular oxacillin: replicate-level diagnostic profiles")
    fig.tight_layout()
    fig.savefig(args.output_dir / "fig_peyrusson_profile_likelihood.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    protocol = {
        "source_workbook": args.source_workbook.name,
        "source_sha256": hashlib.sha256(args.source_workbook.read_bytes()).hexdigest(),
        "analysis_data_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
        "initial_fraction": INITIAL_FRACTION,
        "parameter_bounds": {"D_abs": [1e-3, 30.0], "alpha": [1e-8, 20.0], "beta": [1e-8, 20.0]},
        "profile_lr": "n*log(SSE/SSE_min); conditional iid-Gaussian diagnostic",
        "cluster_bootstrap": "all unordered biological-replicate count configurations with exact multinomial probabilities",
        "warning": "Only three biological replicates and five time points are available; intervals are coarse and conditional on the fixed initial fraction.",
    }
    (args.output_dir / "peyrusson_identifiability_protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")


if __name__ == "__main__":
    main()
