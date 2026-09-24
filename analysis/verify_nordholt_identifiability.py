#!/usr/bin/env python3
"""Verify the Nordholt profile-likelihood and exact cluster-bootstrap outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import expm

from nordholt_identifiability import INITIAL_FRACTION, exact_log_total


EXPECTED_CONFIGURATIONS = {
    "nordholt_h2o2": 126,
    "nordholt_bac": 10,
    "nordholt_ddac": 462,
}
EXPECTED_STATUS = {
    ("nordholt_h2o2", "D_abs"): "two_sided_on_grid",
    ("nordholt_h2o2", "alpha"): "neither_side_crossed",
    ("nordholt_h2o2", "beta"): "neither_side_crossed",
    ("nordholt_bac", "D_abs"): "two_sided_on_grid",
    ("nordholt_bac", "alpha"): "lower_not_crossed",
    ("nordholt_bac", "beta"): "two_sided_on_grid",
    ("nordholt_ddac", "D_abs"): "two_sided_on_grid",
    ("nordholt_ddac", "alpha"): "lower_not_crossed",
    ("nordholt_ddac", "beta"): "two_sided_on_grid",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def verify_closed_form() -> float:
    rng = np.random.default_rng(20260906)
    maximum_error = 0.0
    initial = np.array([1.0 - INITIAL_FRACTION, INITIAL_FRACTION])
    for _ in range(500):
        D_abs, alpha, beta = 10 ** rng.uniform(-4, 2.7, size=3)
        D = -D_abs
        times = np.sort(rng.uniform(0.0, 2.0, size=12))
        matrix = np.array([[D - alpha, beta], [alpha, -beta]])
        reference = np.log(
            np.maximum(
                np.array([(expm(matrix * time) @ initial).sum() for time in times]),
                np.finfo(float).tiny,
            )
        )
        maximum_error = max(maximum_error, float(np.max(np.abs(exact_log_total(times, D, alpha, beta) - reference))))
    require(maximum_error < 1e-9, f"closed-form ODE error is too large: {maximum_error}")
    return maximum_error


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workbook", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    output = args.output_dir
    profiles = pd.read_csv(output / "nordholt_profile_likelihood.csv")
    intervals = pd.read_csv(output / "nordholt_profile_intervals_diagnostic.csv")
    bootstrap = pd.read_csv(output / "nordholt_exact_cluster_bootstrap.csv")
    summary = pd.read_csv(output / "nordholt_cluster_bootstrap_summary.csv")
    protocol = json.loads((output / "nordholt_identifiability_protocol.json").read_text())

    require(protocol["source_sha256"] == hashlib.sha256(args.workbook.read_bytes()).hexdigest(), "source workbook digest mismatch")
    require(len(intervals) == 9, "expected nine profile intervals")
    require(len(summary) == 9, "expected nine bootstrap summaries")
    require(len(bootstrap) == sum(EXPECTED_CONFIGURATIONS.values()), "unexpected number of exact bootstrap configurations")
    require(profiles.converged.astype(bool).all(), "at least one profile optimization failed")
    require(bootstrap.converged.astype(bool).all(), "at least one bootstrap fit failed")
    require(np.isfinite(profiles[["fixed_value", "profile_sse", "conditional_gaussian_lr"]]).all().all(), "non-finite profile output")
    require(np.isfinite(bootstrap[["probability", "D", "alpha", "beta", "weighted_sse"]]).all().all(), "non-finite bootstrap output")

    actual_status = intervals.set_index(["condition", "parameter"]).status.to_dict()
    require(actual_status == EXPECTED_STATUS, f"profile status changed: {actual_status}")
    for key, group in profiles.groupby(["condition", "parameter"]):
        require(float(group.conditional_gaussian_lr.min()) < 1e-7, f"profile minimum is not zero for {key}")

    for condition, expected_count in EXPECTED_CONFIGURATIONS.items():
        group = bootstrap[bootstrap.condition.eq(condition)]
        require(len(group) == expected_count, f"wrong configuration count for {condition}")
        require(abs(float(group.probability.sum()) - 1.0) < 1e-10, f"bootstrap probabilities do not sum to one for {condition}")

    expected_interpretable = {
        ("nordholt_bac", "D"),
        ("nordholt_bac", "beta"),
        ("nordholt_ddac", "D"),
        ("nordholt_ddac", "beta"),
    }
    actual_interpretable = set(
        summary.loc[summary.interval_interpretable.astype(bool), ["condition", "parameter"]]
        .itertuples(index=False, name=None)
    )
    require(actual_interpretable == expected_interpretable, f"incorrect interval interpretation flags: {actual_interpretable}")
    require(np.allclose(summary.total_probability, 1.0, atol=1e-10), "summary probabilities do not equal one")

    for filename in ("fig_nordholt_profile_likelihood.png", "fig_nordholt_cluster_bootstrap.png"):
        require((output / filename).stat().st_size > 20_000, f"missing or incomplete figure: {filename}")

    maximum_error = verify_closed_form()
    print("Nordholt identifiability verification passed")
    print(f"  source SHA-256: {protocol['source_sha256']}")
    print(f"  exact bootstrap configurations: {len(bootstrap)}")
    print(f"  interpretable intervals: {sorted(expected_interpretable)}")
    print(f"  closed-form versus matrix-exponential maximum log error: {maximum_error:.3e}")


if __name__ == "__main__":
    main()
