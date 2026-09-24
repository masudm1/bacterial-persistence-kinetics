#!/usr/bin/env python3
"""Verify the final real-data, ratio, and identifiability analyses."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    real_dir = ROOT / "phase6_final"
    ratio_dir = ROOT / "phase6_ratio_final"
    ident_dir = ROOT / "phase6_peyrusson_identifiability"

    real_protocol = json.loads((real_dir / "phase6_real_protocol.json").read_text())
    ratio_protocol = json.loads((ratio_dir / "phase6_ratio_protocol.json").read_text())
    ident_protocol = json.loads((ident_dir / "peyrusson_identifiability_protocol.json").read_text())

    umetani_data = ROOT / "outputs" / "real_replicate_log_data.csv"
    peyrusson_data = ROOT / "outputs" / "peyrusson_replicate_log_data.csv"
    frozen_rule = ROOT / "calibration_phase2" / "frozen_pinn_rule.json"
    require(sha256(umetani_data) == real_protocol["umetani_data_sha256"], "Umetani analysis-data checksum mismatch")
    require(sha256(peyrusson_data) == real_protocol["peyrusson_data_sha256"], "Peyrusson analysis-data checksum mismatch")
    require(sha256(peyrusson_data) == ident_protocol["analysis_data_sha256"], "Peyrusson identifiability-data checksum mismatch")
    require(sha256(frozen_rule) == ratio_protocol["frozen_rule_sha256"], "frozen PINN rule checksum mismatch")
    require("held-out observations never consulted" in real_protocol["selection"], "real-data model selection is not training-only")
    require("held-out observations never consulted" in ratio_protocol["selection"], "ratio model selection is not training-only")

    records = pd.read_csv(real_dir / "phase6_real_fit_records.csv")
    full = pd.read_csv(real_dir / "phase6_full_data_fits.csv")
    tests = pd.read_csv(real_dir / "phase6_real_paired_tests.csv")
    candidates = pd.read_csv(real_dir / "phase6_full_pinn_candidate_diagnostics.csv")
    require(len(records) == 147, "unexpected real-data fit-record count")
    require(len(full) == 15, "expected five full-data cases and three estimators")
    require(set(full.method) == {"NLLS", "PINN", "NN"}, "real-data estimator set changed")
    require((full.groupby(["dataset", "condition", "initial_condition_scenario"]).size() == 3).all(), "incomplete full-data comparison")
    require(len(candidates) == 40, "expected eight PINN starts for each full-data case")
    require((records.initial_condition_max_abs_error == 0).all(), "an estimator does not impose the shared initial condition exactly")

    held = records[records.n_test_observations > 0].copy()
    require(len(held) == 132, "unexpected held-out fit-record count")
    grouping = ["dataset", "condition", "initial_condition_scenario", "fold"]
    require((held.groupby(grouping).size() == 3).all(), "a held-out fold does not contain all three estimators")
    for col in ["train_times_h", "test_times_h", "n_train_observations", "n_test_observations"]:
        require((held.groupby(grouping)[col].nunique() == 1).all(), f"estimators do not share {col}")
    fold_counts = held.groupby(["dataset", "condition", "initial_condition_scenario"])["fold"].nunique()
    require(set(fold_counts.loc["Umetani"]) == {10}, "Umetani must use ten time-blocked folds")
    require(int(fold_counts.loc["Peyrusson"].iloc[0]) == 4, "Peyrusson must use four held-out time folds")
    require(len(tests) == 10 and ((tests.p_holm >= tests.p_two_sided) & (tests.p_holm <= 1)).all(), "invalid real-data paired tests")
    require((tests.p_holm >= 0.05).all(), "a corrected real-data estimator comparison unexpectedly became significant")

    pinn_full = full[full.method.eq("PINN")]
    require(np.isfinite(pinn_full[["network_to_fitted_ode_rmse_log", "all_RN_rmse_h-1", "all_RP_rmse_h-1"]]).all().all(), "missing PINN ODE diagnostics")
    require((pinn_full.network_to_fitted_ode_rmse_log < 0.03).all(), "a full-data PINN trajectory is materially off its fitted ODE")
    require(np.isfinite(full[["network_observed_sse", "network_r2"]]).all().all(), "non-finite full-data fit metric")

    ratio = pd.read_csv(ratio_dir / "phase6_ratio_trials.csv")
    ratio_tests = pd.read_csv(ratio_dir / "phase6_ratio_paired_tests.csv")
    require(len(ratio) == 60, "expected six ratios, five datasets, and two estimators")
    require(set(ratio.method) == {"NLLS", "PINN"}, "ratio-study estimator set changed")
    require(ratio.beta_alpha_ratio.nunique() == 6, "ratio-study condition count changed")
    require((ratio.groupby(["beta_alpha_ratio", "trial"]).size() == 2).all(), "incomplete paired ratio trial")
    for col in ["seed", "train_times_h", "test_times_h", "n_train_observations", "n_test_observations"]:
        require((ratio.groupby(["beta_alpha_ratio", "trial"])[col].nunique() == 1).all(), f"ratio estimators do not share {col}")
    require((ratio.initial_condition_max_abs_error == 0).all(), "ratio estimator initial condition mismatch")
    require(len(ratio_tests) == 12 and (ratio_tests.n_pairs == 5).all(), "ratio paired-test contract changed")
    require((ratio_tests.p_holm >= 0.05).all(), "a corrected ratio comparison unexpectedly became significant")

    profiles = pd.read_csv(ident_dir / "peyrusson_profile_likelihood.csv")
    intervals = pd.read_csv(ident_dir / "peyrusson_profile_intervals_diagnostic.csv")
    bootstrap = pd.read_csv(ident_dir / "peyrusson_exact_cluster_bootstrap.csv")
    bootstrap_summary = pd.read_csv(ident_dir / "peyrusson_cluster_bootstrap_summary.csv")
    require(set(profiles.parameter) == {"D_abs", "alpha", "beta"}, "Peyrusson profile parameter set changed")
    require((profiles.groupby("parameter").size() >= 61).all(), "Peyrusson profile grid is incomplete")
    require(len(intervals) == 3 and set(intervals.status) == {"two_sided_on_grid"}, "Peyrusson diagnostic profiles are not two-sided")
    require(intervals[["lower", "upper"]].notna().all().all(), "missing Peyrusson profile bounds")
    require(len(bootstrap) == 10 and np.isclose(bootstrap.probability.sum(), 1), "exact three-cluster bootstrap is incomplete")
    require(len(bootstrap_summary) == 3 and bootstrap_summary.interval_interpretable.astype(bool).all(), "Peyrusson bootstrap summary flags changed")

    for path in [
        real_dir / "fig_phase6_real_full_fits.png",
        ratio_dir / "fig_phase6_frozen_ratio_sweep.png",
        ident_dir / "fig_peyrusson_profile_likelihood.png",
    ]:
        require(path.stat().st_size > 50_000, f"missing or incomplete figure: {path.name}")
    print("Phase-VI verification passed")
    print(f"  real-data fit records: {len(records)}")
    print(f"  full-data PINN candidates: {len(candidates)}")
    print(f"  frozen-rule ratio fits: {len(ratio)}")
    print(f"  exact Peyrusson bootstrap configurations: {len(bootstrap)}")


if __name__ == "__main__":
    main()
