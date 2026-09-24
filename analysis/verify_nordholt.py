#!/usr/bin/env python3
"""Fail-fast audit of the robust Nordholt comparison outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workbook", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    raw = pd.read_csv(args.output_dir / "nordholt_replicate_data.csv")
    trials = pd.read_csv(args.output_dir / "nordholt_cv_trials.csv")
    full = pd.read_csv(args.output_dir / "nordholt_full_data_fits.csv")
    local = pd.read_csv(args.output_dir / "nordholt_nlls_local_diagnostics.csv")
    tests = pd.read_csv(args.output_dir / "nordholt_paired_tests.csv")
    protocol = json.loads((args.output_dir / "nordholt_protocol.json").read_text())

    expected_rows = {"nordholt_h2o2": 80, "nordholt_bac": 24, "nordholt_ddac": 96}
    expected_zeros = {"nordholt_h2o2": 45, "nordholt_bac": 0, "nordholt_ddac": 8}
    assert raw.groupby("condition").size().to_dict() == expected_rows
    assert raw.groupby("condition").is_zero_count.sum().astype(int).to_dict() == expected_zeros
    assert raw.loc[raw.is_zero_count, "log_relative_abundance"].isna().all()
    assert raw.loc[~raw.is_zero_count, "log_relative_abundance"].notna().all()

    assert len(trials) == 54
    assert trials.groupby(["condition", "trial"]).method.nunique().eq(3).all()
    assert trials.groupby(["condition", "trial"])[["train_times_h", "test_times_h"]].nunique().to_numpy().max() == 1
    assert trials.groupby(["condition", "method"]).trial.nunique().eq(6).all()
    assert trials.train_times_h.str.split(";").map(lambda values: any(np.isclose(float(x), 0.0) for x in values)).all()
    assert trials.loc[trials.method.isin(["NN", "PINN"]), "initial_condition_max_abs_error"].eq(0).all()
    assert trials.loc[trials.method.eq("PINN"), "network_to_fitted_ode_rmse_log"].notna().all()

    assert len(full) == 9
    assert full.groupby("condition").method.nunique().eq(3).all()
    assert len(local) == 12 and local.converged.all()
    global_sse = full[full.method.eq("NLLS")].set_index("condition").train_sse
    for condition, group in local.groupby("condition"):
        assert np.allclose(group.sse, global_sse[condition], rtol=1e-4, atol=1e-5)
    assert len(tests) == 6 and tests.n_pairs.eq(6).all()

    source_hash = hashlib.sha256(args.workbook.read_bytes()).hexdigest()
    assert protocol["source_sha256"] == source_hash
    assert protocol["pinn"]["network_weight_seeds_per_rate_start"] == 3
    assert protocol["zero_count_rule"].endswith("no imputation")
    print("Nordholt verification passed")
    print(f"source_sha256={source_hash}")
    print(f"raw_rows={len(raw)}; positive_fit_rows={raw.log_relative_abundance.notna().sum()}")
    print(f"validation_records={len(trials)}; full_fit_records={len(full)}")


if __name__ == "__main__":
    main()
