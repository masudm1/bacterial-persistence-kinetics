#!/usr/bin/env python3
"""Fail-fast audit of the frozen PINN confirmation analysis."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration-dir", type=Path, default=Path("calibration_phase2"))
    parser.add_argument("--confirmation-dir", type=Path, default=Path("confirmation_phase2_final"))
    args = parser.parse_args()

    rule_path = args.calibration_dir / "frozen_pinn_rule.json"
    rule = json.loads(rule_path.read_text())
    trials = pd.read_csv(args.confirmation_dir / "confirmation_trials.csv")
    tests = pd.read_csv(args.confirmation_dir / "confirmation_paired_tests.csv")
    manifest = json.loads((args.confirmation_dir / "confirmation_manifest.json").read_text())

    assert rule["confirmation_results_consulted"] is False
    assert set(rule["calibration_seeds"]).isdisjoint(rule["confirmation_seeds_reserved"])
    assert len(trials) == 180
    assert trials.groupby("tier").seed.nunique().eq(20).all()
    assert trials.groupby(["tier", "seed"]).method.nunique().eq(3).all()
    assert trials.groupby(["tier", "trial"]).seed.nunique().eq(1).all()
    assert trials.groupby(["tier", "seed"])[["train_times_h", "test_times_h"]].nunique().to_numpy().max() == 1
    assert trials[["test_rmse_observed_log", "test_rmse_latent_log"]].notna().all().all()

    pinn = trials[trials.method.eq("PINN")]
    assert pinn.initial_condition_max_abs_error.eq(0).all()
    assert pinn.network_to_fitted_ode_rmse_log.notna().all()
    assert len(tests) == 12 and tests.n_pairs.eq(20).all()

    expected_hash = hashlib.sha256(rule_path.read_bytes()).hexdigest()
    assert trials.frozen_rule_sha256.nunique() == 1
    assert trials.frozen_rule_sha256.iloc[0] == expected_hash
    assert manifest["frozen_rule_sha256"] == [expected_hash]

    print("Phase-II verification passed")
    print(f"records={len(trials)}; datasets={trials.groupby(['tier', 'seed']).ngroups}")
    print(f"frozen_rule_sha256={expected_hash}")


if __name__ == "__main__":
    main()
