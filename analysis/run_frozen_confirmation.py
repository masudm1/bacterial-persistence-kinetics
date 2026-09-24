#!/usr/bin/env python3
"""Run the untouched confirmation set using a previously frozen PINN rule."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from core import (
    NOISE_TIERS,
    SYNTHETIC_INITIAL,
    SYNTHETIC_TRUE,
    fit_nlls_synthetic,
    make_time_split,
    mask_for_times,
    predict_nlls,
    rmse,
    solve_total,
    synthetic_dataset,
)
from neural import (
    fitted_parameters,
    fit_neural,
    initial_condition_error,
    physics_diagnostics,
    predict_neural,
)
from run_pipeline import paired_tests


def fit_frozen_pinn(
    rule: dict,
    times: np.ndarray,
    observed: np.ndarray,
    seed: int,
    epochs: int,
    lbfgs_steps: int,
):
    fits = []
    for start_id, initial_rates in enumerate(rule["initial_rate_grid"], start=1):
        result = fit_neural(
            times,
            observed,
            SYNTHETIC_INITIAL,
            12.0,
            seed + 100 * start_id,
            physics=True,
            real=False,
            epochs=epochs,
            lbfgs_steps=lbfgs_steps,
            data_weight=rule["data_weight"],
            architecture=rule["architecture"],
            physics_variant=rule["physics_variant"],
            collocation_strategy=rule["collocation_strategy"],
            initial_rates=tuple(initial_rates),
        )
        fits.append((result.final_loss, start_id, initial_rates, result))
    return min(fits, key=lambda item: item[0])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen-rule", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed-block", type=int, choices=range(4))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frozen_bytes = args.frozen_rule.read_bytes()
    frozen = json.loads(frozen_bytes)
    if frozen.get("confirmation_results_consulted") is not False:
        raise ValueError("Rule was not frozen before confirmation")
    rule = frozen["rule"]
    seeds = frozen["confirmation_seeds_reserved"]
    if seeds != list(range(2001, 2021)):
        raise ValueError("Unexpected confirmation seed set")
    if args.seed_block is not None:
        seeds = seeds[args.seed_block::4]
    frozen_hash = hashlib.sha256(frozen_bytes).hexdigest()
    rows = []
    dense_time = np.linspace(0, 12, 600)

    for tier, noise_fraction in NOISE_TIERS.items():
        for trial, seed in enumerate(seeds, start=1):
            data = synthetic_dataset(seed, noise_fraction)
            split = make_time_split(data.time_h, seed + 50_000)
            train = mask_for_times(data.time_h, split.train_times)
            test = ~train
            times_train = data.time_h.to_numpy()[train]
            observed_train = data.log_observed_total.to_numpy()[train]
            times_test = data.time_h.to_numpy()[test]
            observed_test = data.log_observed_total.to_numpy()[test]
            latent_test = data.log_latent_total.to_numpy()[test]
            common = {
                "tier": tier,
                "noise_fraction": noise_fraction,
                "trial": trial,
                "seed": seed,
                "frozen_rule_sha256": frozen_hash,
                "n_train_observations": int(train.sum()),
                "n_test_observations": int(test.sum()),
                "train_times_h": ";".join(f"{x:.8g}" for x in split.train_times),
                "test_times_h": ";".join(f"{x:.8g}" for x in split.test_times),
            }

            nlls = fit_nlls_synthetic(times_train, observed_train)
            prediction = predict_nlls(times_test, nlls, SYNTHETIC_INITIAL, synthetic=True)
            rows.append(
                common
                | {
                    "method": "NLLS",
                    "alpha": nlls["alpha"],
                    "beta": nlls["beta"],
                    "alpha_relative_error_pct": 100 * abs(nlls["alpha"] / SYNTHETIC_TRUE["alpha"] - 1),
                    "beta_relative_error_pct": 100 * abs(nlls["beta"] / SYNTHETIC_TRUE["beta"] - 1),
                    "test_rmse_observed_log": rmse(prediction, observed_test),
                    "test_rmse_latent_log": rmse(prediction, latent_test),
                    "training_objective": nlls["train_sse"],
                    "selected_start_id": np.nan,
                    "initial_condition_max_abs_error": 0.0,
                    "network_to_fitted_ode_rmse_log": 0.0,
                }
            )

            nn = fit_neural(
                times_train,
                observed_train,
                SYNTHETIC_INITIAL,
                12.0,
                seed,
                physics=False,
                real=False,
                epochs=frozen["epochs"],
                lbfgs_steps=0,
                architecture=rule["architecture"],
            )
            prediction = predict_neural(nn, times_test)
            rows.append(
                common
                | {
                    "method": "NN",
                    "alpha": np.nan,
                    "beta": np.nan,
                    "alpha_relative_error_pct": np.nan,
                    "beta_relative_error_pct": np.nan,
                    "test_rmse_observed_log": rmse(prediction, observed_test),
                    "test_rmse_latent_log": rmse(prediction, latent_test),
                    "training_objective": nn.final_loss,
                    "selected_start_id": np.nan,
                    "initial_condition_max_abs_error": initial_condition_error(nn),
                    "network_to_fitted_ode_rmse_log": np.nan,
                }
            )

            objective, start_id, _initial_rates, pinn = fit_frozen_pinn(
                rule,
                times_train,
                observed_train,
                seed,
                frozen["epochs"],
                frozen["lbfgs_steps"],
            )
            rates = fitted_parameters(pinn)
            prediction = predict_neural(pinn, times_test)
            network_dense = predict_neural(pinn, dense_time)
            ode_dense = np.log(
                solve_total(dense_time, SYNTHETIC_INITIAL, rates["alpha"], rates["beta"])
            )
            rows.append(
                common
                | {
                    "method": "PINN",
                    "alpha": rates["alpha"],
                    "beta": rates["beta"],
                    "alpha_relative_error_pct": 100 * abs(rates["alpha"] / SYNTHETIC_TRUE["alpha"] - 1),
                    "beta_relative_error_pct": 100 * abs(rates["beta"] / SYNTHETIC_TRUE["beta"] - 1),
                    "test_rmse_observed_log": rmse(prediction, observed_test),
                    "test_rmse_latent_log": rmse(prediction, latent_test),
                    "training_objective": objective,
                    "selected_start_id": start_id,
                    "initial_condition_max_abs_error": initial_condition_error(pinn),
                    "network_to_fitted_ode_rmse_log": rmse(network_dense, ode_dense),
                    **physics_diagnostics(pinn, 12.0),
                }
            )
            pd.DataFrame(rows).to_csv(args.output_dir / "confirmation_trials.csv", index=False)
            print(f"confirmation {tier}: {trial}/{len(seeds)}", flush=True)

    frame = pd.DataFrame(rows)
    metrics = [
        "alpha_relative_error_pct",
        "beta_relative_error_pct",
        "test_rmse_observed_log",
        "test_rmse_latent_log",
        "network_to_fitted_ode_rmse_log",
    ]
    summary = frame.groupby(["tier", "method"])[metrics].agg(
        ["median", "mean", "std", "count", lambda x: x.quantile(0.25), lambda x: x.quantile(0.75)]
    )
    summary.to_csv(args.output_dir / "confirmation_summary.csv")
    tests = paired_tests(frame.rename(columns={"seed": "confirmation_seed"}))
    tests.to_csv(args.output_dir / "confirmation_paired_tests.csv", index=False)
    manifest = {
        "frozen_rule_sha256": frozen_hash,
        "frozen_candidate": frozen["candidate"],
        "confirmation_seeds": seeds,
        "trials_per_noise_tier": len(seeds),
        "methods": ["NLLS", "PINN", "NN"],
        "completed_records": len(frame),
    }
    (args.output_dir / "confirmation_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    tier_order = ["low", "moderate", "high"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for method, color in (("NLLS", "#1f4e99"), ("PINN", "#9b1c1c")):
        subset = frame[frame.method.eq(method)]
        med_a = [subset[subset.tier.eq(t)].alpha_relative_error_pct.median() for t in tier_order]
        med_b = [subset[subset.tier.eq(t)].beta_relative_error_pct.median() for t in tier_order]
        axes[0].plot(tier_order, med_a, "o-", label=method, color=color)
        axes[1].plot(tier_order, med_b, "o-", label=method, color=color)
    for method, color in (("NLLS", "#1f4e99"), ("PINN", "#9b1c1c"), ("NN", "#4b8b3b")):
        subset = frame[frame.method.eq(method)]
        med = [subset[subset.tier.eq(t)].test_rmse_observed_log.median() for t in tier_order]
        axes[2].plot(tier_order, med, "o-", label=method, color=color)
    axes[0].set_ylabel(r"Median $\alpha$ relative error (%)")
    axes[1].set_ylabel(r"Median $\beta$ relative error (%)")
    axes[2].set_ylabel("Median held-out RMSE (natural-log units)")
    for ax in axes:
        ax.set_xlabel("Measurement-noise tier")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(args.output_dir / "fig_frozen_confirmation.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
