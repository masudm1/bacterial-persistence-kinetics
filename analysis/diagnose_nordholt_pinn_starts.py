#!/usr/bin/env python3
"""Full-data PINN start and weight-seed diagnostics for Nordholt conditions."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from core import extract_nordholt_disinfectants, rmse
from neural import fitted_parameters, fit_neural, physics_diagnostics, predict_neural
from run_nordholt import CONDITIONS, INITIAL_FRACTION, PINN_STARTS, exact_log_total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workbook", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=2500)
    parser.add_argument("--lbfgs-steps", type=int, default=250)
    parser.add_argument("--data-weight", type=float, default=0.3)
    parser.add_argument("--weight-seeds", type=int, default=3)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    raw = extract_nordholt_disinfectants(args.workbook)
    positive = raw[~raw.is_zero_count]
    initial = (1.0 - INITIAL_FRACTION, INITIAL_FRACTION)
    rows = []
    for condition_index, condition in enumerate(CONDITIONS):
        data = positive[positive.condition.eq(condition)]
        times = data.time_h.to_numpy()
        observed = data.log_relative_abundance.to_numpy()
        final_time = float(raw.loc[raw.condition.eq(condition), "time_h"].max())
        dense = np.linspace(0.0, final_time, 500)
        for start_id, (D_abs, alpha0, beta0) in enumerate(PINN_STARTS):
            for weight_seed_index in range(args.weight_seeds):
                seed = 7100 + condition_index * 100 + start_id * 10 + weight_seed_index
                result = fit_neural(
                    times,
                    observed,
                    initial,
                    final_time,
                    seed,
                    True,
                    True,
                    epochs=args.epochs,
                    lbfgs_steps=args.lbfgs_steps,
                    data_weight=args.data_weight,
                    initial_rates=(alpha0, beta0),
                    initial_D_abs=D_abs,
                    residual_time_scale=final_time,
                )
                parameters = fitted_parameters(result)
                network = predict_neural(result, times)
                implied_ode = exact_log_total(
                    times, parameters["D"], parameters["alpha"], parameters["beta"]
                )
                diagnostics = physics_diagnostics(result, final_time)
                dimensionless_physics_mse = final_time**2 * (
                    diagnostics["all_RN_rmse_h-1"] ** 2
                    + diagnostics["all_RP_rmse_h-1"] ** 2
                )
                data_mse = float(np.mean((network - observed) ** 2))
                rows.append(
                    {
                        "condition": condition,
                        "start_id": start_id,
                        "weight_seed_index": weight_seed_index,
                        "seed": seed,
                        "initial_D_abs": D_abs,
                        "initial_alpha": alpha0,
                        "initial_beta": beta0,
                        **parameters,
                        "reported_training_objective": result.final_loss,
                        "recomputed_data_mse": data_mse,
                        "recomputed_dimensionless_physics_mse": dimensionless_physics_mse,
                        "recomputed_composite_objective": args.data_weight * data_mse
                        + dimensionless_physics_mse,
                        "network_observed_sse": float(np.sum((network - observed) ** 2)),
                        "ode_implied_observed_sse": float(np.sum((implied_ode - observed) ** 2)),
                        "network_to_fitted_ode_rmse_log": rmse(
                            predict_neural(result, dense),
                            exact_log_total(
                                dense,
                                parameters["D"],
                                parameters["alpha"],
                                parameters["beta"],
                            ),
                        ),
                        **diagnostics,
                    }
                )
                pd.DataFrame(rows).to_csv(args.output, index=False)
                print(
                    f"{condition}: rate start {start_id + 1}/{len(PINN_STARTS)}, "
                    f"weight seed {weight_seed_index + 1}/{args.weight_seeds}",
                    flush=True,
                )


if __name__ == "__main__":
    main()
