#!/usr/bin/env python3
"""Robustness analysis for non-exponential resuscitation.

The exponential two-compartment model is k=1. For k>1, persisters pass
through k serial stages at rate k*beta. A cell entering stage P1 has mean
entry-to-exit residence time 1/beta. This does not imply the same mean
remaining time for an initial population distributed across stages.
Two initial-stage allocations are evaluated because aggregate measurements do
not identify the persister-age distribution at treatment onset.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.linalg import expm
from scipy.optimize import least_squares
from scipy.special import gammaln
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "figure_code" / "figure_data" / "outputs"
SEED = 20260906
BOUNDS = (np.log([1e-4, 1e-8, 1e-4]), np.log([100.0, 10.0, 100.0]))
ANCHORS = {"ciprofloxacin": 32 / 197954, "ampicillin": 6 / 377167.8211}
K_POPULATION = range(1, 5)
K_LAG = range(1, 7)


def initial_state(k, fraction, allocation):
    state = np.zeros(k + 1)
    state[0] = 1.0 - fraction
    if allocation == "stage1":
        state[1] = fraction
    elif allocation == "equal":
        state[1:] = fraction / k
    else:
        raise ValueError(allocation)
    return state


def matrix(k, D, alpha, beta):
    rate = k * beta
    A = np.zeros((k + 1, k + 1))
    A[0, 0] = D - alpha
    A[0, k] = rate
    A[1, 0] = alpha
    A[1, 1] = -rate
    for j in range(2, k + 1):
        A[j, j - 1] = rate
        A[j, j] = -rate
    return A


def predict(times, theta, k, fraction, allocation):
    D, alpha, beta = -np.exp(theta[0]), np.exp(theta[1]), np.exp(theta[2])
    A = matrix(k, D, alpha, beta)
    x0 = initial_state(k, fraction, allocation)
    total = np.array([np.sum(expm(A * t) @ x0) for t in times])
    return np.log(np.maximum(total, 1e-300))


def fit(data, k, fraction, allocation, seed, warm=None, starts=28):
    times = data.time_h.to_numpy(float)
    obs = data.log_relative_abundance.to_numpy(float)
    lo, hi = BOUNDS
    rng = np.random.default_rng(seed)
    candidates = []
    if warm is not None:
        candidates.append(np.asarray(warm, float))
    candidates.extend(rng.uniform(lo, hi, size=(starts, 3)))
    # Add starts around the rates found in the primary two-state analysis.
    candidates.extend([
        np.log([4.5, 1e-5, 0.5]), np.log([4.5, 1e-3, 0.5]),
        np.log([4.5, 1e-5, 1.4]), np.log([4.5, 1e-3, 1.4]),
    ])
    best = None
    for x0 in candidates:
        result = least_squares(
            lambda th: predict(times, th, k, fraction, allocation) - obs,
            np.clip(x0, lo, hi), bounds=(lo, hi),
            xtol=1e-11, ftol=1e-11, gtol=1e-11, max_nfev=2500,
        )
        sse = float(np.sum(result.fun ** 2))
        if best is None or sse < best[0]:
            best = (sse, result.x)
    sse, theta = best
    D, alpha, beta = -np.exp(theta[0]), np.exp(theta[1]), np.exp(theta[2])
    # Gaussian SSE likelihood: three kinetic parameters plus residual variance.
    n, n_likelihood_parameters = len(obs), 4
    k_par = n_likelihood_parameters
    aicc = (
        n * np.log(sse / n)
        + 2 * k_par
        + 2 * k_par * (k_par + 1) / (n - k_par - 1)
    )
    return {"sse": sse, "aicc": aicc, "theta": theta,
            "D": D, "alpha": alpha, "beta": beta}


def population_analysis():
    raw = pd.read_csv(DATA / "real_replicate_log_data.csv")
    rows, curves = [], []
    counter = 0
    for condition in ["ciprofloxacin", "ampicillin"]:
        d = raw[raw.condition == condition].copy()
        for scenario, fraction in [("0.1%", 0.001), ("anchor", ANCHORS[condition])]:
            for allocation in ["stage1", "equal"]:
                full = {}
                for k in K_POPULATION:
                    counter += 1
                    full[k] = fit(d, k, fraction, allocation, SEED + counter)
                min_aicc = min(v["aicc"] for v in full.values())
                # Leave-one-time-point-out prediction with identical folds.
                unique_times = np.sort(d.time_h.unique())
                fold_sq = {k: [] for k in full}
                for fold, held_time in enumerate(unique_times[1:], start=1):
                    train = d[d.time_h != held_time]
                    test = d[d.time_h == held_time]
                    for k in full:
                        counter += 1
                        fitted = fit(train, k, fraction, allocation, SEED + counter,
                                     warm=full[k]["theta"], starts=8)
                        pred = predict(test.time_h.to_numpy(float), fitted["theta"],
                                       k, fraction, allocation)
                        fold_sq[k].extend((pred - test.log_relative_abundance.to_numpy(float)) ** 2)
                for k, res in full.items():
                    rows.append({
                        "condition": condition, "scenario": scenario,
                        "initial_allocation": allocation, "k": k,
                        "D": res["D"], "alpha": res["alpha"], "beta": res["beta"],
                        "sse": res["sse"], "aicc": res["aicc"],
                        "delta_aicc": res["aicc"] - min_aicc,
                        "loto_rmse": float(np.sqrt(np.mean(fold_sq[k]))),
                    })
                    grid = np.linspace(0, 7, 281)
                    values = predict(grid, res["theta"], k, fraction, allocation)
                    curves.extend({"condition": condition, "scenario": scenario,
                                   "initial_allocation": allocation, "k": k,
                                   "time_h": t, "log_relative_abundance": y}
                                  for t, y in zip(grid, values))
    return pd.DataFrame(rows), pd.DataFrame(curves)


def lag_analysis():
    lag = pd.read_csv(DATA / "single_cell_lags.csv")
    if "censored" in lag:
        lag = lag[~lag.censored.astype(bool)]
    rows = []
    for condition in ["ciprofloxacin", "ampicillin"]:
        t = lag.loc[lag.condition == condition, "lag_h"].to_numpy(float)
        n = len(t)
        for k in K_LAG:
            # Erlang shape k, rate lambda. MLE lambda = k*n/sum(t).
            lam = k * n / np.sum(t)
            beta = lam / k
            loglik = np.sum(k * np.log(lam) + (k - 1) * np.log(t)
                            - lam * t - gammaln(k))
            p = 1
            aicc = -2 * loglik + 2 * p + 2 * p * (p + 1) / (n - p - 1)
            rows.append({"condition": condition, "k": k, "n": n,
                         "lambda": lam, "beta_mean_rate": beta,
                         "log_likelihood": loglik, "aicc": aicc})
    out = pd.DataFrame(rows)
    out["delta_aicc"] = out.groupby("condition").aicc.transform(lambda x: x - x.min())
    return out


def make_figure(pop, lag):
    colors = {"ciprofloxacin": "#0072B2", "ampicillin": "#D55E00"}
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2))
    # Main comparison uses the measured anchor and equal stage occupancy;
    # sensitivity to allocation is retained in the CSV and third panel.
    q = pop[(pop.scenario == "anchor") & (pop.initial_allocation == "equal")]
    for condition in ["ciprofloxacin", "ampicillin"]:
        d = q[q.condition == condition]
        axes[0].plot(d.k, d.delta_aicc, "o-", color=colors[condition], label=condition.capitalize())
        axes[1].plot(d.k, d.loto_rmse, "o-", color=colors[condition], label=condition.capitalize())
    for condition in ["ciprofloxacin", "ampicillin"]:
        d = lag[lag.condition == condition]
        axes[2].plot(d.k, d.delta_aicc, "o-", color=colors[condition], label=condition.capitalize())
    axes[0].set_title("Population fit: relative AICc")
    axes[0].set_ylabel(r"$\Delta$AICc")
    axes[1].set_title("Population prediction")
    axes[1].set_ylabel("Leave-one-time-point-out RMSE")
    axes[2].set_title("Single-cell lag distribution")
    axes[2].set_ylabel(r"$\Delta$AICc")
    for i, ax in enumerate(axes):
        ax.set_xlabel("Number of persister stages, k")
        ax.set_xticks(list(K_POPULATION) if i < 2 else list(K_LAG))
        ax.grid(axis="y", alpha=0.25)
        ax.text(-0.12, 1.04, chr(65 + i), transform=ax.transAxes,
                fontweight="bold", fontsize=12)
    axes[2].legend(frameon=False)
    fig.suptitle("Sensitivity to non-exponential resuscitation", fontweight="semibold")
    fig.tight_layout()
    fig.savefig(HERE / "fig12_nonexponential_recovery.png", dpi=600,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    pop, curves = population_analysis()
    lag = lag_analysis()
    pop.to_csv(HERE / "erlang_population_model_comparison.csv", index=False)
    curves.to_csv(HERE / "erlang_population_fitted_curves.csv", index=False)
    lag.to_csv(HERE / "erlang_single_cell_model_comparison.csv", index=False)
    make_figure(pop, lag)
    summary = {
        "model": "Erlang k-stage resuscitation; k=1 is the original exponential model",
        "mean_resuscitation_time": (
            "1/beta for a cell entering P1; an initially uniform allocation "
            "across stages has mean remaining time (k+1)/(2*k*beta)"
        ),
        "population_k_values": list(K_POPULATION),
        "single_cell_k_values": list(K_LAG),
        "initial_allocations": ["stage1", "equal"],
        "selection": "AICc and leave-one-time-point-out RMSE",
        "seed": SEED,
    }
    (HERE / "erlang_analysis_protocol.json").write_text(json.dumps(summary, indent=2))
    print(pop.to_string(index=False))
    print("\nSINGLE-CELL\n", lag.to_string(index=False))


if __name__ == "__main__":
    main()
