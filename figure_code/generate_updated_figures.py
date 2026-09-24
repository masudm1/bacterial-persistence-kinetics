#!/usr/bin/env python3
"""Generate the final distribution-based benchmark figures."""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PIPE = HERE / "figure_data"

COLORS = {"NLLS": "#0072B2", "PINN": "#D55E00", "NN": "#009E73"}
MARKERS = {"NLLS": "o", "PINN": "s", "NN": "^"}


def style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.titlesize": 11, "axes.titleweight": "semibold",
        "axes.labelsize": 10, "axes.linewidth": 0.8,
        "xtick.labelsize": 9, "ytick.labelsize": 9,
        "legend.fontsize": 8.5, "savefig.dpi": 600,
        "axes.spines.top": False, "axes.spines.right": False,
    })


def save(fig, name):
    fig.savefig(HERE / name, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def add_distribution(ax, centre, values, method, rng, label=None):
    values = np.asarray(values, dtype=float)
    jitter = rng.uniform(-0.035, 0.035, len(values))
    ax.scatter(np.full(len(values), centre) + jitter, values, s=17,
               color=COLORS[method], alpha=0.32, edgecolors="none", zorder=1)
    med = np.median(values)
    q1, q3 = np.quantile(values, [0.25, 0.75])
    ax.vlines(centre, q1, q3, color=COLORS[method], lw=2.1, zorder=3)
    ax.hlines([q1, q3], centre - 0.045, centre + 0.045,
              color=COLORS[method], lw=1.35, zorder=3)
    ax.scatter(centre, med, marker=MARKERS[method], s=50, facecolor="white",
               edgecolor=COLORS[method], lw=1.8, zorder=4, label=label)
    return med


def synthetic_benchmark():
    df = pd.read_csv(PIPE / "confirmation_phase2_final" / "confirmation_trials.csv")
    tiers, labels = ["low", "moderate", "high"], ["5%", "10%", "20%"]
    panels = [
        ("alpha_relative_error_pct", r"Dormancy-rate recovery, $\alpha$", "Relative error (%)", ["NLLS", "PINN"], True),
        ("beta_relative_error_pct", r"Resuscitation-rate recovery, $\beta$", "Relative error (%)", ["NLLS", "PINN"], True),
        ("test_rmse_observed_log", "Held-out prediction", "Test RMSE (natural-log units)", ["NLLS", "PINN", "NN"], False),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.35))
    rng, x = np.random.default_rng(417), np.arange(3)
    for pi, (ax, (metric, title, ylabel, methods, logy)) in enumerate(zip(axes, panels)):
        offsets = np.linspace(-0.22, 0.22, len(methods))
        for method, off in zip(methods, offsets):
            for i, tier in enumerate(tiers):
                vals = df.loc[(df.tier == tier) & (df.method == method), metric].dropna()
                add_distribution(ax, x[i] + off, vals, method, rng,
                                 method if i == 0 else None)
        ax.set_xticks(x, labels); ax.set_xlabel("Measurement-error level")
        ax.set_ylabel(ylabel); ax.set_title(title, pad=8)
        ax.grid(axis="y", color="#D9D9D9", lw=0.7, alpha=0.65); ax.set_axisbelow(True)
        if logy: ax.set_yscale("log")
        ax.text(-0.12, 1.04, chr(65 + pi), transform=ax.transAxes,
                fontsize=12, fontweight="bold")
    axes[-1].legend(frameon=False, loc="upper left")
    fig.suptitle("Synthetic estimator comparison", fontsize=13,
                 fontweight="semibold", y=1.02)
    fig.tight_layout(); save(fig, "fig4_synthetic_benchmark.png")


def ratio_sweep():
    df = pd.read_csv(PIPE / "phase6_ratio_final" / "phase6_ratio_trials.csv")
    ratios = np.array(sorted(df.beta_alpha_ratio.unique()), dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.45), sharey=True)
    rng = np.random.default_rng(419)
    for pi, (ax, metric, title) in enumerate(zip(
        axes, ["alpha_relative_error_pct", "beta_relative_error_pct"],
        [r"Dormancy rate, $\alpha$", r"Resuscitation rate, $\beta$"])):
        for method, factor in [("NLLS", 0.94), ("PINN", 1.06)]:
            centres, medians = [], []
            for ratio in ratios:
                vals = df.loc[(df.beta_alpha_ratio == ratio) & (df.method == method), metric].dropna().to_numpy()
                centre = ratio * factor
                px = centre * 10 ** rng.uniform(-0.018, 0.018, len(vals))
                ax.scatter(px, vals, s=22, color=COLORS[method], alpha=0.36,
                           edgecolors="none", zorder=1)
                med = np.median(vals); q1, q3 = np.quantile(vals, [0.25, 0.75])
                ax.vlines(centre, q1, q3, color=COLORS[method], lw=2.0, zorder=3)
                ax.scatter(centre, med, marker=MARKERS[method], s=48,
                           facecolor="white", edgecolor=COLORS[method], lw=1.7, zorder=4)
                centres.append(centre); medians.append(med)
            ax.plot(centres, medians, color=COLORS[method], lw=1.35,
                    label=method, zorder=2)
        ax.axvline(100, color="#6F6F6F", ls=":", lw=1.3)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xticks(ratios); ax.set_xticklabels(["0.1", "1", "2", "10", "100", "1000"])
        ax.set_xlabel(r"True rate ratio, $\beta/\alpha$"); ax.set_title(title, pad=8)
        ax.grid(which="major", color="#D9D9D9", lw=0.7, alpha=0.65); ax.set_axisbelow(True)
        ax.text(-0.12, 1.04, chr(65 + pi), transform=ax.transAxes,
                fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Relative error (%)"); axes[1].legend(frameon=False)
    fig.suptitle("Parameter recovery across dynamical regimes", fontsize=13,
                 fontweight="semibold", y=1.02)
    fig.tight_layout(); save(fig, "fig5_ratio_sweep.png")


def real_heldout():
    a = pd.read_csv(PIPE / "phase6_final" / "phase6_real_fit_records.csv")
    a = a[a.fold > 0]
    n = pd.read_csv(PIPE / "nordholt_final_robust" / "nordholt_cv_trials.csv")
    rows = []
    for _, r in a.iterrows():
        if r.condition == "ciprofloxacin":
            name = "Cipro\n0.1%" if r.initial_condition_scenario == "assumed_0.1pct" else "Cipro\nanchor"
        elif r.condition == "ampicillin":
            name = "Ampicillin\n0.1%" if r.initial_condition_scenario == "assumed_0.1pct" else "Ampicillin\nanchor"
        else: name = "Oxacillin"
        rows.append((name, r.method, r.test_rmse_observed_log))
    names = {"nordholt_h2o2": r"H$_2$O$_2$", "nordholt_bac": "BAC", "nordholt_ddac": "DDAC"}
    for _, r in n.iterrows(): rows.append((names[r.condition], r.method, r.test_rmse_observed_log))
    df = pd.DataFrame(rows, columns=["condition", "method", "rmse"])
    order = ["Cipro\n0.1%", "Cipro\nanchor", "Ampicillin\n0.1%", "Ampicillin\nanchor",
             "BAC", "DDAC", r"H$_2$O$_2$", "Oxacillin"]
    offsets = {"NLLS": -0.22, "PINN": 0.0, "NN": 0.22}
    fig, ax = plt.subplots(figsize=(12.4, 4.55)); rng = np.random.default_rng(421)
    for method in ["NLLS", "PINN", "NN"]:
        for i, condition in enumerate(order):
            vals = df.loc[(df.condition == condition) & (df.method == method), "rmse"].dropna()
            add_distribution(ax, i + offsets[method], vals, method, rng,
                             method if i == 0 else None)
    ax.axvline(3.5, color="#BDBDBD", lw=0.9); ax.axvline(6.5, color="#BDBDBD", lw=0.9)
    for xpos, label in [(1.5, "Umetani"), (5.0, "Nordholt"), (7.0, "Peyrusson")]:
        ax.text(xpos, 1.04, label, transform=ax.get_xaxis_transform(), ha="center",
                fontsize=9, color="#6F6F6F")
    ax.set_xticks(np.arange(len(order)), order); ax.set_yscale("log")
    ax.set_ylabel("Test RMSE (natural-log units; log scale)")
    ax.set_title("Held-out prediction across experimental conditions", pad=20)
    ax.grid(axis="y", which="both", color="#D9D9D9", lw=0.7, alpha=0.65)
    ax.set_axisbelow(True); ax.legend(frameon=False, ncol=3, loc="upper left")
    fig.tight_layout(); save(fig, "fig11_real_heldout.png")


if __name__ == "__main__":
    style(); synthetic_benchmark(); ratio_sweep(); real_heldout()
