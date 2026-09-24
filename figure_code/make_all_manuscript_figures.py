#!/usr/bin/env python3
"""Create the base PNG figures from the included processed tables.

Run this script from any directory. Figures are written beside the script.
"""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle


HERE = Path(__file__).resolve().parent
PIPE = HERE / "figure_data"
sys.path.insert(0, str(PIPE))
from core import synthetic_dataset

BLUE = "#2B5DAA"
RED = "#AA2724"
GREEN = "#3A873A"
GRAY = "#777777"
METHOD_COLORS = {"NLLS": BLUE, "PINN": RED, "NN": GREEN}


def setup_style():
    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 8.5,
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def save(fig, name):
    fig.savefig(HERE / name, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def quantile_error(values):
    values = np.asarray(values, dtype=float)
    median = np.nanmedian(values)
    q1, q3 = np.nanquantile(values, [0.25, 0.75])
    return median, median - q1, q3 - median


def workflow():
    fig, ax = plt.subplots(figsize=(12.5, 5.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    def box(x, y, w, h, text, face, edge):
        p = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.018,rounding_size=0.025",
            linewidth=1.6, edgecolor=edge, facecolor=face,
        )
        ax.add_patch(p)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                linespacing=1.2)

    def arrow(start, end):
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>",
                                    mutation_scale=13, linewidth=1.4,
                                    color="#4A4A4A"))

    box(0.03, 0.60, 0.20, 0.23,
        "Synthetic observations\n3 noise levels\n20 data sets per level",
        "#E7F2FA", "#27659B")
    box(0.03, 0.17, 0.20, 0.23,
        "Experimental observations\n6 conditions\nreplicate level",
        "#FCF1E3", "#B36A18")
    box(0.32, 0.39, 0.24, 0.25,
        "Matched estimators\nNLLS  |  PINN  |  NN\n\nSame initial state, training rows,\nand test time points",
        "#F1EAF8", "#7040B0")
    box(0.66, 0.61, 0.29, 0.21,
        "Estimator performance\nParameter recovery  |  Test RMSE",
        "#E7F2FA", "#27659B")
    box(0.66, 0.32, 0.29, 0.17,
        "Mechanistic consistency\nPINN network and ODE agreement",
        "#E8F5EC", "#2D8B57")
    box(0.66, 0.08, 0.29, 0.15,
        "Inference diagnostics\nProfiles  |  Replicate uncertainty",
        "#E8F5EC", "#2D8B57")
    arrow((0.23, 0.715), (0.32, 0.555))
    arrow((0.23, 0.285), (0.32, 0.475))
    arrow((0.56, 0.54), (0.66, 0.715))
    arrow((0.56, 0.49), (0.66, 0.405))
    arrow((0.56, 0.44), (0.66, 0.155))
    ax.text(0.13, 0.92, "DATA", ha="center", weight="bold", color="#5F6368")
    ax.text(0.44, 0.74, "CONTROLLED COMPARISON", ha="center", weight="bold", color="#5F6368")
    ax.text(0.805, 0.92, "SEPARATE QUESTIONS", ha="center", weight="bold", color="#5F6368")
    save(fig, "fig0_workflow.png")


def model_diagram():
    fig, ax = plt.subplots(figsize=(10.8, 4.6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    left = FancyBboxPatch((0.08, 0.28), 0.30, 0.40,
                          boxstyle="round,pad=0.02,rounding_size=0.03",
                          facecolor="#DFEAF7", edgecolor=BLUE, linewidth=2.2)
    right = FancyBboxPatch((0.66, 0.28), 0.26, 0.40,
                           boxstyle="round,pad=0.02,rounding_size=0.03",
                           facecolor="#F7E3E3", edgecolor=RED, linewidth=2.2)
    ax.add_patch(left); ax.add_patch(right)
    ax.text(0.23, 0.48, "Active cells\n$N(t)$", ha="center", va="center",
            fontsize=15, weight="bold", color=BLUE)
    ax.text(0.79, 0.48, "Persister cells\n$P(t)$", ha="center", va="center",
            fontsize=15, weight="bold", color=RED)
    ax.add_patch(FancyArrowPatch((0.38, 0.55), (0.66, 0.55), arrowstyle="-|>",
                                 mutation_scale=16, linewidth=1.8, color="#222222"))
    ax.add_patch(FancyArrowPatch((0.66, 0.41), (0.38, 0.41), arrowstyle="-|>",
                                 mutation_scale=16, linewidth=1.8, color="#222222"))
    ax.text(0.52, 0.61, r"Dormancy  $\alpha$", ha="center", fontsize=12)
    ax.text(0.52, 0.31, r"Resuscitation  $\beta$", ha="center", fontsize=12)
    ax.add_patch(FancyArrowPatch((0.15, 0.88), (0.20, 0.68), arrowstyle="-|>",
                                 mutation_scale=16, linewidth=2.0, color=GREEN))
    ax.text(0.10, 0.91, r"Growth  $\mu$", color=GREEN, fontsize=12, weight="bold")
    ax.add_patch(FancyArrowPatch((0.23, 0.12), (0.23, 0.28), arrowstyle="-|>",
                                 mutation_scale=16, linewidth=2.0, color="#8E1A9A"))
    ax.text(0.08, 0.06, r"Treatment death  $\delta(t)$", color="#8E1A9A",
            fontsize=12, weight="bold")
    save(fig, "fig1_model.png")


def architecture_diagram():
    fig, ax = plt.subplots(figsize=(12.2, 5.4))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    def box(x, y, w, h, txt, fc, ec, fs=11):
        p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.018,rounding_size=0.025",
                           facecolor=fc, edgecolor=ec, linewidth=1.8)
        ax.add_patch(p)
        ax.text(x+w/2, y+h/2, txt, ha="center", va="center", fontsize=fs)
    def arrow(a, b):
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=14,
                                     linewidth=1.5, color="#333333"))
    c = Circle((0.08, 0.55), 0.07, facecolor="#F3F3F3", edgecolor="#555555", lw=1.8)
    ax.add_patch(c); ax.text(0.08, 0.55, "Time\n$t/T$", ha="center", va="center", fontsize=12)
    box(0.20, 0.34, 0.24, 0.42, "State network\n$\tanh$, 2 $\times$ 32\nweights $\theta$",
        "#E8E8E8", "#555555", 13)
    box(0.53, 0.64, 0.18, 0.18, r"$u=\log \widehat N$", "#DFEAF7", BLUE, 13)
    box(0.53, 0.35, 0.18, 0.18, r"$v=\log \widehat P$", "#F7E3E3", RED, 13)
    box(0.49, 0.07, 0.26, 0.16,
        r"$\alpha,\beta=\mathrm{softplus}(\theta)$"+"\n"+r"$D=-\mathrm{softplus}(\theta_D)$",
        "#E7F4EA", GREEN, 11)
    box(0.80, 0.27, 0.18, 0.48,
        "Composite loss\n\nData residual\n+\nODE residual\n\nHard initial state",
        "#E7F4EA", GREEN, 12)
    arrow((0.15,0.55),(0.20,0.55)); arrow((0.44,0.62),(0.53,0.73)); arrow((0.44,0.47),(0.53,0.44))
    arrow((0.71,0.73),(0.80,0.62)); arrow((0.71,0.44),(0.80,0.47)); arrow((0.75,0.15),(0.82,0.30))
    ax.text(0.32, 0.84, "Exact initial condition imposed through the output transform",
            ha="center", fontsize=10, color="#555555")
    save(fig, "fig2_architecture.png")


def synthetic_data_figure():
    data = synthetic_dataset(1, 0.10)
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    ax.plot(data.time_h, data.latent_total, color="black", lw=2.0,
            label="Deterministic ODE total")
    ax.scatter(data.time_h, data.observed_total, s=20, color="#858585",
               alpha=0.68, edgecolors="none", label="Gaussian measurement error (10%)")
    ax.axvspan(2, 7, color="#8E44AD", alpha=0.12, label="Antibiotic exposure")
    ax.axvline(2, color="#8E44AD", lw=0.9, alpha=0.7)
    ax.axvline(7, color="#8E44AD", lw=0.9, alpha=0.7)
    ax.set_yscale("log")
    ax.set_xlabel("Time (h)")
    ax.set_ylabel(r"Total abundance $N(t)+P(t)$")
    ax.set_title("Synthetic biphasic trajectory with heteroscedastic measurement error")
    ax.grid(alpha=0.20); ax.legend(frameon=False)
    fig.tight_layout(); save(fig, "fig3_synthetic_data.png")


def synthetic_benchmark():
    df = pd.read_csv(PIPE / "confirmation_phase2_final" / "confirmation_trials.csv")
    tier_order = ["low", "moderate", "high"]
    labels = ["5%", "10%", "20%"]
    panels = [
        ("alpha_relative_error_pct", r"$\alpha$ relative error", ["NLLS", "PINN"]),
        ("beta_relative_error_pct", r"$\beta$ relative error", ["NLLS", "PINN"]),
        ("test_rmse_observed_log", "Test RMSE", ["NLLS", "PINN", "NN"]),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2))
    x = np.arange(3)
    for ax, (metric, title, methods) in zip(axes, panels):
        offsets = np.linspace(-0.18, 0.18, len(methods))
        for off, method in zip(offsets, methods):
            meds, lows, highs = [], [], []
            for tier in tier_order:
                v = df.loc[(df.tier == tier) & (df.method == method), metric]
                m, lo, hi = quantile_error(v)
                meds.append(m); lows.append(lo); highs.append(hi)
            ax.errorbar(x + off, meds, yerr=[lows, highs], fmt="o", capsize=4,
                        lw=1.6, ms=6, color=METHOD_COLORS[method], label=method)
        ax.set_xticks(x, labels)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Relative error (%)")
    axes[1].set_ylabel("Relative error (%)")
    axes[2].set_ylabel("Natural log units")
    axes[2].legend(frameon=False)
    fig.suptitle("Frozen synthetic confirmation: median and interquartile range", y=1.02)
    fig.tight_layout()
    save(fig, "fig4_synthetic_benchmark.png")


def ratio_sweep():
    df = pd.read_csv(PIPE / "phase6_ratio_final" / "phase6_ratio_trials.csv")
    ratios = np.array(sorted(df.beta_alpha_ratio.unique()))
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2), sharey=True)
    for ax, metric, title in zip(
        axes,
        ["alpha_relative_error_pct", "beta_relative_error_pct"],
        [r"$\alpha$ relative error", r"$\beta$ relative error"],
    ):
        for method in ["NLLS", "PINN"]:
            meds, lows, highs = [], [], []
            for ratio in ratios:
                v = df.loc[(df.beta_alpha_ratio == ratio) & (df.method == method), metric]
                m, lo, hi = quantile_error(v)
                meds.append(m); lows.append(lo); highs.append(hi)
            ax.errorbar(ratios, meds, yerr=[lows, highs], marker="o", capsize=3,
                        lw=1.5, color=METHOD_COLORS[method], label=method)
        ax.axvline(100, color=GRAY, ls=":", lw=1.5)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"True $\beta/\alpha$")
        ax.set_title(title)
        ax.grid(alpha=0.25, which="major")
    axes[0].set_ylabel("Relative error (%)")
    axes[1].legend(frameon=False)
    fig.suptitle("Frozen protocol across parameter ratios: median and interquartile range", y=1.02)
    fig.tight_layout()
    save(fig, "fig5_ratio_sweep.png")


def antibiotic_fits():
    curves = pd.read_csv(PIPE / "phase6_final" / "phase6_full_fit_curves.csv")
    u = pd.read_csv(PIPE / "outputs" / "real_replicate_log_data.csv")
    p = pd.read_csv(PIPE / "outputs" / "peyrusson_replicate_log_data.csv")
    specs = [
        ("ciprofloxacin", "assumed_0.1pct", "Ciprofloxacin, 0.1%"),
        ("ciprofloxacin", "classification_anchor", "Ciprofloxacin, anchor"),
        ("ampicillin", "assumed_0.1pct", "Ampicillin, 0.1%"),
        ("ampicillin", "classification_anchor", "Ampicillin, anchor"),
    ]
    styles = {
        "NLLS": (BLUE, "-"), "NN": (GREEN, "--"),
        "PINN-network": (RED, "-"), "PINN-implied-ODE": (RED, ":"),
    }
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.0), sharey=False)
    axes = axes.ravel()
    for ax, (condition, scenario, title) in zip(axes, specs):
        obs = p if condition.startswith("peyrusson") else u
        obs = obs[obs.condition == condition]
        ax.scatter(obs.time_h, obs.log_relative_abundance, s=12, c="#888888",
                   alpha=0.65, edgecolors="none", label="Replicates")
        sub = curves[(curves.condition == condition) &
                     (curves.initial_condition_scenario == scenario)]
        for method, (color, ls) in styles.items():
            q = sub[sub.method == method]
            ax.plot(q.time_h, q.log_relative_abundance, color=color, ls=ls,
                    lw=1.5, label=method)
        ax.set_title(title)
        ax.set_xlabel("Time (h)")
        ax.grid(alpha=0.20)
    axes[0].set_ylabel(r"$\ln[Y(t)/Y(0)]$")
    handles, labels = axes[-1].get_legend_handles_labels()
    axes[-1].legend(handles, labels, frameon=False, loc="best", fontsize=7)
    fig.tight_layout()
    save(fig, "fig6_antibiotic_fits.png")


def disinfectant_fits():
    curves = pd.read_csv(PIPE / "nordholt_final_robust" / "nordholt_full_fit_curves.csv")
    obs = pd.read_csv(PIPE / "nordholt_final_robust" / "nordholt_replicate_data.csv")
    pey_curves = pd.read_csv(PIPE / "phase6_final" / "phase6_full_fit_curves.csv")
    pey_obs = pd.read_csv(PIPE / "outputs" / "peyrusson_replicate_log_data.csv")
    specs = [
        ("nordholt_h2o2", r"H$_2$O$_2$"),
        ("nordholt_bac", "BAC"),
        ("nordholt_ddac", "DDAC"),
        ("peyrusson_extracellular_oxacillin", "Oxacillin"),
    ]
    styles = {
        "NLLS": (BLUE, "-"), "NN": (GREEN, "--"),
        "PINN-network": (RED, "-"), "PINN-implied-ODE": (RED, ":"),
    }
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.2))
    axes = axes.ravel()
    for ax, (condition, title) in zip(axes, specs):
        if condition.startswith("peyrusson"):
            o = pey_obs[pey_obs.condition == condition]
            ax.scatter(o.time_h, o.log_relative_abundance, s=17,
                       c="#6B9FC4", alpha=0.65, edgecolors="none", label="Replicates")
            sub = pey_curves[(pey_curves.condition == condition) &
                             (pey_curves.initial_condition_scenario == "assumed_0.1pct")]
            for method, (color, ls) in styles.items():
                q = sub[sub.method == method]
                ax.plot(q.time_h, q.log_relative_abundance, color=color,
                        ls=ls, lw=1.5, label=method)
            ax.set_title(title); ax.set_xlabel("Time (h)"); ax.grid(alpha=0.20)
            continue
        o = obs[obs.condition == condition]
        pos = o[~o.is_zero_count]
        ax.scatter(pos.time_min, pos.log_relative_abundance, s=17,
                   c="#6B9FC4", alpha=0.65, edgecolors="none",
                   label="Positive counts")
        y_floor = pos.log_relative_abundance.min() - 0.5
        zero = o[o.is_zero_count]
        if len(zero):
            ax.scatter(zero.time_min, np.full(len(zero), y_floor), marker="v",
                       s=22, c="black", label="Zero count")
        sub = curves[curves.condition == condition]
        for method, (color, ls) in styles.items():
            q = sub[sub.method == method]
            ax.plot(60 * q.time_h, q.log_relative_abundance, color=color,
                    ls=ls, lw=1.5, label=method)
        if condition == "nordholt_h2o2":
            last_positive = pos.time_min.max()
            ax.axvspan(last_positive, sub.time_h.max() * 60, color="#EEEEEE", alpha=0.7)
            ax.text(last_positive + 0.5, 0.03, "curve extrapolation",
                    color=GRAY, fontsize=8)
            ax.set_ylim(y_floor - 0.5, pos.log_relative_abundance.max() + 0.5)
        ax.set_title(title)
        ax.set_xlabel("Time (min)")
        ax.grid(alpha=0.20)
    axes[0].set_ylabel(r"$\ln[Y(t)/Y(0)]$")
    handles, labels = axes[-1].get_legend_handles_labels()
    axes[-1].legend(handles, labels, frameon=False, fontsize=7)
    fig.tight_layout()
    save(fig, "fig10_crosslab_fits.png")


def real_heldout():
    a = pd.read_csv(PIPE / "phase6_final" / "phase6_real_fit_records.csv")
    a = a[a.fold > 0].copy()
    n = pd.read_csv(PIPE / "nordholt_final_robust" / "nordholt_cv_trials.csv")
    rows = []
    for _, r in a.iterrows():
        if r["condition"] == "ciprofloxacin":
            name = "Cipro\n0.1%" if r.initial_condition_scenario == "assumed_0.1pct" else "Cipro\nanchor"
        elif r["condition"] == "ampicillin":
            name = "Ampicillin\n0.1%" if r.initial_condition_scenario == "assumed_0.1pct" else "Ampicillin\nanchor"
        else:
            name = "Oxacillin"
        rows.append((name, r.method, r.test_rmse_observed_log))
    for _, r in n.iterrows():
        name = {"nordholt_h2o2": r"H$_2$O$_2$", "nordholt_bac": "BAC",
                "nordholt_ddac": "DDAC"}[r["condition"]]
        rows.append((name, r.method, r.test_rmse_observed_log))
    df = pd.DataFrame(rows, columns=["condition", "method", "rmse"])
    order = ["Cipro\n0.1%", "Cipro\nanchor", "Ampicillin\n0.1%",
             "Ampicillin\nanchor", "BAC", "DDAC", r"H$_2$O$_2$", "Oxacillin"]
    fig, ax = plt.subplots(figsize=(12.3, 4.1))
    x = np.arange(len(order))
    offsets = {"NLLS": -0.20, "PINN": 0.0, "NN": 0.20}
    for method in ["NLLS", "PINN", "NN"]:
        meds, lows, highs = [], [], []
        for condition in order:
            v = df.loc[(df.condition == condition) & (df.method == method), "rmse"]
            m, lo, hi = quantile_error(v)
            meds.append(m); lows.append(lo); highs.append(hi)
        ax.errorbar(x + offsets[method], meds, yerr=[lows, highs], fmt="o",
                    capsize=4, lw=1.5, ms=5.5, color=METHOD_COLORS[method],
                    label=method)
    ax.set_xticks(x, order)
    ax.set_ylabel("Test RMSE (natural log units)")
    ax.set_title("Matched time blocked evaluation: median and interquartile range")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=3)
    fig.tight_layout()
    save(fig, "fig11_real_heldout.png")


def profile_likelihood():
    df = pd.read_csv(PIPE / "outputs" / "profile_likelihood.csv")
    specs = [
        ("synthetic_moderate_trial_1", "Synthetic, 10% noise"),
        ("ciprofloxacin", "Ciprofloxacin"),
        ("ampicillin", "Ampicillin"),
    ]
    markers = {
        ("synthetic_moderate_trial_1", "alpha"): 0.001,
        ("synthetic_moderate_trial_1", "beta"): 0.1,
        ("ciprofloxacin", "alpha"): 1e-8,
        ("ciprofloxacin", "beta"): 0.544111,
        ("ampicillin", "alpha"): 1e-8,
        ("ampicillin", "beta"): 1.407855,
    }
    fig, axes = plt.subplots(2, 3, figsize=(11.2, 6.8))
    for col, (condition, title) in enumerate(specs):
        for row, parameter in enumerate(["alpha", "beta"]):
            ax = axes[row, col]
            d = df[(df.condition == condition) & (df.parameter == parameter)]
            ax.plot(d.fixed_value, d.likelihood_ratio_statistic, "o-", ms=3,
                    lw=1.4, color=BLUE if parameter == "alpha" else RED)
            ax.axhline(3.841459, color=GRAY, ls="--", lw=1.2,
                       label=r"$\chi^2_{1,0.95}$")
            ax.axvline(markers[(condition, parameter)], color="#168C2C", ls=":", lw=1.3)
            ax.set_xscale("log")
            ax.set_xlabel(rf"${parameter}$ ($h^{{-1}}$)")
            ax.set_ylabel(r"$\Lambda$")
            ax.set_title(f"{title}: " + rf"${parameter}$")
            ax.set_ylim(bottom=0)
            ax.grid(alpha=0.20)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.tight_layout(); save(fig, "fig7_profile_likelihood.png")


def initial_fraction_and_lags():
    sens = pd.read_csv(PIPE / "outputs" / "initial_fraction_sensitivity.csv")
    lags = pd.read_csv(PIPE / "outputs" / "single_cell_lags.csv")
    anchors = {"ciprofloxacin": 32 / 197954, "ampicillin": 6 / 377167.8211}
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.2))
    for ax, condition, title in zip(axes, ["ciprofloxacin", "ampicillin"],
                                    ["Ciprofloxacin", "Ampicillin"]):
        d = sens[sens.condition == condition]
        ax.plot(100 * d.initial_fraction, d.beta, "o-", color=RED, lw=1.5, ms=4)
        ax.axvline(0.1, color=GRAY, ls=":", lw=1.5, label="0.1% assumption")
        ax.axvline(100 * anchors[condition], color=BLUE, lw=1.5,
                   label="Classification anchor")
        ax.set_xscale("log")
        ax.set_xlabel("Initial persister fraction (%)")
        ax.set_ylabel(r"NLLS $\beta$ ($h^{-1}$)")
        ax.set_title(title)
        ax.grid(alpha=0.2)
        ax.legend(frameon=False)
    fig.tight_layout()
    save(fig, "fig8_initial_fraction.png")

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.0))
    for ax, condition, title in zip(axes, ["ciprofloxacin", "ampicillin"],
                                    ["Ciprofloxacin lags", "Ampicillin lags"]):
        d = lags[(lags.condition == condition) & (~lags.censored)]
        bins = min(6, max(3, len(d) // 3))
        ax.hist(d.lag_h, bins=bins, color="#4C78A8", alpha=0.9,
                edgecolor="white")
        ax.axvline(d.lag_h.mean(), color=RED, ls="--", lw=1.5,
                   label=f"Mean = {d.lag_h.mean():.2f} h")
        ax.set_xlabel("First division after washout (h)")
        ax.set_ylabel("Lineages")
        ax.set_title(title)
        ax.legend(frameon=False)
        ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    save(fig, "fig9_single_cell_lags.png")


def main():
    setup_style()
    workflow()
    model_diagram()
    architecture_diagram()
    synthetic_data_figure()
    synthetic_benchmark()
    ratio_sweep()
    antibiotic_fits()
    profile_likelihood()
    disinfectant_fits()
    real_heldout()
    initial_fraction_and_lags()


if __name__ == "__main__":
    main()
