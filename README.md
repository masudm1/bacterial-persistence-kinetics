# Bacterial persistence kinetics: code and reproducibility package

This repository accompanies the manuscript:

**Recovering Bacterial Persistence Kinetics from Kill-Curve Data: Identifiability and a Matched Comparison of Physics-Informed and Classical Estimation**

Authors: Md Masud Mondol, Shahrear Pabel, Md Shahidul Islam, and Jin Wang.

## Purpose

The study asks whether a method that predicts a bacterial time-kill curve well
also recovers the kinetic rates used in a two-compartment persistence model.
The main comparison is between multistart nonlinear least squares (NLLS) and a
physics-informed neural network (PINN). An architecture-matched neural network
without the ODE residual is included as a prediction comparator.

This repository contains the final analysis code, recorded computational
outputs, processed tables used for plotting, figure-generation scripts, and
the final manuscript figures.

## Repository contents

| Path | Contents |
|---|---|
| `analysis/` | Synthetic experiments, model fitting, validation, uncertainty analyses, identifiability diagnostics, and recorded outputs |
| `figure_code/` | Python scripts and processed tables used to generate the manuscript figures |
| `erlang_extension/` | Erlang-chain residence-time sensitivity analysis |
| `final_manuscript_figures/` | Final PNG figures used in the manuscript |
| `generate_all_final_figures.py` | One command for regenerating the complete figure set |
| `DATA_SOURCES.md` | Original data sources, expected filenames, and access links |
| `CITATION.cff` | Citation metadata for this repository |

Development-only scripts, obsolete intermediate results, and manuscript
editing notes are intentionally not included.

## Quick start: regenerate the figures

Python 3.10 or newer is recommended.

### Windows PowerShell

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
py -m pip install --upgrade pip
py -m pip install -r requirements.txt
py generate_all_final_figures.py --skip-erlang-refit
```

### macOS or Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python generate_all_final_figures.py --skip-erlang-refit
```

The command regenerates Figures 0 through 11 from the included processed
tables and retains the supplied final Figure 12. The figures are collected in
`final_manuscript_figures/`.

To refit the Erlang models and regenerate Figure 12 as well, run:

```bash
python generate_all_final_figures.py
```

The Erlang refit is slower because it performs multistart nonlinear fitting
and leave-one-time-point-out validation.

## Reproducibility scope

| Task | Status |
|---|---|
| Regenerate every manuscript figure from included processed tables | Supported |
| Inspect the code and recorded outputs for the reported analyses | Supported |
| Rerun the synthetic experiments | Supported; neural-network runs are computationally expensive |
| Rerun all empirical analyses from the original files | Supported after downloading the three cited third-party data sources |
| Obtain bit-for-bit identical neural-network results on every machine | Not guaranteed because numerical results can depend on software versions and hardware |

The original third-party workbooks are not redistributed in this repository.
See `DATA_SOURCES.md` for the official access locations and expected filenames.
The processed tables needed to regenerate the figures are included.

## Figure-to-script map

| Figure file | Content | Script |
|---|---|---|
| `fig0_workflow.png` | Study workflow | `figure_code/make_all_manuscript_figures.py` |
| `fig1_model.png` | Two-compartment model | `figure_code/make_all_manuscript_figures.py` |
| `fig2_architecture.png` | PINN architecture | `figure_code/generate_fig2_architecture.py` |
| `fig3_synthetic_data.png` | Synthetic observations | `figure_code/make_all_manuscript_figures.py` |
| `fig4_synthetic_benchmark.png` | Synthetic recovery and prediction | `figure_code/generate_updated_figures.py` |
| `fig5_ratio_sweep.png` | Rate-ratio sensitivity | `figure_code/generate_updated_figures.py` |
| `fig6_antibiotic_fits.png` | Antibiotic population fits | `figure_code/make_all_manuscript_figures.py` |
| `fig7_profile_likelihood.png` | Profile diagnostics | `figure_code/make_all_manuscript_figures.py` |
| `fig8_initial_fraction.png` | Initial-fraction sensitivity | `figure_code/make_all_manuscript_figures.py` |
| `fig9_single_cell_lags.png` | First-division lag distributions | `figure_code/make_all_manuscript_figures.py` |
| `fig10_crosslab_fits.png` | Additional experimental fits | `figure_code/make_all_manuscript_figures.py` |
| `fig11_real_heldout.png` | Experimental held-out errors | `figure_code/generate_updated_figures.py` |
| `fig12_nonexponential_recovery.png` | Erlang residence-time sensitivity | `erlang_extension/erlang_recovery_extension.py` |

The root generator runs the plotting scripts in the required order. Some
figures are redrawn by a later script because they use the final panel layout.

## Full analysis workflow

The complete command sequence is documented in `analysis/README.md`. The main
rules used throughout the final analysis are:

1. Biological replicates are retained as separate observations.
2. Each replicate is normalized by its own time-zero measurement.
3. Training and held-out splits are made by time point.
4. NLLS, PINN, and the unconstrained neural comparator use the same training
   rows, held-out rows, and initial conditions.
5. PINN candidates are selected using the training objective, without using
   held-out observations.
6. Synthetic observations are deterministic ODE totals with independent
   Gaussian measurement error in population space.
7. Zero-count observations are retained in extraction records but excluded
   from logarithmic fitting without imputation.

The experimental held-out split summaries are descriptive because repeated
splits reuse the same biological measurements.

## Citation

Citation metadata are provided in `CITATION.cff`. After the manuscript is
published, the article DOI and final bibliographic details should be added to
this file and to this README.

## Licence

The software code in this repository is released under the MIT License. The
original experimental datasets remain subject to the terms specified by their
respective publishers and repositories.

