# Analysis workflow

This directory contains the final computational workflow and recorded outputs
used in the manuscript. Run the commands below from this directory.

## 1. Install dependencies

From the repository root:

```bash
python -m pip install -r requirements.txt
```

The full neural-network analyses require PyTorch and can take several hours or
longer, depending on the computer. Recorded outputs are included so the final
figures can be regenerated without rerunning every fit.

## 2. Download the original data

Download the three source datasets described in `../DATA_SOURCES.md`:

- the extracted Umetani Dryad directory;
- `41467_2020_15966_MOESM3_ESM.xlsx`; and
- `spectrum.03276-22-s0002.xlsx`.

The commands below use placeholders for their local paths.

## 3. Verify and extract the source data

```bash
python verify_pipeline.py --dryad-dir /path/to/DataForDryad_Uploaded

python run_pipeline.py \
  --dryad-dir /path/to/DataForDryad_Uploaded \
  --output-dir outputs \
  --data-weight 0.3

python run_peyrusson.py \
  --workbook /path/to/41467_2020_15966_MOESM3_ESM.xlsx \
  --output-dir outputs \
  --data-weight 0.3

python single_cell_analysis.py \
  --dryad-dir /path/to/DataForDryad_Uploaded \
  --output-dir outputs
```

The Umetani population data use all available replicate-level observations.
The single-cell summary reports lag estimates and descriptive distribution
parameters. It does not report ordinary fixed-parameter Kolmogorov-Smirnov
probabilities for distributions fitted to the same observations.

## 4. PINN calibration and frozen synthetic confirmation

The calibration seeds are separate from the reported confirmation seeds.

```bash
python calibrate_pinn.py \
  --output calibration/pinn_weight_calibration.csv

python diagnose_pinn_collapse.py \
  --output-dir calibration_phase2
```

The final PINN rule is stored in
`calibration_phase2/frozen_pinn_rule.json`. To rerun the four independent
confirmation blocks:

```bash
python run_frozen_confirmation.py \
  --rule calibration_phase2/frozen_pinn_rule.json \
  --output-dir confirmation_blocks/block_0 \
  --seed-block 0

python run_frozen_confirmation.py \
  --rule calibration_phase2/frozen_pinn_rule.json \
  --output-dir confirmation_blocks/block_1 \
  --seed-block 1

python run_frozen_confirmation.py \
  --rule calibration_phase2/frozen_pinn_rule.json \
  --output-dir confirmation_blocks/block_2 \
  --seed-block 2

python run_frozen_confirmation.py \
  --rule calibration_phase2/frozen_pinn_rule.json \
  --output-dir confirmation_blocks/block_3 \
  --seed-block 3

python merge_confirmation_blocks.py \
  --blocks-root confirmation_blocks \
  --output-dir confirmation_phase2_final

python verify_phase2.py
```

The four block commands are independent and may be run separately. The
included `confirmation_phase2_final/` directory contains the merged recorded
outputs used for the manuscript.

## 5. Umetani uncertainty analyses

```bash
python downstream_analysis.py \
  --real-data outputs/real_replicate_log_data.csv \
  --output-dir outputs \
  --data-weight 0.3

python cluster_bootstrap.py \
  --real-data outputs/real_replicate_log_data.csv \
  --output-dir outputs \
  --resamples 200
```

## 6. Nordholt analyses

```bash
python run_nordholt.py \
  --workbook /path/to/spectrum.03276-22-s0002.xlsx \
  --output-dir nordholt_final_robust \
  --trials 6 \
  --epochs 2500 \
  --lbfgs-steps 250 \
  --data-weight 0.3 \
  --pinn-weight-seeds 3

python diagnose_nordholt_pinn_starts.py \
  --workbook /path/to/spectrum.03276-22-s0002.xlsx \
  --output nordholt_final_robust/nordholt_pinn_start_diagnostics.csv \
  --epochs 2500 \
  --lbfgs-steps 250 \
  --data-weight 0.3 \
  --weight-seeds 3

python nordholt_identifiability.py \
  --data nordholt_final_robust/nordholt_replicate_data.csv \
  --full-fits nordholt_final_robust/nordholt_full_data_fits.csv \
  --source-workbook /path/to/spectrum.03276-22-s0002.xlsx \
  --output-dir nordholt_identifiability_final_refined \
  --profile-points 61

python verify_nordholt.py \
  --workbook /path/to/spectrum.03276-22-s0002.xlsx \
  --output-dir nordholt_final_robust

python verify_nordholt_identifiability.py \
  --workbook /path/to/spectrum.03276-22-s0002.xlsx \
  --output-dir nordholt_identifiability_final_refined
```

## 7. Final real-data and rate-ratio analyses

```bash
python run_phase6_real.py \
  --umetani-data outputs/real_replicate_log_data.csv \
  --peyrusson-data outputs/peyrusson_replicate_log_data.csv \
  --output-dir phase6_final \
  --workers 4

python peyrusson_identifiability.py \
  --data outputs/peyrusson_replicate_log_data.csv \
  --phase6-fits phase6_final/phase6_full_data_fits.csv \
  --source-workbook /path/to/41467_2020_15966_MOESM3_ESM.xlsx \
  --output-dir phase6_peyrusson_identifiability \
  --profile-points 61

python run_phase6_ratio.py \
  --rule calibration_phase2/frozen_pinn_rule.json \
  --output-dir phase6_ratio_final \
  --workers 4

python verify_phase6.py
```

The protocol JSON files in the output directories record seeds, settings,
source checksums, and analysis rules. Where settings differ across analyses,
the corresponding protocol file is authoritative.

## 8. Erlang-chain sensitivity analysis

Run this command from the repository root:

```bash
python erlang_extension/erlang_recovery_extension.py
```

For aggregate Gaussian-residual AICc, the script counts four likelihood
parameters: three kinetic parameters and the residual variance. The
single-cell candidate shapes are fixed, so only the mean-rate parameter is
estimated within each candidate model.

## Interpretation notes

- The mechanistic parameter comparison is between NLLS and PINN. The
  unconstrained neural network is a prediction comparator and does not estimate
  the same kinetic rates.
- Experimental repeated-split errors are descriptive because the splits reuse
  the same biological measurements.
- Zero counts are excluded from log-space fitting without imputation; the
  resulting fits are conditional on positive observations.
- Exact equality of neural-network results across hardware and software
  versions is not guaranteed.

