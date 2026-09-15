# 50 cm: direct precipitation × predicted 5 cm input

This experiment keeps the audited rows, physical-site identities,
five outer folds, and three seeds (42, 7, 555). Only the two input switches vary.

| Configuration | Predicted 5 cm stream | Direct P7/P30/P90 |
|---|---|---|
| A | yes | no |
| B | no | no |
| C | yes | yes |
| D | no | yes |

The upstream 5 cm model already uses precipitation. Thus “no direct precipitation”
does not imply absence of indirect precipitation information through the cascade.
The common fitting policy selects an epoch on an internal 15% site holdout, then
refits on all permitted outer-training sites for at least 10 epochs. The original
150-epoch learning-rate and Huber schedules, model dimensions, dropout and loss
remain fixed. The upstream cascade inputs remain fixed too. These are downstream
input sensitivity fits, not an end-to-end retrained release.

`protocol.json` was frozen before the new fits. All OOF outputs are completed and
ranked by mean three-seed OOF RMSE before the new test inference. Historic test
results have already been examined; this is not a new untouched test benchmark.
All declared configurations and contrasts are reported, regardless of outcome.

## Files

- `run_factorial.py`: fitting and protocol/source verification.
- `check_design.py`: development-only row, input and upstream-exclusion checks.
- `analyze_factorial.py`: OOF freeze, gated test inference, statistics and report.
- `verify_factorial.py`: second implementation of metrics/bootstrap, source-hash
  checks, preprocessing reconstruction and full saved-weight prediction replay.
- `models/{A,B,C,D}_seed{42,7,555}/outer{0..4}/`: manifests and predictions.
- Each fold manifest records `weight_directory`, relative to the repository root.
  New fits also have `model.pt` and `preprocessor.joblib` here. Identity-reused
  fits point explicitly to the original or prior verified floor-10 checkpoint.
- `oof_freeze.json`: fixed ranking and hashes before test inference.
- `results.json`, `RESULTS_ZH.md`, `independent_verification.json`: final evidence.

Training inputs are in `../data/cascade/50cm/outer{k}/train.parquet`; OOF and test
inputs are in the paired `oof.parquet` and `test.parquet`. Shared Python modules
are in `../oasm/`. No original models, source tables, manuscripts, or released
reference products are replaced by this experiment.

## Reproduce / resume

Use the same ML environment as the main package (PyTorch 2.7.1+cu118, NumPy, pandas,
scikit-learn and joblib). From this directory:

```bash
python -X utf8 run_factorial.py --stage freeze
python -X utf8 check_design.py
python -X utf8 run_factorial.py --configs ABCD
python -X utf8 analyze_factorial.py
python -X utf8 verify_factorial.py
```

Existing artifacts are checked and reused. Freeze verification deliberately fails
if the frozen sources or fitting script change; use a new experiment directory
for a genuinely changed protocol. The current two training workers cover `C` and
`ABD`, respectively; `workers.json` and the separate logs record their execution.
