# Files distributed through the Zenodo record

The Zenodo record holds the archives below. Every archive unpacks into the same top-level folder `OASM10/`; extract them into one directory; `OASM10_repository.zip` is identical to this repository.

| Archive | Contents |
|---|---|
| `OASM10_repository.zip` | Public code repository: package, scripts, notebooks, protocols, frozen experiment records, model manifests, analysis and validation summaries, figures and the eight example GeoTIFF sets (identical to the GitHub tree) |
| `OASM10_models_reference.zip` | Product reference models for inference: the five outer-fold weights (model.pt) and training-only preprocessing transforms (preprocessor.joblib) of models/5cm/ma_seed42, models/20cm/ma_seed42 and models/50cm/ma_floor10_seed42 |
| `OASM10_models_training_extras.zip` | Weights and transforms needed only to reproduce training: the nested upstream surface fits (models/upstream_nested) that generate the deep-training cascade inputs, and the 50 cm selected-duration anchor fit (models/50cm/ma_seed42) used by the companion comparisons |

Paths that this repository omits because of size. Items marked "not in this record version" are not deposited in the current version of the Zenodo record:

| Path | Contents | Supplied by |
|---|---|---|
| `data/tables/train_{5,20,50}cm.parquet` (+ `.csv.gz`), `data/tables/test_{5,20,50}cm.parquet` (+ `.csv.gz`) | Compact observation/predictor tables (77 columns) | not in this record version |
| `data/all_{5,20,50}cm.parquet` | Canonical tables with provenance fields and frozen hashes | not in this record version |
| `data/cascade/{20,50}cm/outer{0..4}/{train,oof,test}.parquet` + `provenance.json` | Fold-specific deep-training inputs with nested surface predictions | not in this record version |
| `data/ancillary/era5land_precip_daily_all_stations.parquet`, `data/ancillary/precip_daily_by_site.parquet` | Daily ERA5-Land rainfall by station and by site | not in this record version |
| `models/5cm/ma_seed42/`, `models/20cm/ma_seed42/`, `models/50cm/ma_floor10_seed42/` (`outer{0..4}/model.pt`, `preprocessor.joblib`) | Product reference weights and training-only transforms (inference) | `OASM10_models_reference.zip` |
| `models/upstream_nested/outer{k}/inner{j}/` and `models/50cm/ma_seed42/` (`model.pt`, `preprocessor.joblib`) | Nested upstream surface fits for deep-training inputs; 50 cm selected-duration anchor | `OASM10_models_training_extras.zip` |
| `oof_predictions.parquet`, `test_predictions.parquet`, `cascade_training_predictions.parquet` of the runs above | Reference, anchor and upstream prediction tables | not in this record version |
| other `models/{5,20,50}cm/<run>/` directories | Weights and predictions of the comparison runs | not in this record version |
| `models/spatial5cm/` | Spatial-block diagnostic fits | not in this record version |
| `training_strategy_review/models/`, `depth50_input_factorial/models/` | Alternative-policy and input-factorial checkpoints and predictions | not in this record version |
| `validation/map_pipeline/regions/`, `validation/map_pipeline/checks/`, `validation/era5_site_pieces/` | Cached feature grids, station-pixel extractions and ERA5-Land pieces | not in this record version |

`ZENODO_MANIFEST.json` in the record lists every file with its SHA-256 digest and archive; `MANIFEST.json` here covers the repository files.
