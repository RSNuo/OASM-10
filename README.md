# OASM-10: on-demand soil moisture at 5, 20 and 50 cm on a nominal 10 m grid

This repository contains the code, metadata, evaluation records and example outputs of the OASM-10 reference product.
The observation tables, fitted model weights and cascade inputs are large and are distributed through the accompanying
Zenodo archive (see `ZENODO_CONTENTS.md`); extract that archive into this folder and every path below resolves.

Archived release: this code tree, the model manifests and the fitted product reference weights are deposited at Zenodo, https://doi.org/10.5281/zenodo.22779714 (access restricted while the accompanying Data Descriptor is under peer review; opened on publication).

Frozen experiment records (protocol, freeze and verification files under `training_strategy_review/`,
`depth50_input_factorial/` and the per-fold model manifests) are kept byte-identical to the verified originals so that
their SHA-256 chains remain checkable; inside them the implementation is labelled by its working name `v6_site_isolated`,
which denotes exactly the code and data released here.

## Layout

| Path | Contents |
|---|---|
| `oasm/` | Python package: feature definitions, Google Earth Engine extraction, preprocessing, models, training, inference |
| `notebooks/` | `prediction_multidepth.ipynb` and `prediction_5cm.ipynb`: region/time request, GEE feature extraction and fold-paired prediction; `training_5cm.ipynb`, `training_20cm.ipynb`, `training_50cm.ipynb`: depth-explicit training entry points |
| `run_campaign.py` | Reference fits, nested upstream inputs and all matched comparison fits (`--stage all`) |
| `analyze_results.py` | Evaluation of the fitted runs (site-grouped statistics, paired bootstrap); also imported by the frozen record scripts |
| `data/` | Site registry, station aliases, shared split and fold assignments, field dictionary, schema, counts; small ancillary layers |
| `models/` | `product_reference.json` (which run serves each depth) and every fitted run's manifest and metrics (weights on Zenodo) |
| `product/example_geotiffs/` | Eight regional examples: three moisture layers, a true-colour reflectance raster, quality table and manifest |
| `analysis/`, `validation/` | Result JSON files, per-site tables, matched benchmark records and map-pathway checks |
| `training_strategy_review/`, `depth50_input_factorial/`, `reference_50cm_floor10/` | Frozen protocols, results and verification of the fitting-policy experiment, the 50 cm input factorial and the 50 cm reference decision |
| `verification/` | Integrity and self-check records (`training_input_integrity.json`, `verification_*.json`) and the scripts that produced them |
| `MANIFEST.json`, `ZENODO_CONTENTS.md` | SHA-256 digest of every repository file; what the Zenodo record adds |

Directory paths cited in the data descriptor are relative to this folder. The preprocessing pipeline that built the
canonical tables from the ISMN and Google Earth Engine archives, the figure scripts and the external-product comparison
scripts are not part of this repository; the tables and comparison records they produced are included under `data/`,
`analysis/` and `validation/`.

## Product reference

`models/product_reference.json` maps each depth to its released run: `5cm/ma_seed42`, `20cm/ma_seed42` and `50cm/ma_floor10_seed42`.
For every outer fold k, the 5 cm model of fold k predicts the surface input for the 20 and 50 cm models of fold k; the five
depth-specific outputs are averaged. `oasm.inference.predict_product` implements this route and applies each fold's stored
training-only transformations and imputation values. Predictions are not clipped; validity and missing-input flags accompany them.

## Running

1. Create the environment from `requirements.txt` (versions in `runtime.json`; PyTorch with CUDA is optional).
2. Extract the Zenodo archive into this folder (adds `data/tables/`, `data/all_*.parquet`, `data/cascade/`, the precipitation tables under `data/ancillary/` and the fitted weights under `models/`).
3. Authenticate Google Earth Engine with your own account for map requests.
4. Open `notebooks/prediction_multidepth.ipynb`, set the bounds and target time, and run. Outputs are written to `outputs/<date>/` with a quality table and manifest.
5. To retrain, run the training notebooks under `notebooks/` or `python run_campaign.py --stage all`.

Inputs are pre-standardization values; rainfall descriptors are stored as ln(1 + P). Do not apply any global scaler.
Landsat bands are empirical DN x 1e-4 inputs, and the two derived vegetation contrasts are Landsat DN differences, not
calibrated reflectance indices. `data/data_dictionary.csv` gives units, keys and missing-value meanings.

## Evaluation conventions

Physical sites (co-located sensors grouped within 30 m) are the unit of exclusion. Sites in the test set are excluded at
every depth from preprocessing, gradient fitting and epoch selection; the same test sites had been used in earlier exploratory work and
this is disclosed as such. Selection of fold, ensemble, policy and input configurations uses training-set out-of-fold results only.
The nominal 10 m grid is a sampling lattice; it is not a validated effective resolution.

## Citation and licence

Companion manuscripts: a Scientific Data descriptor and an IEEE Transactions on Geoscience and Remote Sensing methods study
(Xu, Daccache and Ahmadi, under review). Licence: see `LICENSE`.
