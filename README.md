# OASM-10: training and inference

OASM-10 estimates soil moisture at 5, 20 and 50 cm on a nominal 10 m grid.
This repository keeps the training and inference code, feature definitions, site
splits and the 15 released fold manifests. Generated maps, analysis reports,
comparison experiments and checkpoints are distributed separately.

**Data availability:** the documented Zenodo version does not include the three
canonical training tables. Cloning this repository or downloading only the
weights is insufficient to retrain. See [ZENODO_CONTENTS.md](ZENODO_CONTENTS.md)
for exact inputs, archive availability and the climate raster download.

## Contents

| Path | Purpose |
|---|---|
| `oasm/` | Features, preprocessing, models, training, inference and Earth Engine extraction |
| `run_campaign.py` | Seed-42 reference training, including nested surface inputs |
| `notebooks/training_*cm.ipynb` | Per-depth training entry points |
| `notebooks/prediction_*.ipynb` | Surface-only and multi-depth map inference |
| `data/` | Feature schema, field dictionary, site registry, aliases and shared split |
| `models/` | Product run mapping and released fold manifests; weights are external |
| `runtime.json` | Original release environment, including archived analysis dependencies |

## Installation

Use a dedicated Python 3.12 or 3.13 environment; the original release used 3.13.2.
Install PyTorch for your hardware, then the remaining dependencies:

```sh
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

For the original CUDA 11.8 build, replace the first command's index URL with
`https://download.pytorch.org/whl/cu118`. These follow the
[official PyTorch 2.7.1 instructions](https://pytorch.org/get-started/previous-versions/).
`requirements.txt` accepts either build. For notebooks, also install `jupyterlab`
and `ipykernel`, then start `jupyter lab` in this repository.

## Train the reference product

Provide `data/all_5cm.parquet`, `data/all_20cm.parquet` and
`data/all_50cm.parquet` with the original physical-site IDs and fold assignments.
Preserve raw predictors and `development` / `test` split labels. Run:

```sh
python run_campaign.py --stage all
```

This fits five surface folds, 25 nested upstream models for deep-training inputs,
five 20 cm folds and five 50 cm folds. It does not run the archived architecture,
seed or policy comparisons. Internal site holdout selects the epoch count;
50 cm refits for `max(selected_epochs, 10)`, matching the released policy.
The learning-rate schedule retains its 150-epoch horizon by default.
Changing `--epochs` changes the experiment.

New weights, transforms, manifests and predictions go to
`outputs/retrained/models/`; generated cascade tables go to
`outputs/retrained/cascade/`. Released models under `models/` stay separate.
Completed compatible folds can be reused. Changed inputs or configuration
require a fresh output directory:

```sh
python run_campaign.py --data-dir data --output-dir outputs/my_run --stage all
```

Relative paths in these options resolve from the repository, even when the
command is launched elsewhere. Individual stages are `surface`, `nested`,
`depth --depth 20`, `depth --depth 50` and `summarize`; run them in that order
with the same output directory. Surface-only training needs only the 5 cm table.
Nested training needs all three tables to preserve shared site exclusions.

## Predict

Extract released weights into `models/` as described in
[ZENODO_CONTENTS.md](ZENODO_CONTENTS.md). For newly trained models, use
`outputs/retrained/models/` instead.

Map extraction also needs the Beck climate raster under `data/ancillary/` and
your own Earth Engine credentials and enabled Cloud project:

```sh
earthengine authenticate
```

Open `notebooks/prediction_multidepth.ipynb` or `prediction_5cm.ipynb` and set
`MODEL_ROOT`, `EE_PROJECT`, `CLIMATE_RASTER`, `BOUNDS` and `TARGET_TIME`.
See Google's [authentication guide](https://developers.google.com/earth-engine/guides/auth).
Results go to `outputs/<date>/`. No personal project ID or machine-specific
folder is embedded in the code. Relative `climate_raster` arguments resolve
from the repository.

Offline prediction requires a raw feature table containing all predictors;
Earth Engine and the climate raster are then unnecessary:

```python
from pathlib import Path
import pandas as pd
from oasm.inference import predict_product

# Run from the repository root; supply your raw feature table.
frame = pd.read_parquet("data/features.parquet")
prediction = predict_product(frame, Path("models"), depths=(5, 20, 50))
```

`models/product_reference.json` selects `5cm/ma_seed42`, `20cm/ma_seed42` and
`50cm/ma_floor10_seed42`. Each deep fold receives its paired surface prediction;
the five outputs are averaged. Predictions are unclipped, with validity,
missing-input and out-of-range flags.

## Reproduction conventions

Physical sites are the exclusion unit. Test sites must not enter preprocessing,
epoch selection or gradient training. These test sites also appeared in earlier
exploratory work; the released policy is not an untouched prospective test.
Features use pre-standardization values and rainfall `ln(1 + P)`; do not fit a
global scaler. Landsat features follow the empirical DN-based definitions in
`data/data_dictionary.csv`. Optical selection is retrospective within +/-14 days.
Nominal grid spacing is not validated effective resolution. Numerical results
can vary with hardware and library builds.

Historical experiment protocols and verification records belong to the original
release. Its old `MANIFEST.json` cannot verify this edited tree and is omitted.
Original sources and metadata remain in Git history and the archived release.
A license file was absent in the audited repository; this cleanup assigns no
new license.
