# External inputs and availability

The original repository identifies [Zenodo record 22779714](https://doi.org/10.5281/zenodo.22779714)
as the model archive and describes access as restricted during peer review.
The contents below come from that repository's release inventory; current
archive download access has not been independently confirmed.
This reduced tree is not byte-identical to `OASM10_repository.zip`.

## Released-model inference

`OASM10_models_reference.zip` is documented to contain `model.pt` and
`preprocessor.joblib` for every fold (`outer0` through `outer4`) under:

```text
models/5cm/ma_seed42/
models/20cm/ma_seed42/
models/50cm/ma_floor10_seed42/
```

This code tree keeps the original fold manifests and `product_reference.json`.
Place each weight and preprocessor alongside its fold manifest. Archives may
have an `OASM10/` wrapper: merge its contents into the repository root.
Do not overwrite this reduced tree with the old repository archive's code.

Map extraction also needs `Beck_KG_V1_present_0p0083.tif`. Download the exact
[historical raster](https://raw.githubusercontent.com/RSNuo/OASM-10/e11551c5d061790ba6f8a24d1904af774be2ab9c/data/ancillary/Beck_KG_V1_present_0p0083.tif)
and save it as `data/ancillary/Beck_KG_V1_present_0p0083.tif`, or supply its location
via `climate_raster` / the notebook's `CLIMATE_RASTER`. Its SHA-256 is:

```text
a343fdfdb4c4a427c774bedd75ae4f2a6420c3f465de7d51892e8712dfaa2d27
```

The cleanup deliverables also include it in `OASM-10-climate-data.zip`.
It is runtime data, not source code. The original Git commit must stay reachable
for the historical download link to work.

## Training from scratch

The required canonical inputs are:

```text
data/all_5cm.parquet
data/all_20cm.parquet
data/all_50cm.parquet
```

**These tables were explicitly marked "not in this record version" in the
original inventory.** The public tree alone cannot reproduce training.
The data owner needs to provide these exact tables and their checksums.
The original raw-data preparation pipeline is also absent; training starts
from canonical tables.

Each table needs raw predictors from `data/feature_schema.json`, depth-specific
SoilGrids fields for upstream pairing, and the metadata columns `record_id`,
`source_station`, `physical_site_id`, `target_time`, `y`, `split`, `outer_fold`.
Preserve original row order, values and site assignments in `data/shared_split.csv`
and `data/site_registry.csv`. Split values are `development` and `test`;
development folds are 0 through 4. Targets are observed moisture in m3/m3.
Do not substitute synthetic targets or independently split each depth.

The reduced training command regenerates nested upstream models and cascade
tables. Old cascade files, ancillary station soil/precipitation tables,
prediction tables and `OASM10_models_training_extras.zip` are unnecessary when
canonical inputs are available and training is run from scratch.

## Generated files

Keep checkpoints, datasets, cached grids and GeoTIFFs in versioned external
archives with checksums. These files are ignored by Git. Keep old experiment
protocols and hash chains in the original release; they do not verify this
modified code tree.
