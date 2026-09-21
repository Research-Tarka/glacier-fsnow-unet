# glacier-fsnow-unet

Data pipeline for building a 1984-2025 glacier surface time series and
end-of-season snow-fraction ($F_{snow}$) record from optical satellite
imagery, developed for the preprint *"Benchmarked and leakage-validated
deep-learning glacier segmentation: a 42-year, five-sensor snow-fraction
record for western North America"*
([SSRN](https://ssrn.com/abstract=7418104), DOI:
[10.2139/ssrn.7418104](http://dx.doi.org/10.2139/ssrn.7418104)).

Given a set of glacier outlines (RGI), the pipeline downloads a DEM and every
available Landsat 5/7/8/9 and Sentinel-2 scene over the study period, trains
a 4-class U-Net (Snow / Ice / Cloud / Other) to segment each scene, and
derives from it, per glacier and per year, the reference glacier surface and
its snow fraction. It also collects a full set of temporal and static
climate/environmental covariates over the same period. It is built around
the RGI glacier inventory, so it applies to any RGI region, not only the one
used in the paper.

## Companion tools

This pipeline does not include glacier delineation/annotation or the
downstream visual exploration of its outputs; those live in two separate
tools:

- [MaskForge](https://github.com/Research-Tarka/maskforge) -- standalone,
  general-purpose successor to the internal prototype used to build the
  training/validation annotation corpus for this paper; MaskForge itself was
  not used to produce those annotations.
- [GlacierScope](https://github.com/Research-Tarka/GlacierScope) -- per-glacier
  visualization of the pipeline's outputs, including analyses beyond what is
  reported in the paper.

## Install

```bash
conda env create -f environment.yml
conda activate glacier-fsnow-unet
pip install -r requirements-torch.txt
pip install -e .
```

Then copy the configuration and environment templates and fill in the
placeholders:

```bash
cp configs/config.example.yaml configs/config.yaml
cp .env.example .env
```

## Configuration

Everything the pipeline needs is read from one YAML config
(`configs/config.yaml`), organized by stage (`isolated_glacier`, `dem`,
`scene_download`, `training`, `unet_interpretation`, `inference`,
`features`, ...). `configs/config.example.yaml` documents every key inline;
the non-path hyperparameters in it are the ones used for the paper's
results and should be kept unless you are deliberately re-tuning.

Two kinds of placeholders appear in the config:

- `CHANGE_ME` -- must be filled in directly (paths specific to your machine).
- `${VAR}` -- resolved from your shell environment or from a `.env` file at
  the repo root (see `.env.example`). This keeps machine-specific roots and
  credentials (data/output/model directories, Earth Engine project id) out
  of the tracked config file.

Point any script at a non-default config with `--config path/to/config.yaml`
or the `GLACIER_FSNOW_CONFIG` environment variable.

## Running the pipeline

Each stage is an independent script under `scripts/`, numbered in the order
they are meant to run:

```bash
python scripts/01_select_glaciers.py --config configs/config.yaml
python scripts/02_download_dem.py --config configs/config.yaml
# ... and so on through stage 13
```

Run them one at a time so you can inspect each stage's output before moving
on, re-run a single stage in isolation, or fan out later stages (e.g. scene
download, feature download) across parallel splits without depending on an
orchestrator.

| # | Script | Stage | Needs |
|---|---|---|---|
| 01 | `01_select_glaciers.py` | Select isolated glaciers from RGI 7.0 -> glacier registry | RGI shapefiles |
| 02 | `02_download_dem.py` | Per-glacier DEM at 10/15/30 m (ArcticDEM, Copernicus GLO-30 fallback) | network (STAC) |
| 03 | `03_split_glaciers.py` | Distribute glaciers across N processing batches | -- |
| 04 | `04_download_scenes.py` | Landsat 5/7/8/9 + Sentinel-2 scenes over the ablation window | Google Earth Engine |
| 05a | `05a_build_train_cache_from_zenodo.py` | Rebuild the per-scene training cache (11 spectral indices + labels) from the published `UNet_Train_Data` corpus | annotated corpus (TOA/Mask GeoTIFFs) |
| 05 | `05_train_model.py` | Train the 4-class U-Net (single run / bootstrap / cross-validation / HPO) | GPU, annotated corpus |
| 06 | `06_interpret_model.py` | Ambiguity calibration, permutation importance, SHAP, Grad-CAM | trained model, corpus |
| 07 | `07_run_inference.py` | Sliding-window inference -> per-scene class maps | trained model |
| 08 | `08_annual_state_map.py` | Annual worst-state composite per glacier-year | stage 7 output |
| 09 | `09_build_vgs.py` | Visible Glacier Surface (VGS) reference mask | stage 8 output |
| 10 | `10_compute_stats.py` | $F_{snow}$ and the delivered per-glacier statistics tables | stage 9 output |
| 11 | `11_download_features.py` | Per-glacier climate/environmental covariates (5 sources) | Google Earth Engine |
| 12 | `12_merge_features.py` | Merge climate sources + climatology into delivered feature tables | stage 11 output |
| 13 | `13_download_static_sources.py` | Static per-glacier sources (RGI attributes, Koppen-Geiger, WorldClim, permafrost, coastline) | network |

Stage 6 is analyst-run, not part of the unattended data pipeline: nothing
downstream reads its outputs except `ambiguity_threshold.json`, which stage 7
picks up automatically once placed beside the checkpoint (opt in with
`--install-ambiguity-threshold`). Its four analyses are independently
switchable under `unet_interpretation` in the config.

Stage 05a exists because the training cache stage 05 reads (`ia_cache_all/
scene_features.npz` per scene: 11 precomputed spectral-index channels, the
remapped labels, and scene identity) is not itself a published artifact --
only the raw `TOA*.tif`/`Mask_30m.tif` pairs are. Run it once, with no
arguments beyond `--config`, to rebuild that cache from the published corpus
before stage 05:

```bash
python scripts/05a_build_train_cache_from_zenodo.py --config configs/config.yaml
python scripts/05_train_model.py --config configs/config.yaml
```

It reads `paths.zenodo_train_data_root` (the published corpus) and writes into
`paths.train_root` (what stage 05 reads by default), so both stages need no
flags beyond `--config` once `configs/config.yaml` is filled in. It only
rebuilds scenes whose cache is missing or stale, so re-running it after adding
more annotated scenes is cheap.

## Baseline comparisons

`scripts/baselines/` holds a separate, optional suite of non-U-Net model
comparisons (NDSI/NDVI threshold, Random Forest, DeepLabv3, SegFormer) run
against the exact same glacier-level split, features, and metrics as the
published U-Net, plus a generic statistical comparator. `05_train_model.py`
never reads this suite and these scripts never modify the published model.

| # | Script | Comparison | Needs |
|---|---|---|---|
| 14 | `14_ndsi_baseline.py` | NDSI/NDVI threshold tree (simplest possible classifier) | annotated corpus |
| 15 | `15_rf_baseline.py` | Random Forest on the same 11 spectral indices (grid or Optuna search) | annotated corpus |
| 16 | `16_deeplab_baseline.py` | DeepLabv3 (ResNet50, ASPP tuned for the 48x48 patch grid) | GPU, annotated corpus |
| 17 | `17_segformer_baseline.py` | SegFormer (3-stage MiT encoder, patch-size scale check) | GPU, annotated corpus |
| 18 | `18_compare_models.py` | Welch/paired t-test + bootstrap CI, between any two score series or from a checkpoint family | one or more result files |
| 19 | `19_uncertainty_analysis.py` | Predictive-entropy AUROC for any trained checkpoint (U-Net or a baseline) | trained checkpoint |

Each of 14-17 runs with no arguments (reads `baselines.<name>` from
`configs/config.yaml`); 16 and 17 default to running an Optuna search then a
multi-seed confirmation of its winner in one invocation. See each script's
module docstring for its exact protocol and `--help` for every override.

## Citation

If you use this pipeline, please cite the preprint:

> Tarka, Maxime and Baraër, Michel and Aubry-Wake, Caroline, Benchmarked and
> leakage-validated deep-learning glacier segmentation: a 42-year, five-sensor
> snow-fraction record for western North America (September 21, 2026).
> Available at SSRN: https://ssrn.com/abstract=7418104 or
> http://dx.doi.org/10.2139/ssrn.7418104

A snapshot of this repository and its released model checkpoints/data is
archived on Zenodo:

> https://doi.org/10.5281/zenodo.22311127

## License

MIT -- see [LICENSE](LICENSE).
