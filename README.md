# Reliability-Aware Multimodal Landslide Detection (HAEFNet)

Official PyTorch implementation of “Reliability Aware Multimodal Detection of Active Landslides via Evidence Theoretic Fusion of InSAR, Optical, and DEM Data.” The framework fuses InSAR, optical, and DEM cues for pixel-wise landslide mapping with reliability indicators.

## Repository Layout
- Entry points: `train.py` and `predict.py`.
- Configs: `configs/exp_haefnet.yml` (set data paths and modality flags). Update `root_dir: ./data/multimodal-landslide-dataset`.
- Modules: `models/` (fusion blocks, encoders, heads) and `utils/` (dataset loader, augmentations, losses, optimizer, logging, checkpoints).
- Assets: place datasets in `data/`, checkpoints in `pretrained/`, and experiment outputs in `work_dir/` or the `--output_dir` you choose.

## Requirements and Installation
1) Install Miniconda/Anaconda with CUDA-enabled PyTorch support.
2) Use the provided `environment.yml` (repo root) to reproduce dependencies:  
   ```bash
   conda env create -f environment.yml
   conda activate haefnet
   ```
   This installs Python ≥3.8, PyTorch 1.12+, timm, rasterio, GDAL bindings, and other utilities pinned for reproducibility.
3) Ensure `gdal`/`rasterio` can read your GeoTIFFs; set `CUDA_VISIBLE_DEVICES` if needed.

## Data Preparation
- Dataset + pretrained backbones + trained checkpoints:  
  Link: https://pan.baidu.com/s/1a7MxWXdbgVcjf2GQ36ic5w?pwd=1234  
  Extract the package, then place the dataset at `data/multimodal-landslide-dataset`, Swin weights under `pretrained/`, and any provided HAEFNet checkpoints under `checkpoints/`. Keep the folder names referenced by `data.train_list` / `data.val_list` in `configs/exp_haefnet.yml`.
- Set `root_dir: /home/.../data/multimodal-landslide-dataset` in the config. Use relative paths for portability.

## InSAR Preprocessing (ASF + MintPy)
1) Discover and download SAR stacks from ASF DAAC (https://search.asf.alaska.edu/#/) for your AOI.  
2) Process interferograms and time series with MintPy (https://github.com/insarlab/MintPy): unwrap, geocode, and export LOS displacement/coherence to GeoTIFFs.  
3) Resample outputs to the optical/DEM grid used by the dataset; store them under the InSAR modality folders referenced in your split lists. Record processing parameters for reproducibility.

## Training
- Edit `configs/exp_haefnet.yml` (paths, batch size, learning rate, modality toggles).
- Launch training:  
  ```bash
  python train.py --config configs/exp_haefnet.yml --name exp_haefnet
  ```
  Checkpoints are saved under `checkpoints/<timestamp>_<name>/` by default.

## Inference & Visualization
- Run full prediction/diagnostics:  
  ```bash
  python predict.py --config configs/exp_haefnet.yml \
    --model_path <your_ckpt> \
    --output_dir work_dir/demo_pred
  ```
  Generates per-pixel maps, overlays on RGB, probability maps, and uncertainty visualizations.

## Code Availability Statement
The source code for this study is openly available at https://github.com/SAR-Disaster/HAEFNet under the terms of this repository. Data and weights: BaiduPan package above (pwd: 1234). InSAR preprocessing relies on ASF data access and the open-source MintPy toolkit (links above).
