# Reliability-Aware Multimodal Landslide Detection (HAEFNet)

Official code release for “Reliability Aware Multimodal Detection of Active Landslides via Evidence Theoretic Fusion of InSAR, Optical, and DEM Data.” The repo implements a single-GPU training/inference pipeline that fuses heterogeneous remote sensing modalities, produces pixel-wise landslide maps, and reports aleatoric/epistemic cues for reliability-aware decisions.

## Repository Layout
- Entry points: `train.py` (training + optional `--evaluate`) and `predict.py` (inference/visualization).
- Configs: `configs/exp_haefnet.yml` (edit data roots, modality toggles, optimization, logging).
- Models: `models/` (`builder.py`, `haef_net.py`, `gem.py`, `mrg.py`, `scca.py`, `haef_encoder.py`).
- Data utilities: `utils/` (`multimodal_dataset.py`, `augmentations.py`, `losses.py`, `optimizer.py`, `meter.py`, `checkpoint.py`, `helpers.py`).
- Artifacts: checkpoints in `checkpoints/`, pretrained weights in `pretrained/`, logs in `logs/`.

## Requirements
- python>=3.8
- torch>=1.12 (CUDA build recommended for training)
- timm>=0.6
- numpy>=1.22
- pyyaml>=6.0
- rasterio>=1.3
- matplotlib>=3.5
- Pillow>=9.0
- opencv-python>=4.5
- tqdm>=4.64
- Data layout under `data/`:

## Training & Evaluation
- Edit `configs/exp_haefnet.yml`:
  - Set `data.root_dir`, `data.train_list`, `data.val_list`.
  - Enable modalities via `modalities.use_*` flags; adjust `model.modality_norms` if stats differ.
  - Tune `training.batch_size`, `learning_rate`, `num_epochs`, `lambda_head`, `lambda_modal`, `target_size` as needed.
- Launch training:
  ```bash
  python train.py --config configs/exp_haefnet.yml
  ```
  Checkpoints are written to `checkpoints/<timestamp>_<name>/` with `model_latest.pth`, `model_best.pth`, and periodic snapshots.

## Inference & Visualization
- Run prediction/analysis (saves `pred_vis/`, `overlay_vis/`, `prob_vis/`, `uncertainty_vis/`, modality contribution diagnostics when available):
  ```bash
  python predict.py --config configs/exp_haefnet.yml --model_path <path_to_ckpt> --output_dir work_dir/demo_pred
  ```
- The script auto-selects modalities from config and produces overlays using RGB backgrounds when present.

