import argparse
import datetime
import os
import yaml
import torch
import numpy as np
import rasterio
from rasterio.transform import from_origin
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.helpers import resolve_experiment_tag

from models.builder import build_haefnet
from utils.multimodal_dataset import MultiModalRSDataset
from utils.augmentations import get_traditional_val_augmentation


def load_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    config.setdefault("prediction", {})
    return config


def resolve_paths(config, model_path, output_dir):
    run_tag = resolve_experiment_tag(config["training"])
    if model_path is None:
        model_path = os.path.join(config["training"]["ckpt_dir"], run_tag, "model_best.pth")

    if output_dir is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join("work_dir", "binary_pred", f"{run_tag}_{timestamp}")

    os.makedirs(output_dir, exist_ok=True)
    return model_path, output_dir


def resolve_file_list(config, file_list_arg, split):
    if file_list_arg:
        return file_list_arg

    split_key = f"{split}_list"
    if "data" not in config or split_key not in config["data"]:
        raise ValueError(f"找不到数据列表配置 data.{split_key}")

    return os.path.join(config["data"]["root_dir"], config["data"][split_key])


def build_loader(config, file_list, modalities):
    dataset = MultiModalRSDataset(
        root_dir=config["data"]["root_dir"],
        file_list=file_list,
        modalities=modalities,
        transform=get_traditional_val_augmentation(config["model"]["modality_norms"], require_label=False),
        stage="test",
        require_label=False,
    )

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config["training"]["num_workers"],
        pin_memory=True,
    )
    return loader


def forward_to_mask(model, images, has_analysis, threshold):
    if has_analysis:
        report = model.analyze_modalities(images, foreground_class=1, compute_loo=False)
        logits = report["final_logits"]
    else:
        logits = model(images)

    if isinstance(logits, tuple):
        logits = logits[0]
    if isinstance(logits, list):
        logits = logits[-1]

    if logits.shape[1] == 1:
        prob = torch.sigmoid(logits)
        pred = (prob >= threshold).long()
    else:
        pred = logits.argmax(dim=1)

    return pred


def save_mask(mask, out_path, ref_path=None):
    mask_np = (mask.astype(np.uint8) > 0).astype(np.uint8) * 255
    height, width = mask_np.shape

    profile = None
    if ref_path and os.path.exists(ref_path):
        try:
            with rasterio.open(ref_path) as src:
                profile = src.profile
        except Exception:
            profile = None

    if profile is None:
        profile = {
            "driver": "GTiff",
            "dtype": rasterio.uint8,
            "count": 1,
            "width": width,
            "height": height,
            "crs": None,
            "transform": from_origin(0, 0, 1, 1),
            "compress": "lzw",
            "nodata": 0,
        }
    else:
        profile.update(
            driver="GTiff",
            dtype=rasterio.uint8,
            count=1,
            width=width,
            height=height,
            compress="lzw",
            nodata=0,
        )

    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(mask_np, 1)


def predict_binary(config_path, model_path=None, output_dir=None, file_list=None, split="val", threshold=0.5):
    config = load_config(config_path)
    model_path, output_dir = resolve_paths(config, model_path, output_dir)

    modalities = [k[4:] for k, v in config["modalities"].items() if v and k.startswith("use_")]
    file_list_path = resolve_file_list(config, file_list, split)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    print(f"模型路径: {model_path}")
    print(f"数据列表: {file_list_path}")
    print(f"输出目录: {output_dir}")

    model = build_haefnet(config).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    has_analysis = hasattr(model, "analyze_modalities")

    loader = build_loader(config, file_list_path, modalities)
    if len(loader.dataset) == 0:
        raise RuntimeError(f"数据列表 {file_list_path} 无有效样本，请检查模态文件与路径。")

    mask_dir = os.path.join(output_dir, "pred_masks_tif")
    os.makedirs(mask_dir, exist_ok=True)

    saved = 0
    with torch.no_grad():
        for images, _, meta in tqdm(loader, desc="推理中"):
            file_id = meta["file_id"]
            if isinstance(file_id, list):
                file_id = file_id[0]

            ref_path = meta["modalities"][modalities[0]]
            if isinstance(ref_path, list):
                ref_path = ref_path[0]

            images_cuda = [img.to(device) for img in images]
            pred = forward_to_mask(model, images_cuda, has_analysis, threshold)

            mask_np = pred.squeeze(0).cpu().numpy()
            out_path = os.path.join(mask_dir, f"{file_id}.tif")
            save_mask(mask_np, out_path, ref_path=ref_path)
            saved += 1

    print(f"推理完成，共保存 {saved} 个二值掩码。")
    print(f"掩码TIF目录: {mask_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="简化推理，仅输出二值分割结果")
    parser.add_argument("--config", required=True, type=str, help="YAML配置文件路径")
    parser.add_argument("--model", type=str, default=None, help="模型权重路径，默认读取config.prediction.model_path")
    parser.add_argument(
        "--output", type=str, default=None, help="输出目录，默认在config.prediction.output_dir下生成带时间戳的文件夹"
    )
    parser.add_argument("--file_list", type=str, default=None, help="自定义推理列表txt，优先级最高")
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["train", "val", "test"],
        help="使用config.data.<split>_list 作为推理列表",
    )
    parser.add_argument("--threshold", type=float, default=0.5, help="当输出为单通道时的阈值")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    predict_binary(
        config_path=args.config,
        model_path=args.model,
        output_dir=args.output,
        file_list=args.file_list,
        split=args.split,
        threshold=args.threshold,
    )
