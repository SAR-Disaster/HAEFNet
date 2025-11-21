import os
import yaml
import torch
import datetime
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from argparse import ArgumentParser
from tqdm import tqdm

from utils.multimodal_dataset import MultiModalRSDataset
from utils.augmentations import get_traditional_train_augmentation, get_traditional_val_augmentation
from utils.losses import build_loss
from models.builder import build_haefnet
from utils.optimizer import PolyWarmupAdamW, CosineAnnealingWarmupAdamW
from utils.helpers import print_log
from utils.meter import AverageMeter, confusion_matrix, getScores

import warnings
import logging

warnings.filterwarnings("ignore")


def parse_args():

    parser = ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/experiment.yml",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        default=False,
    )
    return parser.parse_args()


def setup_device(args):
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        args.gpu = 0
    else:
        args.gpu = None
    return args


def create_dataloader(config, is_train=True):

    try:
        modalities = [k[4:] for k, v in config["modalities"].items() if v and k.startswith("use_")]

        norm_config = config["model"]["modality_norms"]

        root_dir = config["data"]["root_dir"]
        file_list = os.path.join(root_dir, config["data"]["train_list"] if is_train else config["data"]["val_list"])

        target_size = None
        if "target_size" in config["data"]:
            ts = config["data"]["target_size"]
            if isinstance(ts, (list, tuple)) and len(ts) == 2:
                target_size = (int(ts[0]), int(ts[1]))

        dataset = MultiModalRSDataset(
            root_dir=root_dir,
            file_list=file_list,
            modalities=modalities,
            transform=(
                get_traditional_train_augmentation(norm_config)
                if is_train
                else get_traditional_val_augmentation(norm_config)
            ),
            stage="train" if is_train else "val",
            target_size=target_size,
        )

        if len(dataset) == 0:
            raise RuntimeError("Dataset is empty!")

        sampler = None

        dataloader = DataLoader(
            dataset,
            batch_size=config["training"]["batch_size"],
            shuffle=(sampler is None and is_train),
            num_workers=config["training"]["num_workers"],
            pin_memory=True,
            sampler=sampler,
        )

        return dataloader, sampler
    except Exception as e:
        print(f"Error creating dataloader: {str(e)}")
        raise


def train_epoch(
    model,
    dataloader,
    optimizer,
    criterion,
    epoch,
    print_freq=10,
    lambda_head=0.0,
    lambda_modal=0.0,
):

    model.train()
    total_loss = 0
    batch_loss = AverageMeter()

    with tqdm(total=len(dataloader)) as pbar:
        for i, (images, labels, meta) in enumerate(dataloader):
            images = [img.cuda() for img in images] if torch.cuda.is_available() else images
            labels = labels.cuda() if torch.cuda.is_available() else labels

            outputs, aux_info = model(images)

            modal_logits = None
            theta = None
            beta = None
            product_logprob = None
            if isinstance(aux_info, dict):
                modal_logits = aux_info.get("modal_logits", None)
                theta = aux_info.get("theta", None)
                beta = aux_info.get("beta", None)
                product_logprob = aux_info.get("product_logprob", None)

            total_loss_value = criterion(
                outputs[0],
                labels,
                modal_logits=modal_logits if lambda_modal > 0 else None,
                theta=theta,
                beta=beta,
                product_logprob=product_logprob,
            )

            if lambda_head > 0 and len(outputs) > 1:
                head_loss = criterion(
                    outputs[1], labels, modal_logits=None, theta=None, beta=None, product_logprob=None
                )
                total_loss_value = total_loss_value + lambda_head * head_loss

            optimizer.zero_grad()
            total_loss_value.backward()
            optimizer.step()

            batch_loss.update(total_loss_value.item())
            total_loss += total_loss_value.item()

            if i % print_freq == 0:
                pbar.set_description(f"Epoch {epoch} Loss: {batch_loss.avg:.4f}")
            pbar.update(1)

    return total_loss / len(dataloader)


class Saver:
    def __init__(self, args, ckpt_dir, best_val=0, condition=lambda x, y: x > y, save_interval=10):

        self.args = args
        self.directory = ckpt_dir
        self.best_val = best_val
        self.condition = condition
        self.save_interval = save_interval

        os.makedirs(self.directory, exist_ok=True)

    def save(self, val_score, state_dict, epoch=None):

        latest_path = os.path.join(self.directory, "model_latest.pth")
        torch.save(state_dict, latest_path)

        if self.condition(val_score, self.best_val):
            best_path = os.path.join(self.directory, "model_best.pth")
            torch.save(state_dict, best_path)
            self.best_val = val_score

            with open(os.path.join(self.directory, "best_score.txt"), "w") as f:
                f.write(f"Best IoU: {self.best_val:.4f}, Epoch: {epoch}")

            print_log(f"Saved new best model with IoU: {self.best_val:.4f}")

        if epoch is not None and (epoch + 1) % self.save_interval == 0:
            periodic_path = os.path.join(self.directory, f"model_epoch_{epoch+1}.pth")
            torch.save(state_dict, periodic_path)
            print_log(f"Saved periodic checkpoint at epoch {epoch}")


def validate(model, dataloader, criterion, epoch=0, lambda_head=0.0, lambda_modal=0.0):

    model.eval()
    total_loss = 0
    batch_loss = AverageMeter()

    conf_mat = np.zeros((2, 2))

    with torch.no_grad():
        for i, (images, labels, meta) in enumerate(dataloader):
            images = [img.cuda() for img in images] if torch.cuda.is_available() else images
            labels = labels.cuda() if torch.cuda.is_available() else labels

            outputs, aux_info = model(images)

            modal_logits = None
            theta = None
            beta = None
            product_logprob = None
            if isinstance(aux_info, dict):
                modal_logits = aux_info.get("modal_logits", None)
                theta = aux_info.get("theta", None)
                beta = aux_info.get("beta", None)
                product_logprob = aux_info.get("product_logprob", None)

            total_loss_value = criterion(
                outputs[0],
                labels,
                modal_logits=modal_logits if lambda_modal > 0 else None,
                theta=theta,
                beta=beta,
                product_logprob=product_logprob,
            )

            if lambda_head > 0 and len(outputs) > 1:
                head_loss = criterion(
                    outputs[1], labels, modal_logits=None, theta=None, beta=None, product_logprob=None
                )
                total_loss_value = total_loss_value + lambda_head * head_loss

            if isinstance(outputs, list):
                ensemble_output = outputs[0]
            else:
                ensemble_output = outputs

            batch_loss.update(total_loss_value.item())
            total_loss += total_loss_value.item()

            ensemble_output = nn.functional.interpolate(
                ensemble_output, size=labels.shape[1:], mode="bilinear", align_corners=False
            )

            try:

                labels_np = labels.cpu().detach().numpy()
                predictions = torch.argmax(ensemble_output, dim=1).cpu().detach().numpy()

                labels_np = labels_np.astype(np.int64)
                predictions = predictions.astype(np.int64)

                batch_conf_mat = confusion_matrix(labels_np.flatten(), predictions.flatten(), 2, ignore_label=255)
                conf_mat += batch_conf_mat

            except Exception as e:
                print(f"Error processing batch {i}: {str(e)}")
                continue

    overall_acc, class_acc, iou = getScores(conf_mat)

    return total_loss / len(dataloader), iou


def getIoUPerClass(confusion_matrix):

    iou_list = []

    for i in range(confusion_matrix.shape[0]):

        true_positive = confusion_matrix[i, i]

        union = np.sum(confusion_matrix[i, :]) + np.sum(confusion_matrix[:, i]) - true_positive

        if union > 0:
            iou = true_positive / union * 100
        else:
            iou = 0

        iou_list.append(iou)

    return iou_list


def main():

    args = parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    current_time = datetime.datetime.now().strftime("%Y%m%d")

    config["training"]["current_time"] = current_time

    with open(args.config, "w") as f:
        yaml.safe_dump(config, f, default_flow_style=False)

    args = setup_device(args)

    model = build_haefnet(config)
    if torch.cuda.is_available():
        model = model.cuda()

    train_loader, train_sampler = create_dataloader(config, is_train=True)
    val_loader, _ = create_dataloader(config, is_train=False)

    param_groups = model.get_param_groups()

    scheduler_type = config["training"].get("scheduler", {}).get("type", "polynomial")

    if scheduler_type.lower() == "cos":

        optimizer = CosineAnnealingWarmupAdamW(
            params=[
                {
                    "params": param_groups[0],
                    "lr": config["training"]["learning_rate"],
                    "weight_decay": config["training"]["weight_decay"],
                },
                {"params": param_groups[1], "lr": config["training"]["learning_rate"], "weight_decay": 0.0},
                {
                    "params": param_groups[2],
                    "lr": config["training"]["learning_rate"] * 10,
                    "weight_decay": config["training"]["weight_decay"],
                },
            ],
            lr=float(config["training"]["learning_rate"]),
            weight_decay=float(config["training"]["weight_decay"]),
            betas=[0.9, 0.999],
            T_max=int(config["training"]["scheduler"].get("T_max", 40000)),
            eta_min=float(config["training"]["scheduler"].get("eta_min", 1e-7)),
            warmup_iterations=int(config["training"]["scheduler"].get("warmup_iterations", 3000)),
            warmup_ratio=float(config["training"]["scheduler"].get("warmup_ratio", 1e-6)),
        )
    else:

        optimizer = PolyWarmupAdamW(
            params=[
                {
                    "params": param_groups[0],
                    "lr": config["training"]["learning_rate"],
                    "weight_decay": config["training"]["weight_decay"],
                },
                {"params": param_groups[1], "lr": config["training"]["learning_rate"], "weight_decay": 0.0},
                {
                    "params": param_groups[2],
                    "lr": config["training"]["learning_rate"] * 10,
                    "weight_decay": config["training"]["weight_decay"],
                },
            ],
            lr=float(config["training"]["learning_rate"]),
            weight_decay=float(config["training"]["weight_decay"]),
            betas=[0.9, 0.999],
            warmup_iter=int(config["training"].get("optimizer_params", {}).get("warmup_iter", 1500)),
            max_iter=int(config["training"].get("optimizer_params", {}).get("max_iter", 40000)),
            warmup_ratio=float(config["training"].get("optimizer_params", {}).get("warmup_ratio", 1e-6)),
            power=float(config["training"].get("optimizer_params", {}).get("power", 1.0)),
        )

    criterion = build_loss(config)

    loss_weights = config["training"].get("loss_weights", {})
    lambda_head = float(loss_weights.get("head", 0.0))
    lambda_modal = float(loss_weights.get("modal", 0.0))

    if lambda_modal <= 0 and getattr(model, "use_evidential_fusion", False):
        lambda_modal = 1.0

    saver = Saver(
        args=config,
        ckpt_dir=os.path.join(
            config["training"]["ckpt_dir"], config["training"]["name"] + "_" + config["training"]["current_time"]
        ),
        best_val=0,
        condition=lambda x, y: x > y,
        save_interval=10,
    )

    for epoch in range(config["training"]["num_epochs"]):

        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, epoch, lambda_head=lambda_head, lambda_modal=lambda_modal
        )

        if (epoch + 1) % config["training"]["save_interval"] == 0:
            val_loss, val_iou = validate(
                model, val_loader, criterion, epoch, lambda_head=lambda_head, lambda_modal=lambda_modal
            )
            print_log(
                f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}, Val IoU = {val_iou:.4f}%"
            )

            saver.save(
                val_iou,
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                },
                epoch=epoch,
            )

    print_log(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}, Val IoU = {val_iou:.4f}%")

    saver.save(
        val_iou,
        {
            "model": model.state_dict(),
            "epoch": epoch,
        },
        epoch=epoch,
    )


if __name__ == "__main__":
    main()
