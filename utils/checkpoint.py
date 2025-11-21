"""
Lightweight checkpoint helpers without external dependencies.
Used for loading Swin weights (supports DDP prefixes and positional bias resize).
"""

from typing import Any, Dict
import os

import torch
import torch.nn.functional as F


def _strip_prefix(state_dict: Dict[str, Any], prefix: str) -> Dict[str, Any]:
    if not prefix:
        return state_dict
    if all(k.startswith(prefix) for k in state_dict.keys()):
        return {k[len(prefix) :]: v for k, v in state_dict.items()}
    return state_dict


def _reshape_absolute_pos_embed(state_dict: Dict[str, Any], model, logger=None) -> None:
    if "absolute_pos_embed" not in state_dict or not hasattr(model, "absolute_pos_embed"):
        return
    abs_pos = state_dict["absolute_pos_embed"]
    tgt = model.absolute_pos_embed
    N1, L, C1 = abs_pos.size()
    N2, C2, H, W = tgt.size()
    if (N1, C1, L) == (N2, C2, H * W):
        state_dict["absolute_pos_embed"] = abs_pos.view(N2, H, W, C2).permute(0, 3, 1, 2)
    else:
        if logger:
            logger.warning("Skipping absolute_pos_embed due to shape mismatch.")
        state_dict.pop("absolute_pos_embed", None)


def _resize_rel_pos_bias_tables(state_dict: Dict[str, Any], model_state: Dict[str, Any], logger=None) -> None:
    table_keys = [k for k in state_dict.keys() if "relative_position_bias_table" in k]
    for table_key in table_keys:
        pretrained = state_dict[table_key]
        target_key = table_key
        if target_key not in model_state:
            alt_key = ".".join(table_key.split(".")[:-1] + ["module", table_key.split(".")[-1]])
            if alt_key in model_state:
                target_key = alt_key
            else:
                continue
        current = model_state[target_key]
        L1, nH1 = pretrained.size()
        L2, nH2 = current.size()
        if nH1 != nH2:
            if logger:
                logger.warning(f"Skip resizing {table_key} due to head mismatch ({nH1} vs {nH2}).")
            state_dict.pop(table_key, None)
            continue
        if L1 != L2:
            S1, S2 = int(L1**0.5), int(L2**0.5)
            resized = F.interpolate(
                pretrained.permute(1, 0).view(1, nH1, S1, S1),
                size=(S2, S2),
                mode="bicubic",
            )
            state_dict[table_key] = resized.view(nH2, L2).permute(1, 0)


def load_checkpoint(model, filename: str, map_location: str = "cpu", strict: bool = False, logger=None) -> Dict[str, Any]:
    """
    Pure-torch checkpoint loader that supports common Swin weights and DDP prefixes.
    """
    if not os.path.isfile(filename):
        raise FileNotFoundError(f"Checkpoint not found: {filename}")

    checkpoint = torch.load(filename, map_location=map_location)
    if not isinstance(checkpoint, dict):
        raise RuntimeError(f"No state_dict found in checkpoint file {filename}")

    state_dict = checkpoint.get("state_dict") or checkpoint.get("model") or checkpoint
    state_dict = _strip_prefix(state_dict, "module.")
    if state_dict and sorted(state_dict.keys())[0].startswith("encoder."):
        state_dict = {k.replace("encoder.", "", 1): v for k, v in state_dict.items()}

    _reshape_absolute_pos_embed(state_dict, model, logger)
    _resize_rel_pos_bias_tables(state_dict, model.state_dict(), logger)

    missing, unexpected = model.load_state_dict(state_dict, strict=strict)
    if logger:
        if missing:
            logger.warning(f"Missing keys when loading checkpoint: {missing}")
        if unexpected:
            logger.warning(f"Unexpected keys when loading checkpoint: {unexpected}")

    return checkpoint


def save_checkpoint(model, filename: str, optimizer=None, meta: Dict[str, Any] = None) -> None:
    """
    Minimal torch.save wrapper; saves model state and optional optimizer/meta.
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    payload = {"state_dict": model.state_dict()}
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict() if hasattr(optimizer, "state_dict") else optimizer
    if meta:
        payload["meta"] = meta
    torch.save(payload, filename)
