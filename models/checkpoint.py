import logging
from collections import OrderedDict
from typing import Any, Mapping, Optional

import torch


def load_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str,
    map_location: str = "cpu",
    strict: bool = False,
    logger: Optional[logging.Logger] = None,
) -> Any:
    if logger is None:
        logger = logging.getLogger(__name__)

    checkpoint: Mapping[str, Any] = torch.load(checkpoint_path, map_location=map_location)
    state_dict = checkpoint.get("state_dict", checkpoint)

    cleaned_state_dict = OrderedDict()
    for key, value in state_dict.items():
        new_key = key
        if new_key.startswith("module."):
            new_key = new_key[len("module.") :]
        cleaned_state_dict[new_key] = value

    load_info = model.load_state_dict(cleaned_state_dict, strict=strict)

    if load_info.missing_keys:
        logger.warning("Missing keys when loading %s: %s", checkpoint_path, ", ".join(load_info.missing_keys))
    if load_info.unexpected_keys:
        logger.warning(
            "Unexpected keys when loading %s: %s", checkpoint_path, ", ".join(load_info.unexpected_keys)
        )

    logger.info("Loaded checkpoint from %s (strict=%s)", checkpoint_path, strict)
    return load_info
