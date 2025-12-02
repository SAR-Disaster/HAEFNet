import logging
from typing import Dict, Any, List

from .haef_net import HAEFNet
from .modules import set_num_parallel


def _collect_modalities(config: Dict[str, Any]) -> List[str]:
    modalities = [k[4:] for k, v in config.get("modalities", {}).items() if v and k.startswith("use_")]
    if not modalities:
        raise ValueError("No modalities enabled in config.modalities (use_* flags).")
    return modalities


def build_haefnet(config: Dict[str, Any]) -> HAEFNet:
    model_cfg = config.get("model", {})
    modalities = _collect_modalities(config)
    num_modalities = len(modalities)
    set_num_parallel(num_modalities)

    backbone = model_cfg.get("backbone", "swin_tiny")
    num_classes = model_cfg.get("num_classes", 2)
    n_heads = model_cfg.get("n_heads", 8)
    dpr = model_cfg.get("drop_path_rate", 0.1)
    drop_rate = model_cfg.get("drop_rate", 0.0)

    fusion_cfg = model_cfg.get("fusion", {})
    fusion_params = {
        "type": fusion_cfg.get("type", "conditional"),
        "sparsity": fusion_cfg.get("sparsity", 0.5),
    }

    gem_prototype_dim = model_cfg.get("gem_prototype_dim", 20)
    gem_geo_prior_weight = model_cfg.get("gem_geo_prior_weight", 0.1)
    use_evidential_fusion = model_cfg.get("use_evidential_fusion", True)
    use_aux_head = model_cfg.get("use_aux_head", False)
    use_mrg = model_cfg.get("use_mrg", False)
    prob_fusion = model_cfg.get("prob_fusion", "product")
    use_dempster = model_cfg.get("use_dempster", True)
    mrg_discount_on_mass = model_cfg.get("mrg_discount_on_mass", True)
    keep_pca_before_gem = model_cfg.get("keep_pca_before_gem", True)
    aggregation_channels = model_cfg.get("aggregation_channels", 256)

    logging.info(
        "Building HAEF-Net | backbone=%s | classes=%d | modalities=%s | evidential=%s | mrg=%s | dempster=%s",
        backbone,
        num_classes,
        ",".join(modalities),
        use_evidential_fusion,
        use_mrg,
        use_dempster,
    )

    return HAEFNet(
        backbone=backbone,
        num_classes=num_classes,
        n_heads=n_heads,
        dpr=dpr,
        drop_rate=drop_rate,
        num_parallel=num_modalities,
        fusion_params=fusion_params,
        gem_prototype_dim=gem_prototype_dim,
        gem_geo_prior_weight=gem_geo_prior_weight,
        use_evidential_fusion=use_evidential_fusion,
        use_aux_head=use_aux_head,
        use_mrg=use_mrg,
        prob_fusion=prob_fusion,
        use_dempster=use_dempster,
        mrg_discount_on_mass=mrg_discount_on_mass,
        keep_pca_before_gem=keep_pca_before_gem,
        aggregation_channels=aggregation_channels,
    )
