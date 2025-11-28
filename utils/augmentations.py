import torch
import torch.nn.functional as F
import random
import numpy as np
from typing import Union, Tuple, List, Optional


class TraditionalImageAugmentation:
                                   

    def __init__(self, config=None):
                
        self.aug_params = {
            "brightness": 0.2,          
            "contrast": 0.2,           
            "saturation": 0.2,           
            "hflip_prob": 0.5,          
            "vflip_prob": 0.3,          
            "rotate_prob": 0.3,        
            "rotate_limit": 15,          
            "scale_prob": 0.3,        
            "scale_range": [0.9, 1.1],        
        }

                         
        if config and "augmentation" in config and "traditional" in config["augmentation"]:
            traditional_config = config["augmentation"]["traditional"]
            for key, value in traditional_config.items():
                if key in self.aug_params:
                    self.aug_params[key] = value

    def __call__(self, sample: dict) -> dict:
                         
                            
        sample = self._apply_spatial_transforms(sample)

                          
        for modality, tensor in sample.items():
            if modality == "label" or modality == "metadata" or modality == "file_id":
                continue

            if modality == "rgb":
                sample[modality] = self._apply_color_jitter(tensor)

        return sample

    def _apply_spatial_transforms(self, sample):
                             
              
        if random.random() < self.aug_params["hflip_prob"]:
            for key in sample:
                if key not in ["metadata", "file_id"]:
                    sample[key] = torch.flip(sample[key], [-1])

                                  
        if random.random() < self.aug_params["vflip_prob"]:
            for key in sample:
                if key not in ["metadata", "file_id"]:
                    sample[key] = torch.flip(sample[key], [-2])

              
        if random.random() < self.aug_params["rotate_prob"]:
            angle = random.uniform(-self.aug_params["rotate_limit"], self.aug_params["rotate_limit"])
                  
            theta = torch.tensor(
                [
                    [
                        torch.cos(torch.tensor(angle * torch.pi / 180)),
                        -torch.sin(torch.tensor(angle * torch.pi / 180)),
                        0,
                    ],
                    [
                        torch.sin(torch.tensor(angle * torch.pi / 180)),
                        torch.cos(torch.tensor(angle * torch.pi / 180)),
                        0,
                    ],
                ],
                dtype=torch.float,
            )

                          
            for key in sample:
                if key not in ["metadata", "file_id"]:
                    if key == "label":
                                    
                        mode = "nearest"
                    else:
                                    
                        mode = "bilinear"

                          
                    if key == "label":
                        h, w = sample[key].shape
                                       
                        sample[key] = sample[key].unsqueeze(0).unsqueeze(0)                
                        grid = F.affine_grid(theta.unsqueeze(0), sample[key].size(), align_corners=False)
                        sample[key] = F.grid_sample(sample[key].float(), grid, mode=mode, align_corners=False)
                        sample[key] = sample[key].squeeze(0).squeeze(0).long()             
                    else:
                                
                        c, h, w = sample[key].shape
                              
                        grid = F.affine_grid(theta.unsqueeze(0), torch.Size([1, c, h, w]), align_corners=False)
                        sample[key] = F.grid_sample(sample[key].unsqueeze(0), grid, mode=mode, align_corners=False)
                        sample[key] = sample[key].squeeze(0)

              
        if random.random() < self.aug_params["scale_prob"]:
            scale = random.uniform(self.aug_params["scale_range"][0], self.aug_params["scale_range"][1])

            for key in sample:
                if key not in ["metadata", "file_id"]:
                    if key == "label":
                                    
                        mode = "nearest"
                    else:
                                    
                        mode = "bilinear"

                          
                    if key == "label":
                        h, w = sample[key].shape
                              
                        sample[key] = sample[key].unsqueeze(0).unsqueeze(0)                

                                
                        theta = torch.tensor([[scale, 0, 0], [0, scale, 0]], dtype=torch.float).unsqueeze(0)

                        grid = F.affine_grid(theta, sample[key].size(), align_corners=False)
                        sample[key] = F.grid_sample(sample[key].float(), grid, mode=mode, align_corners=False)
                        sample[key] = sample[key].squeeze(0).squeeze(0).long()             
                    else:
                                
                        c, h, w = sample[key].shape

                                
                        theta = torch.tensor([[scale, 0, 0], [0, scale, 0]], dtype=torch.float).unsqueeze(0)

                        grid = F.affine_grid(theta, torch.Size([1, c, h, w]), align_corners=False)
                        sample[key] = F.grid_sample(sample[key].unsqueeze(0), grid, mode=mode, align_corners=False)
                        sample[key] = sample[key].squeeze(0)

        return sample

    def _apply_color_jitter(self, tensor):
                            
              
        if random.random() < 0.5:
            factor = random.uniform(1 - self.aug_params["brightness"], 1 + self.aug_params["brightness"])
            tensor = tensor * factor

               
        if random.random() < 0.5:
            factor = random.uniform(1 - self.aug_params["contrast"], 1 + self.aug_params["contrast"])
            tensor = (tensor - 0.5) * factor + 0.5

                          
        if random.random() < 0.5 and tensor.shape[0] == 3:
            factor = random.uniform(1 - self.aug_params["saturation"], 1 + self.aug_params["saturation"])
                   
            gray = tensor.mean(dim=0, keepdim=True).expand_as(tensor)
                   
            tensor = gray + factor * (tensor - gray)

                  
        tensor = torch.clamp(tensor, 0, 1) if tensor.max() <= 1 else tensor

        return tensor


class TraditionalNormalize:
                             

    def __init__(self, config=None):
        self.config = config
        self.modalities_to_normalize = ["dem", "insar_vel", "insar_phase", "rgb"]

                 
        self.default_norm = {
            "rgb": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},               
            "dem": {"mean": 0.0, "std": 1.0},
            "insar_vel": {"mean": 0.0, "std": 1.0},
        }

    def __call__(self, sample: dict) -> dict:
                       
        for modality in sample.keys():
            if modality in self.modalities_to_normalize:
                if modality not in ["metadata", "file_id", "label"]:
                    tensor = sample[modality]

                             
                    norm_params = self.default_norm.get(modality, {"mean": 0.0, "std": 1.0})

                                            
                    if self.config and modality in self.config:
                        if "mean" in self.config[modality]:
                            norm_params["mean"] = self.config[modality]["mean"]
                        if "std" in self.config[modality]:
                            norm_params["std"] = self.config[modality]["std"]

                           
                    if modality == "rgb" and isinstance(norm_params["mean"], list):
                                   
                        mean = torch.tensor(norm_params["mean"]).view(3, 1, 1)
                        std = torch.tensor(norm_params["std"]).view(3, 1, 1)
                        tensor = (tensor - mean) / std
                    else:
                                           
                        mean = norm_params["mean"]
                        std = norm_params["std"]
                        tensor = (tensor - mean) / std

                    sample[modality] = tensor

        return sample


                    
class Compose:
    def __init__(self, transforms: list, require_label: bool = True) -> None:
        self.transforms = transforms
        self.require_label = require_label

    def __call__(self, sample: dict) -> dict:
                
        assert isinstance(sample, dict), "Sample must be a dictionary"
        if self.require_label:
            assert "label" in sample, "Sample must contain 'label' key"

                        
        modality_shapes = {k: v.shape for k, v in sample.items() if k not in ["metadata", "label", "file_id"]}

                         
        if len(modality_shapes) > 0:
            first_shape = next(iter(modality_shapes.values()))[1:]        
            for modality, shape in modality_shapes.items():
                assert (
                    shape[1:] == first_shape
                ), f"Spatial dimensions mismatch for {modality}: {shape[1:]} vs {first_shape}"

              
        for transform in self.transforms:
            sample = transform(sample)

        return sample


def get_traditional_train_augmentation(config=None):
                     
    return Compose(
        [
            TraditionalImageAugmentation(config),
            TraditionalNormalize(config),
        ],
        require_label=True,
    )


def get_traditional_val_augmentation(config=None, require_label: bool = True):
                     
    return Compose(
        [
            TraditionalNormalize(config),
        ],
        require_label=require_label,
    )
