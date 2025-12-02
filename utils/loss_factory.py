import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import distance_transform_edt
from utils.losses import CompositeLoss


class LossFactory:
                                                                           

    @staticmethod
    def create_loss(config):
                   
        loss_config = config.get("training", {}).get("loss", {})
        loss_type = loss_config.get("type", "composite").lower()

                                                         
        if loss_type == "composite":
            ignore_index = loss_config.get("ignore_index", 255)
            task = loss_config.get("task", None)
            modal = loss_config.get("modal", None)
            uncertainty = loss_config.get("uncertainty", None)
            beta_reg = loss_config.get("beta_reg", None)
            fuse_consistency = loss_config.get("fuse_consistency", None)
            return CompositeLoss(
                ignore_index=ignore_index,
                task=task,
                modal=modal,
                uncertainty=uncertainty,
                beta_reg=beta_reg,
                fuse_consistency=fuse_consistency,
            )

                                      
                            
        ignore_index = loss_config.get("ignore_index", 255)
        alpha = loss_config.get("alpha", 0.5)                       
        weight = None

                                           
        if "class_weights" in loss_config:
            weight = torch.tensor(loss_config["class_weights"]).float().cuda()

    @staticmethod
    def _create_nll_loss(ignore_index=255, weight=None):
                                       
        criterion = nn.NLLLoss(weight=weight, ignore_index=ignore_index).cuda()

        def loss_fn(outputs, labels):
            total = 0.0
            for output in outputs:
                                                                      
                                                                   
                output = F.interpolate(output, size=labels.shape[1:], mode="bilinear", align_corners=False)
                total += criterion(output, labels)
            return total

        return loss_fn

    @staticmethod
    def _create_dice_loss(ignore_index=255):
                                        

        def loss_fn(outputs, labels):
            total = 0.0
            eps = 1e-6
            for output in outputs:
                                                                        
                                                                   
                output = F.interpolate(output, size=labels.shape[1:], mode="bilinear", align_corners=False)

                probs = torch.exp(output)                      
                B, C, H, W = probs.shape

                                                  
                valid_mask = (labels != ignore_index).unsqueeze(1)             
                one_hot = torch.zeros_like(probs)
                one_hot.scatter_(1, labels.clamp_min(0).unsqueeze(1), 1.0)                                    
                one_hot = one_hot * valid_mask                    

                                                  
                intersection = (probs * one_hot).sum(dim=(0, 2, 3))
                denom = probs.sum(dim=(0, 2, 3)) + one_hot.sum(dim=(0, 2, 3))
                dice = (2.0 * intersection + eps) / (denom + eps)
                dice_loss = 1.0 - dice.mean()
                total += dice_loss

            return total

        return loss_fn

    @staticmethod
    def _create_focal_loss(gamma=2.0, ignore_index=255, weight=None):
                                         

        def loss_fn(outputs, labels):
            loss = 0
            for output in outputs:
                                                                         
                                                                   
                output = F.interpolate(output, size=labels.shape[1:], mode="bilinear", align_corners=False)

                probs = torch.exp(output)

                                               
                batch_size, num_classes = output.shape[0], output.shape[1]
                one_hot = torch.zeros_like(probs)

                                     
                valid_mask = labels != ignore_index

                                                           
                for i in range(batch_size):
                    for c in range(num_classes):
                        one_hot[i, c, valid_mask[i]] = (labels[i, valid_mask[i]] == c).float()

                                                 
                if weight is not None:
                    weight_tensor = weight.view(1, -1, 1, 1)
                    one_hot = one_hot * weight_tensor

                                                               
                pt = torch.sum(probs * one_hot, dim=1) + 1e-10
                focal_weight = (1 - pt) ** gamma

                                    
                ce_loss = -torch.log(pt)

                                                     
                focal_loss = focal_weight * ce_loss

                                           
                focal_loss = torch.sum(focal_loss * valid_mask.float()) / (torch.sum(valid_mask.float()) + 1e-6)

                loss += focal_loss

            return loss

        return loss_fn

    @staticmethod
    def _create_contrastive_loss(temperature=0.1, ignore_index=255):
                   

        def loss_fn(outputs, labels):
                                                                            
                                                                       

                                                                 
            if len(outputs) <= 1 or not hasattr(outputs, "features"):
                print("Warning: No feature maps available for contrastive loss. Falling back to NLL loss.")
                return LossFactory._create_nll_loss(ignore_index)(outputs, labels)

                                                
            features = outputs.features                                     
            loss = 0

            for feat_map, output in zip(features, outputs):
                                                                   
                output = F.interpolate(output, size=labels.shape[1:], mode="bilinear", align_corners=False)

                                                                     
                feat_map = F.interpolate(feat_map, size=labels.shape[1:], mode="bilinear", align_corners=False)

                batch_size, num_features, height, width = feat_map.shape

                                    
                feat_map = F.normalize(feat_map, p=2, dim=1)

                                                               
                feat_map = feat_map.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, num_features)
                labels_flat = labels.view(batch_size, -1)

                                     
                valid_mask = labels_flat != ignore_index

                                     
                batch_loss = 0

                for b in range(batch_size):
                                      
                    b_valid = valid_mask[b]
                    if torch.sum(b_valid) == 0:
                        continue

                                                              
                    b_feat = feat_map[b, b_valid]
                    b_labels = labels_flat[b, b_valid]

                                               
                    similarity = torch.matmul(b_feat, b_feat.transpose(0, 1)) / temperature

                                                                 
                    pos_mask = b_labels.unsqueeze(1) == b_labels.unsqueeze(0)
                    neg_mask = ~pos_mask

                                                                 
                    pos_mask.fill_diagonal_(False)

                                  
                                                                                       
                    exp_sim = torch.exp(similarity)

                                                           
                    pos_exp_sum = torch.sum(exp_sim * pos_mask, dim=1)

                                                               
                    all_exp_sum = torch.sum(exp_sim, dim=1) - torch.diag(exp_sim)

                                                           
                                                       
                    loss_per_anchor = -torch.log((pos_exp_sum + 1e-10) / (all_exp_sum + 1e-10))

                                                           
                    b_loss = torch.mean(loss_per_anchor)
                    batch_loss += b_loss

                                           
                loss += batch_loss / batch_size

            return loss

        return loss_fn

    @staticmethod
    def _create_combined_loss(losses, weights, ignore_index=255):
                                                                           

        def loss_fn(outputs, labels):
            total_loss = 0

            for i, (loss_fn, weight) in enumerate(zip(losses, weights)):
                loss_value = loss_fn(outputs, labels)
                total_loss += weight * loss_value

            return total_loss

        return loss_fn
