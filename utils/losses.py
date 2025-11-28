import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict


def _resize(input: torch.Tensor, target_hw) -> torch.Tensor:
                                                                  
    if input.shape[2:] == target_hw:
        return input
    return F.interpolate(input, size=target_hw, mode="bilinear", align_corners=False)


class CompositeLoss(nn.Module):
           

    def __init__(
        self,
        *,
        ignore_index: int = 255,
        task: Optional[Dict] = None,
        modal: Optional[Dict] = None,
        uncertainty: Optional[Dict] = None,
        beta_reg: Optional[Dict] = None,
        fuse_consistency: Optional[Dict] = None,
    ):
        super().__init__()
        self.ignore_index = ignore_index

                          
        task = task or {"name": "ce_dice", "ce_weight": 1.0, "dice_weight": 1.0}
        self.task_name = task.get("name", "ce_dice")
        self.task_ce_weight = float(task.get("ce_weight", 1.0))
        self.task_dice_weight = float(task.get("dice_weight", 1.0))

                                
        modal = modal or {"weight": 0.0}
        self.modal_weight = float(modal.get("weight", 0.0))

                                    
        uncertainty = uncertainty or {"weight": 0.0, "target": 0.3}
        self.uncertainty_weight = float(uncertainty.get("weight", 0.0))
        self.uncertainty_target = float(uncertainty.get("target", 0.3))

                          
        beta_reg = beta_reg or {"weight": 0.0}
        self.beta_weight = float(beta_reg.get("weight", 0.0))

                            
        fuse_consistency = fuse_consistency or {"weight": 0.0}
        self.consistency_weight = float(fuse_consistency.get("weight", 0.0))

    def forward(self, outputs, labels, *, modal_logits=None, theta=None, beta=None, product_logprob=None):
                   
        target_hw = labels.shape[1:]
        total = 0.0

                                        
        fused_logprob = _resize(outputs, target_hw)
        total += self._task_loss(fused_logprob, labels)

                           
        if self.modal_weight > 0 and modal_logits:
            modal_total = 0.0
            for logit in modal_logits:
                logit = _resize(logit, target_hw)
                modal_total += self._task_loss(logit, labels)
            modal_total = modal_total / float(len(modal_logits))
            total += self.modal_weight * modal_total

                                             
        if self.uncertainty_weight > 0 and theta is not None:
            if isinstance(theta, (list, tuple)):
                theta_stack = torch.stack([_resize(t, target_hw) for t in theta], dim=0)
                theta_mean = theta_stack.mean(dim=0)
            else:
                theta_mean = _resize(theta, target_hw)
            valid = (labels != self.ignore_index).float()
            target = torch.full_like(theta_mean, self.uncertainty_target)
            mse = ((theta_mean.squeeze(1) - target.squeeze(1)) ** 2) * valid
            total += self.uncertainty_weight * (mse.sum() / (valid.sum() + 1e-12))

                                         
        if self.beta_weight > 0 and beta is not None:
                                                                
            if beta.dim() > 2:
                beta_vec = beta.squeeze(-1).squeeze(-1)
            else:
                beta_vec = beta
            beta_clamped = beta_vec.clamp(1e-6, 1 - 1e-6)
            ent = -beta_clamped * beta_clamped.log() - (1 - beta_clamped) * (1 - beta_clamped).log()
            total += self.beta_weight * ent.mean()

                                        
        if self.consistency_weight > 0 and product_logprob is not None:
            fused_logprob = _resize(fused_logprob, product_logprob.shape[2:])
                                                    
            valid = (labels != self.ignore_index).float().unsqueeze(1)
            kl = torch.exp(fused_logprob) * (fused_logprob - product_logprob)
            total += self.consistency_weight * (kl * valid).sum() / (valid.sum() + 1e-12)

        return total

    def _task_loss(self, logprob: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
                                                         
        ce = F.nll_loss(logprob, labels, ignore_index=self.ignore_index)
        if self.task_name == "ce":
            return ce * self.task_ce_weight
                                 
        prob = logprob.exp()
        valid_mask = (labels != self.ignore_index).unsqueeze(1)             
        one_hot = torch.zeros_like(prob)
        one_hot.scatter_(1, labels.clamp_min(0).unsqueeze(1), 1.0)
        one_hot = one_hot * valid_mask
        eps = 1e-6
        intersection = (prob * one_hot).sum(dim=(0, 2, 3))
        denom = prob.sum(dim=(0, 2, 3)) + one_hot.sum(dim=(0, 2, 3))
        dice = (2.0 * intersection + eps) / (denom + eps)
        dice_loss = 1.0 - dice.mean()
        return self.task_ce_weight * ce + self.task_dice_weight * dice_loss
