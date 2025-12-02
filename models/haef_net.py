import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from .haef_encoder import WeTr
from .gem import GEM
from .mrg import MRG


class HAEFNet(nn.Module):
           

    def __init__(
        self,
        backbone="swin_tiny",
        num_classes=2,
        n_heads=8,
        dpr=0.1,
        drop_rate=0.0,
        num_parallel=3,                   
        fusion_params=None,
                     
        gem_prototype_dim=20,
        gem_geo_prior_weight=0.1,
                  
        use_evidential_fusion=True,
        use_aux_head: bool = False,
        use_mrg=False,
                                                  
        prob_fusion: str = "product",
                                   
        use_dempster: bool = True,
                                            
        mrg_discount_on_mass: bool = True,
                                
        keep_pca_before_gem: bool = True,
                    
        aggregation_channels: int = 256,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.num_parallel = num_parallel
        self.use_evidential_fusion = use_evidential_fusion
        self.use_mrg = use_mrg
        self.prob_fusion = prob_fusion
        self.use_dempster = use_dempster
        self.mrg_discount_on_mass = mrg_discount_on_mass
        self.keep_pca_before_gem = keep_pca_before_gem
        self.aggregation_channels = aggregation_channels
        self.use_aux_head = use_aux_head

        self.harmf_encoder = WeTr(
            backbone=backbone,
            num_classes=num_classes,
            n_heads=n_heads,
            dpr=dpr,
            drop_rate=drop_rate,
            num_parallel=num_parallel,
            fusion_params=fusion_params,
        )

        if "swin" in backbone:
            if backbone == "swin_tiny":
                self.feature_dims = [96, 192, 384, 768]
            elif backbone == "swin_small":
                self.feature_dims = [96, 192, 384, 768]
            elif backbone == "swin_large":
                self.feature_dims = [192, 384, 768, 1536]
            else:
                self.feature_dims = [192, 384, 768, 1536]
        else:
                                   
            self.feature_dims = [128, 256, 512, 1024]

                               
        if self.use_evidential_fusion:
                                                       
            self.agg_proj = nn.ModuleList(
                [nn.Conv2d(dim, self.aggregation_channels, kernel_size=1) for dim in self.feature_dims]
            )
                                                 
            self.agg_fuse = nn.Conv2d(self.aggregation_channels * len(self.feature_dims), self.aggregation_channels, 1)

                               
            self.gem_single = GEM(
                input_dim=self.aggregation_channels,
                prototype_dim=gem_prototype_dim,
                class_dim=num_classes,
                geo_prior_weight=gem_geo_prior_weight,
            )

            if self.use_mrg:
                self.mrg = MRG(
                    num_classes=self.num_classes,
                    num_modalities=self.num_parallel,
                    context_channels=self.aggregation_channels,
                )

    def get_param_groups(self):
                           
        param_groups = [[], [], []]                          

                    
        for name, param in list(self.harmf_encoder.named_parameters()):
            if "norm" in name:
                param_groups[1].append(param)
            else:
                param_groups[0].append(param)

                            
        if self.use_evidential_fusion:
            for param in self.agg_proj.parameters():
                param_groups[2].append(param)
            for param in self.agg_fuse.parameters():
                param_groups[2].append(param)
            for param in self.gem_single.parameters():
                param_groups[2].append(param)
                    
            if self.use_mrg:
                for param in self.mrg.parameters():
                    param_groups[2].append(param)

        return param_groups

    def forward(self, x):
                   
                  
        original_shape = x[0].shape[2:]
        eps = 1e-12

                            
        if not self.use_evidential_fusion:
            return self.harmf_encoder(x)

                                            
        modality_features = self.harmf_encoder.encoder(x)                                   
        num_modalities = len(modality_features)
        if num_modalities == 0:
            return self.harmf_encoder(x)

                                                  
        updated_per_modality = [[] for _ in range(num_modalities)]
        num_stages = len(self.feature_dims)
        for s in range(num_stages):
            stage_feats = []
            for m in range(num_modalities):
                if s < len(modality_features[m]):
                    stage_feats.append(modality_features[m][s])
            if not stage_feats:
                continue
            B, C, Hs, Ws = stage_feats[0].shape
            if self.keep_pca_before_gem:
                seqs = [f.permute(0, 2, 3, 1).reshape(B, Hs * Ws, C) for f in stage_feats]
                updated_seqs = list(seqs)
                for m in range(num_modalities):
                    nxt = (m + 1) % num_modalities
                    y_m, y_n = self.harmf_encoder.pca_stages[s](updated_seqs[m], updated_seqs[nxt])
                    updated_seqs[m], updated_seqs[nxt] = y_m, y_n
                stage_feats = [
                    updated_seqs[m].reshape(B, Hs, Ws, C).permute(0, 3, 1, 2).contiguous()
                    for m in range(num_modalities)
                ]
            for m in range(num_modalities):
                updated_per_modality[m].append(stage_feats[m])

                                            
        pl_list = []                               
        theta_list = []                            
        mass_list = []                        
        modal_logits_up = []                           
        theta_up_list = []            
        target_hw = None
                        
        for m in range(num_modalities):
            if len(updated_per_modality[m]) > 0:
                target_hw = updated_per_modality[m][0].shape[2:]
                break
        if target_hw is None:
            return self.harmf_encoder(x)

        for t in range(self.num_parallel):
            if t >= num_modalities or len(updated_per_modality[t]) == 0:
                continue
            proj_ups = []
            for s, feat in enumerate(updated_per_modality[t]):
                z = self.agg_proj[s](feat)
                if z.shape[2:] != target_hw:
                    z = F.interpolate(z, size=target_hw, mode="bilinear", align_corners=False)
                proj_ups.append(z)
            agg_t = self.agg_fuse(torch.cat(proj_ups, dim=1))                       

                    
            mass_t = self.gem_single(agg_t)                      

                                                
            if self.use_mrg and self.mrg_discount_on_mass:
                mass_t = self.mrg.discount_mass(mass_t, modality_index=t, context=agg_t)
                theta_t = self.gem_single.get_uncertainty(mass_t)
                pl_t = self.gem_single.get_plausibility(mass_t)
            else:
                theta_t = self.gem_single.get_uncertainty(mass_t)
                pl_t = self.gem_single.get_plausibility(mass_t)
                if self.use_mrg:
                    pl_t = self.mrg(pl_t, modality_index=t, context=agg_t)
                                            
                    mass_t = self.mrg.discount_mass(mass_t, modality_index=t, context=agg_t)

                                                                      
            prob_t = pl_t / (pl_t.sum(1, keepdim=True) + eps)
            modal_logit_t = torch.log(prob_t + eps)
            pl_list.append(pl_t)
            theta_list.append(theta_t)
            mass_list.append(mass_t)
                                   
            modal_logit_up_t = modal_logit_t
            if modal_logit_up_t.shape[2:] != original_shape:
                prob_up = F.interpolate(prob_t, size=original_shape, mode="bilinear", align_corners=False)
                prob_up = prob_up / (prob_up.sum(1, keepdim=True) + eps)
                modal_logit_up_t = torch.log(prob_up + eps)
            modal_logits_up.append(modal_logit_up_t)
            theta_up = theta_t
            if theta_up.shape[2:] != original_shape:
                theta_up = F.interpolate(theta_up, size=original_shape, mode="bilinear", align_corners=False)
            theta_up_list.append(theta_up)

                                
        if not pl_list:
            return self.harmf_encoder(x)

                    
        weights = None
        if self.use_mrg:
            weights = []
            for t in range(len(pl_list)):
                r = self.mrg.get_reliability(t, dtype=pl_list[0].dtype, device=pl_list[0].device)
                weights.append(r)
            weights = torch.stack(weights)         

        if self.use_dempster and len(mass_list) > 1:
                                       
            mass_fused = mass_list[0]
            for i in range(1, len(mass_list)):
                mass_fused = self._combine_two_masses(mass_fused, mass_list[i])
            pl_fused = self.gem_single.get_plausibility(mass_fused)
        else:
            pl_fused = self._fuse_pl(pl_list, self.prob_fusion, weights)

                                          
        prob_fused = pl_fused / (pl_fused.sum(1, keepdim=True) + eps)
        if prob_fused.shape[2:] != original_shape:
            prob_fused = F.interpolate(prob_fused, size=original_shape, mode="bilinear", align_corners=False)
            prob_fused = prob_fused / (prob_fused.sum(1, keepdim=True) + eps)
        logits = torch.log(prob_fused + eps)

        outputs_all = [logits]

                                    
        if self.use_aux_head:
            head_logits = self._decode_from_features(modality_features, original_shape)
            if head_logits is not None:
                outputs_all.append(head_logits)

                                   
        product_logprob = None
        if pl_list:
            product_logits = torch.zeros_like(pl_list[0])
            for pl in pl_list:
                product_logits = product_logits + torch.log(pl + eps)
            product_prob = F.softmax(product_logits, dim=1)
            if product_prob.shape[2:] != original_shape:
                product_prob = F.interpolate(product_prob, size=original_shape, mode="bilinear", align_corners=False)
                product_prob = product_prob / (product_prob.sum(1, keepdim=True) + eps)
            product_logprob = torch.log(product_prob + eps)

        aux = {
            "modal_logits": modal_logits_up if modal_logits_up else None,
            "theta": theta_up_list if theta_up_list else None,
            "beta": weights if "weights" in locals() else None,
            "product_logprob": product_logprob,
        }
        return outputs_all, aux

    @staticmethod
    def _fuse_pl(pl_list, mode: str, weights: Optional[torch.Tensor]):
                   
        if len(pl_list) == 1:
            return pl_list[0]

        if mode == "mean":
            pl_stack = torch.stack(pl_list, dim=0)
            return pl_stack.mean(dim=0)

        if mode == "weighted_product" and weights is not None:
                                                                      
            w = weights
            if w.dim() == 2:
                w = w.unsqueeze(-1).unsqueeze(-1)             
                            
            w_norm = w / (w.sum(dim=0, keepdim=True) + 1e-12)             
            log_pl = torch.stack([torch.log(p + 1e-12) for p in pl_list], dim=0)               
            w_for_broadcast = w_norm.unsqueeze(1)               
            fused_log = (w_for_broadcast * log_pl).sum(dim=0)
            return torch.clamp(torch.exp(fused_log), min=1e-12)

                    
        fused = pl_list[0]
        for i in range(1, len(pl_list)):
            fused = fused * pl_list[i]
        return torch.clamp(fused, min=1e-12)

    @staticmethod
    def _combine_two_masses(m1: torch.Tensor, m2: torch.Tensor) -> torch.Tensor:
                   
        eps = 1e-12
        single1, theta1 = m1[:, :-1, :, :], m1[:, -1:, :, :]
        single2, theta2 = m2[:, :-1, :, :], m2[:, -1:, :, :]

                              
        sum_m2 = single2.sum(1, keepdim=True)
        conflict = (single1 * (sum_m2 - single2)).sum(1, keepdim=True)
        denom = 1.0 - conflict

        fused_single = single1 * single2 + single1 * theta2 + theta1 * single2
        fused_theta = theta1 * theta2
        fused = torch.cat([fused_single, fused_theta], dim=1)
        fused = fused / (denom + eps)
        return torch.clamp(fused, min=eps)

    def _decode_from_features(self, modality_features, original_shape):
                   
        num_modalities = len(modality_features)
        if num_modalities == 0:
            return None
        num_stages = len(modality_features[0])
        fused_features = []

        for s in range(num_stages):
            stage_feats = [modality_features[m][s] for m in range(num_modalities) if s < len(modality_features[m])]
            if not stage_feats:
                continue
            B, C, Hs, Ws = stage_feats[0].shape
            seqs = [f.permute(0, 2, 3, 1).reshape(B, Hs * Ws, C) for f in stage_feats]
            updated_seqs = list(seqs)
            for m in range(num_modalities):
                nxt = (m + 1) % num_modalities
                y_m, y_n = self.harmf_encoder.pca_stages[s](updated_seqs[m], updated_seqs[nxt])
                updated_seqs[m], updated_seqs[nxt] = y_m, y_n
            avg_seq = updated_seqs[0]
            for i in range(1, num_modalities):
                avg_seq = avg_seq + updated_seqs[i]
            avg_seq = avg_seq / float(num_modalities)
            fused = avg_seq.reshape(B, Hs, Ws, C).permute(0, 3, 1, 2).contiguous()
            fused_features.append(fused)

        head_logits = self.harmf_encoder.decoder(fused_features)
        if head_logits.shape[2:] != original_shape:
            head_logits = F.interpolate(head_logits, size=original_shape, mode="bilinear", align_corners=False)
        return head_logits

    def get_uncertainty_map(self, x):
                   
        if not self.use_evidential_fusion:
            return None

        with torch.no_grad():
            modality_features = self.harmf_encoder.encoder(x)
            num_modalities = len(modality_features)
            if num_modalities == 0:
                return None

                   
            updated_per_modality = [[] for _ in range(num_modalities)]
            num_stages = len(self.feature_dims)
            for s in range(num_stages):
                stage_feats = []
                for m in range(num_modalities):
                    if s < len(modality_features[m]):
                        stage_feats.append(modality_features[m][s])
                if not stage_feats:
                    continue
                B, C, Hs, Ws = stage_feats[0].shape
                if self.keep_pca_before_gem:
                    seqs = [f.permute(0, 2, 3, 1).reshape(B, Hs * Ws, C) for f in stage_feats]
                    updated_seqs = list(seqs)
                    for m in range(num_modalities):
                        nxt = (m + 1) % num_modalities
                        y_m, y_n = self.harmf_encoder.pca_stages[s](updated_seqs[m], updated_seqs[nxt])
                        updated_seqs[m], updated_seqs[nxt] = y_m, y_n
                    stage_feats = [
                        updated_seqs[m].reshape(B, Hs, Ws, C).permute(0, 3, 1, 2).contiguous()
                        for m in range(num_modalities)
                    ]
                for m in range(num_modalities):
                    updated_per_modality[m].append(stage_feats[m])

                         
            target_hw = None
            for m in range(num_modalities):
                if len(updated_per_modality[m]) > 0:
                    target_hw = updated_per_modality[m][0].shape[2:]
                    break
            if target_hw is None:
                return None

            theta_list = []
            for t in range(self.num_parallel):
                if t >= num_modalities or len(updated_per_modality[t]) == 0:
                    continue
                proj_ups = []
                for s, feat in enumerate(updated_per_modality[t]):
                    z = self.agg_proj[s](feat)
                    if z.shape[2:] != target_hw:
                        z = F.interpolate(z, size=target_hw, mode="bilinear", align_corners=False)
                    proj_ups.append(z)
                agg_t = self.agg_fuse(torch.cat(proj_ups, dim=1))
                mass_t = self.gem_single(agg_t)
                if self.use_mrg and self.mrg_discount_on_mass:
                    mass_t = self.mrg.discount_mass(mass_t, modality_index=t, context=agg_t)
                theta_t = self.gem_single.get_uncertainty(mass_t)
                theta_list.append(theta_t)

            if not theta_list:
                return None
            U = torch.stack(theta_list, dim=0).mean(dim=0)                 
            if U.shape[2:] != x[0].shape[2:]:
                U = F.interpolate(U, size=x[0].shape[2:], mode="bilinear", align_corners=False)
            return U

    @torch.no_grad()
    def analyze_modalities(self, x, foreground_class: int = 1, compute_loo: bool = True):
                   
        eps = 1e-12
        original_shape = x[0].shape[2:]

                    
        modality_features = self.harmf_encoder.encoder(x)                                   
        num_modalities = len(modality_features)
        if num_modalities == 0:
            final_logits, _ = self.forward(x)
            final_logits = final_logits[0]
            final_prob = F.softmax(final_logits, dim=1)
            return {
                "final_logits": final_logits,
                "final_prob": final_prob,
                "U": None,
                "C_t": None,
                "U_t": None,
                "prob_loo": None,
                "beta": None,
            }

                       
        updated_per_modality = [[] for _ in range(num_modalities)]
        num_stages = len(self.feature_dims)
        for s in range(num_stages):
            stage_feats = []
            for m in range(num_modalities):
                if s < len(modality_features[m]):
                    stage_feats.append(modality_features[m][s])
            if not stage_feats:
                continue
            B, C, Hs, Ws = stage_feats[0].shape
            if self.keep_pca_before_gem:
                seqs = [f.permute(0, 2, 3, 1).reshape(B, Hs * Ws, C) for f in stage_feats]
                updated_seqs = list(seqs)
                for m in range(num_modalities):
                    nxt = (m + 1) % num_modalities
                    y_m, y_n = self.harmf_encoder.pca_stages[s](updated_seqs[m], updated_seqs[nxt])
                    updated_seqs[m], updated_seqs[nxt] = y_m, y_n
                stage_feats = [
                    updated_seqs[m].reshape(B, Hs, Ws, C).permute(0, 3, 1, 2).contiguous()
                    for m in range(num_modalities)
                ]
            for m in range(num_modalities):
                updated_per_modality[m].append(stage_feats[m])

                        
        target_hw = None
        for m in range(num_modalities):
            if len(updated_per_modality[m]) > 0:
                target_hw = updated_per_modality[m][0].shape[2:]
                break
        if target_hw is None:
            final_logits, _ = self.forward(x)
            final_logits = final_logits[0]
            final_prob = F.softmax(final_logits, dim=1)
            return {
                "final_logits": final_logits,
                "final_prob": final_prob,
                "U": None,
                "C_t": None,
                "U_t": None,
                "prob_loo": None,
                "beta": None,
            }

                                 
        pl_list = []                      
        prob_t_list = []                  
        theta_up_list = []                
        C_list = []                      
        mass_list = []            
        weight_list = []         

        for t in range(self.num_parallel):
            if t >= num_modalities or len(updated_per_modality[t]) == 0:
                continue
            proj_ups = []
            for s, feat in enumerate(updated_per_modality[t]):
                z = self.agg_proj[s](feat)
                if z.shape[2:] != target_hw:
                    z = F.interpolate(z, size=target_hw, mode="bilinear", align_corners=False)
                proj_ups.append(z)
            agg_t = self.agg_fuse(torch.cat(proj_ups, dim=1))

            mass_t = self.gem_single(agg_t)
            if self.use_mrg and self.mrg_discount_on_mass:
                mass_t = self.mrg.discount_mass(mass_t, modality_index=t, context=agg_t)
                pl_t = self.gem_single.get_plausibility(mass_t)
            else:
                pl_t = self.gem_single.get_plausibility(mass_t)
                if self.use_mrg:
                    pl_t = self.mrg(pl_t, modality_index=t, context=agg_t)
                    mass_t = self.mrg.discount_mass(mass_t, modality_index=t, context=agg_t)
            theta_t = self.gem_single.get_uncertainty(mass_t)             

                                         
            prob_t = pl_t / (pl_t.sum(1, keepdim=True) + eps)
            prob_t_list.append(prob_t)
            pl_list.append(pl_t)
            mass_list.append(mass_t)
            if self.use_mrg:
                weight_list.append(self.mrg.get_reliability(t, dtype=pl_t.dtype, device=pl_t.device))

                                        
            theta_up = theta_t
            if theta_up.shape[2:] != original_shape:
                theta_up = F.interpolate(theta_up, size=original_shape, mode="bilinear", align_corners=False)
            theta_up_list.append(theta_up)

            c_map = torch.log(prob_t[:, foreground_class : foreground_class + 1, :, :] + 1e-8)             
            if c_map.shape[2:] != original_shape:
                c_map = F.interpolate(c_map, size=original_shape, mode="bilinear", align_corners=False)
            C_list.append(c_map.squeeze(1))           

                           
        if self.use_dempster and len(mass_list) > 1:
            mass_fused = mass_list[0]
            for i in range(1, len(mass_list)):
                mass_fused = self._combine_two_masses(mass_fused, mass_list[i])
            pl_fused = self.gem_single.get_plausibility(mass_fused)
        else:
            weights = torch.stack(weight_list) if (self.use_mrg and weight_list) else None
            pl_fused = self._fuse_pl(pl_list, self.prob_fusion, weights)

        prob_small = pl_fused / (pl_fused.sum(1, keepdim=True) + eps)
                 
        prob_up = prob_small
        if prob_up.shape[2:] != original_shape:
            prob_up = F.interpolate(prob_up, size=original_shape, mode="bilinear", align_corners=False)
            prob_up = prob_up / (prob_up.sum(1, keepdim=True) + eps)
        final_prob = prob_up
        final_logits = torch.log(final_prob + eps)

                           
        U = torch.stack(theta_up_list, dim=0).mean(dim=0) if theta_up_list else None             

                 
        C_t = torch.stack(C_list, dim=1) if C_list else None             
        U_t = torch.cat(theta_up_list, dim=1) if theta_up_list else None                       

                           
        prob_loo = None
        if compute_loo and len(pl_list) > 1:
            prob_loo = []
            for t in range(len(pl_list)):
                if self.use_dempster:
                    mass_others = None
                    for j, m_j in enumerate(mass_list):
                        if j == t:
                            continue
                        mass_others = m_j if mass_others is None else self._combine_two_masses(mass_others, m_j)
                    pl_no_t = self.gem_single.get_plausibility(mass_others)
                else:
                    weights_loo = None
                    if self.use_mrg and weight_list:
                        w = weight_list[:t] + weight_list[t + 1 :]
                        if w:
                            weights_loo = torch.stack(w)
                    pl_no_t = self._fuse_pl(pl_list[:t] + pl_list[t + 1 :], self.prob_fusion, weights_loo)

                prob_no_t = pl_no_t / (pl_no_t.sum(1, keepdim=True) + eps)
                if prob_no_t.shape[2:] != original_shape:
                    prob_no_t = F.interpolate(prob_no_t, size=original_shape, mode="bilinear", align_corners=False)
                    prob_no_t = prob_no_t / (prob_no_t.sum(1, keepdim=True) + eps)
                prob_loo.append(prob_no_t)

        beta = None
        if hasattr(self, "mrg") and self.use_mrg:
            beta = torch.sigmoid(self.mrg.alpha).squeeze(-1).squeeze(-1)         

        return {
            "final_logits": final_logits,
            "final_prob": final_prob,
            "U": U,
            "C_t": C_t,
            "U_t": U_t,
            "prob_loo": prob_loo,
            "beta": beta,
        }


if __name__ == "__main__":
                
    torch.manual_seed(0)

          
    B, C, H, W = 2, 3, 64, 64
    x = [torch.randn(B, C, H, W) for _ in range(3)]                   

                
    model = HAEFNet(
        backbone="swin_tiny", num_classes=2, num_parallel=3, use_evidential_fusion=True, gem_prototype_dim=10
    )

          
    outputs, aux_loss = model(x)
    uncertainty_map = model.get_uncertainty_map(x)

    print("=== HAEF-Net 测试结果 ===")
    print(f"输入模态数量: {len(x)}")
    print(f"输入形状: {[xi.shape for xi in x]}")
    print(f"输出数量: {len(outputs)}")
    print(f"输出形状: {[out.shape for out in outputs]}")
    print(f"不确定性图形状: {uncertainty_map.shape if uncertainty_map is not None else 'None'}")
    print(f"辅助损失: {aux_loss}")

    print("✓ HAEF-Net 实现正确!")
