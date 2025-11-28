import torch
from torch import nn
from typing import Optional


class MRG(nn.Module):
           

    def __init__(self, num_classes: int, num_modalities: int, context_channels: Optional[int] = None):
        super().__init__()
        self.num_classes = num_classes
        self.num_modalities = num_modalities
                                                                             
        self.alpha = nn.Parameter(torch.zeros(num_modalities, num_classes, 1, 1))

                              
        self.context_channels = context_channels
        if context_channels is not None:
            hidden = max(context_channels // 2, num_classes)
            self.context_heads = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Conv2d(context_channels, hidden, kernel_size=3, padding=1, bias=False),
                        nn.BatchNorm2d(hidden),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(hidden, num_classes, kernel_size=1),
                    )
                    for _ in range(num_modalities)
                ]
            )
        else:
            self.context_heads = None

    def _beta_from_context(self, modality_index: int, context: torch.Tensor) -> torch.Tensor:
                                                          
        beta_map = self.context_heads[modality_index](context)             
        return torch.sigmoid(beta_map)

    def _beta_global(self, modality_index: int, *, dtype=None, device=None) -> torch.Tensor:
        beta = torch.sigmoid(self.alpha[modality_index]).squeeze(-1).squeeze(-1)       
        if dtype is not None:
            beta = beta.to(dtype=dtype)
        if device is not None:
            beta = beta.to(device=device)
        return beta

    def forward(self, pl: torch.Tensor, modality_index: int, context: Optional[torch.Tensor] = None) -> torch.Tensor:
                          
        if context is not None and self.context_heads is not None:
            beta_map = self._beta_from_context(modality_index, context)             
        else:
            beta_vec = self._beta_global(modality_index, dtype=pl.dtype, device=pl.device)
            beta_map = beta_vec.view(1, self.num_classes, 1, 1).expand_as(pl)

        pl_hat = 1.0 - beta_map + beta_map * pl
        return pl_hat

    def get_reliability(self, modality_index: int, *, dtype=None, device=None) -> torch.Tensor:
                   
        return self._beta_global(modality_index, dtype=dtype, device=device)

    def discount_mass(
        self, mass: torch.Tensor, modality_index: int, context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
                   
        B, Kp1, H, W = mass.size()
        K = self.num_classes
        assert Kp1 == K + 1, "Mass tensor must have K+1 channels (including Theta)."

        singletons = mass[:, :K, :, :]
        m_theta = mass[:, K : K + 1, :, :]

        if context is not None and self.context_heads is not None:
            beta_map = self._beta_from_context(modality_index, context)             
            beta_expanded = beta_map
                            
            r_mean = beta_map.mean(dim=(2, 3))         
            r_mean = r_mean.mean(dim=1, keepdim=True).reshape(B, 1, 1, 1)             
        else:
            beta_vec = self._beta_global(modality_index, dtype=mass.dtype, device=mass.device)       
            beta_expanded = beta_vec.view(1, K, 1, 1).expand_as(singletons)
            r_mean = beta_vec.mean().view(1, 1, 1, 1)

        discounted_singletons = singletons * beta_expanded
        discounted_theta = (1.0 - r_mean) + r_mean * m_theta

        return torch.cat([discounted_singletons, discounted_theta], dim=1)

    @staticmethod
    def fuse_discounted_pl(pl_list):
        if not isinstance(pl_list, (list, tuple)) or len(pl_list) == 0:
            raise ValueError("pl_list must be non-empty")
        fused = pl_list[0]
        for t in range(1, len(pl_list)):
            fused = fused * pl_list[t]
        return fused

    @staticmethod
    def normalize_to_prob(fused_pl: torch.Tensor) -> torch.Tensor:
        K_sum = fused_pl.sum(1, keepdim=True)
        return fused_pl / (K_sum + 1e-12)
