import torch
import numpy as np


class PolyWarmupAdamW(torch.optim.AdamW):

    def __init__(self, params, lr, weight_decay, betas, warmup_iter=None, max_iter=None, warmup_ratio=None, power=None):
        super().__init__(params, lr=lr, betas=betas, weight_decay=weight_decay, eps=1e-8)

        self.global_step = 0
        self.warmup_iter = warmup_iter
        self.warmup_ratio = warmup_ratio
        self.max_iter = max_iter
        self.power = power

        self.__init_lr = [group["lr"] for group in self.param_groups]

    def step(self, closure=None):
                    
        if self.global_step < self.warmup_iter:

            lr_mult = 1 - (1 - self.global_step / self.warmup_iter) * (1 - self.warmup_ratio)
            for i in range(len(self.param_groups)):
                self.param_groups[i]["lr"] = self.__init_lr[i] * lr_mult

        elif self.global_step < self.max_iter:

            lr_mult = (1 - self.global_step / self.max_iter) ** self.power
            for i in range(len(self.param_groups)):
                self.param_groups[i]["lr"] = self.__init_lr[i] * lr_mult

              
        super().step(closure)

        self.global_step += 1


class CosineAnnealingWarmupAdamW(torch.optim.AdamW):
                                                                                  

    def __init__(
        self, params, lr, weight_decay, betas, T_max=40000, eta_min=0, warmup_iterations=1500, warmup_ratio=1e-6
    ):
        super().__init__(params, lr=lr, betas=betas, weight_decay=weight_decay, eps=1e-8)

        self.global_step = 0
        self.T_max = T_max
        self.eta_min = eta_min
        self.warmup_iterations = warmup_iterations
        self.warmup_ratio = warmup_ratio

        self.__init_lr = [float(group["lr"]) for group in self.param_groups]                        

    def step(self, closure=None):
                      
        if self.global_step < self.warmup_iterations:
            lr_mult = 1 - (1 - self.global_step / self.warmup_iterations) * (1 - self.warmup_ratio)
            for i in range(len(self.param_groups)):
                self.param_groups[i]["lr"] = self.__init_lr[i] * lr_mult

                                
        else:
                                                 
            adjusted_step = self.global_step - self.warmup_iterations
                                             
            if adjusted_step < self.T_max:
                for i in range(len(self.param_groups)):
                                                                            
                    lr = float(self.eta_min) + 0.5 * (float(self.__init_lr[i]) - float(self.eta_min)) * (
                        1 + np.cos(np.pi * adjusted_step / self.T_max)
                    )
                    self.param_groups[i]["lr"] = lr

              
        super().step(closure)
        self.global_step += 1
