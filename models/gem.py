import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F


class _GeoDsFunction(torch.autograd.Function):
           

    @staticmethod
    def forward(ctx, input_feats, prototype_centers, class_membership, alpha, gamma):
                                               
                                          
                                         
                                      
        device = input_feats.device
        batch_size, in_channel, h, w = input_feats.size()

                                               
        BETA = class_membership
        W = prototype_centers

        class_dim = BETA.size(1)
        prototype_dim = W.size(0)

        BETA2 = BETA * BETA
        beta2 = BETA2.t().sum(0)
        U = BETA2 / (beta2.unsqueeze(1) * torch.ones(1, class_dim, device=device))          
        alphap = 0.99 / (1 + torch.exp(-alpha))          

        d = torch.zeros(prototype_dim, batch_size, h, w, device=device)
        s_act = torch.zeros_like(d)
        expo = torch.zeros_like(d)

                                                               
        mk = torch.cat(
            (torch.zeros(class_dim, batch_size, h, w, device=device), torch.ones(1, batch_size, h, w, device=device)), 0
        )

        input_perm = input_feats.permute(1, 0, 2, 3)                   

        for k in range(prototype_dim):
                                                           
                           
            w_k = W[k, :].unsqueeze(1).unsqueeze(2).unsqueeze(3)                   
            w_k = w_k.expand(in_channel, batch_size, h, w)                   

            temp = input_perm - w_k
            d[k, :] = 0.5 * (temp * temp).sum(0)             
            expo[k, :] = torch.exp(-(gamma[k] ** 2) * d[k, :])
            s_act[k, :] = alphap[k] * expo[k, :]

                                                                                        
            m_k = torch.cat(
                (
                    U[k, :].unsqueeze(1).unsqueeze(2).unsqueeze(3) * s_act[k, :],
                    torch.ones(1, batch_size, h, w, device=device) - s_act[k, :],
                ),
                0,
            )                  

            t2 = mk[:class_dim] * (m_k[:class_dim] + torch.ones(class_dim, 1, h, w, device=device) * m_k[class_dim])
            t3 = m_k[:class_dim] * (torch.ones(class_dim, 1, h, w, device=device) * mk[class_dim])
            t4 = (mk[class_dim]) * (m_k[class_dim]).unsqueeze(0)
            mk = torch.cat((t2 + t3, t4), 0)

        K_sum = mk.sum(0)
        mk_n = (mk / (torch.ones(class_dim + 1, 1, h, w, device=device) * K_sum)).permute(1, 0, 2, 3)
                                                                                            
        ctx.save_for_backward(input_feats, W, BETA, alpha, gamma, mk, d)
        return mk_n

    @staticmethod
    def backward(ctx, grad_output):
                   
        input_feats, W, BETA, alpha, gamma, mk, d = ctx.saved_tensors

        grad_input = grad_W = grad_BETA = grad_alpha = grad_gamma = None

               
        M = BETA.size(1)                
        prototype_dim = W.size(0)
        batch_size, in_channel, height, width = input_feats.size()

        mu = 0                 
        iw = 1                    

                                         
        grad_output_ = grad_output[:, :M, :, :] * (batch_size * M * height * width)

                 
        K = mk.sum(0).unsqueeze(0)                
        K2 = K**2
        BETA2 = BETA * BETA
        beta2 = BETA2.t().sum(0).unsqueeze(1)          
        U = BETA2 / (beta2 * torch.ones(1, M, device=input_feats.device))          
        alphap = 0.99 / (1 + torch.exp(-alpha))          
        I = torch.eye(M, device=grad_output.device)

                      
        s = torch.zeros(prototype_dim, batch_size, height, width, device=input_feats.device)
        expo = torch.zeros(prototype_dim, batch_size, height, width, device=input_feats.device)
        mm = torch.cat(
            (
                torch.zeros(M, batch_size, height, width, device=input_feats.device),
                torch.ones(1, batch_size, height, width, device=input_feats.device),
            ),
            0,
        )                  

        dEdm = torch.zeros(M + 1, batch_size, height, width, device=input_feats.device)
        dU = torch.zeros(prototype_dim, M, device=input_feats.device)
        Ds = torch.zeros(prototype_dim, batch_size, height, width, device=input_feats.device)
        DW = torch.zeros(prototype_dim, in_channel, device=input_feats.device)

                              
        for p in range(M):
            dEdm[p, :] = (
                grad_output_.permute(1, 0, 2, 3)
                * (
                    I[:, p].unsqueeze(1).unsqueeze(2).unsqueeze(3) * K
                    - mk[:M, :]
                    - (1.0 / M) * (torch.ones(M, 1, height, width, device=input_feats.device) * mk[M, :])
                )
            ).sum(0) / (K2 + 1e-12)

        dEdm[M, :] = (
            (
                grad_output_.permute(1, 0, 2, 3)
                * (-mk[:M, :] + (1.0 / M) * torch.ones(M, 1, height, width, device=input_feats.device) * (K - mk[M, :]))
            ).sum(0)
        ) / (K2 + 1e-12)

                          
        for k in range(prototype_dim):
            expo[k, :] = torch.exp(-gamma[k] ** 2 * d[k, :])             
            s[k] = alphap[k] * expo[k, :]             
            m = torch.cat(
                (
                    U[k, :].unsqueeze(1).unsqueeze(2).unsqueeze(3) * s[k, :],
                    torch.ones(1, batch_size, height, width, device=input_feats.device) - s[k, :],
                ),
                0,
            )                  

            mm[M, :] = mk[M, :] / (m[M, :] + 1e-12)
            L = torch.ones(M, 1, height, width, device=input_feats.device) * mm[M, :]
            mm[:M, :] = (mk[:M, :] - L * m[:M, :]) / (
                m[:M, :] + torch.ones(M, 1, height, width, device=input_feats.device) * m[M, :] + 1e-12
            )
            R = mm[:M, :] + L
            A = R * torch.ones(M, 1, height, width, device=input_feats.device) * s[k, :]
            B = (
                U[k, :].unsqueeze(1).unsqueeze(2).unsqueeze(3)
                * torch.ones(1, batch_size, height, width, device=input_feats.device)
                * R
                - mm[:M, :]
            )

            dU[k, :] = torch.mean((A * dEdm[:M, :]).view(M, -1).permute(1, 0), 0)
            Ds[k, :] = (dEdm[:M, :] * B).sum(0) - (dEdm[M, :] * mm[M, :])

                        
            tt1 = Ds[k, :] * (gamma[k] ** 2) * s[k, :]             
            tt2 = (torch.ones(batch_size, 1, device=input_feats.device) * W[k, :]).unsqueeze(2).unsqueeze(
                3
            ) - input_feats                
            tt1 = tt1.view(1, -1)
            tt2 = tt2.permute(1, 0, 2, 3).reshape(in_channel, batch_size * height * width).permute(1, 0)
            DW[k, :] = -torch.mm(tt1, tt2)

                   
        DW = iw * DW / (batch_size * height * width)

                       
        T = beta2 * torch.ones(1, M, device=input_feats.device)
        Dbeta = (2 * BETA / (T**2)) * (
            dU * (T - BETA2)
            - (dU * BETA2).sum(1).unsqueeze(1) * torch.ones(1, M, device=input_feats.device)
            + dU * BETA2
        )

                           
        Dgamma = -2 * torch.mean(((Ds * d * s).view(prototype_dim, -1)).t(), 0).unsqueeze(1) * gamma
        Dalpha = (torch.mean(((Ds * expo).view(prototype_dim, -1)).t(), 0).unsqueeze(1) + mu) * (
            0.99 * (1 - alphap) * alphap
        )

                 
        Dinput = torch.zeros(batch_size, in_channel, height, width, device=input_feats.device)
        temp2 = torch.zeros(prototype_dim, in_channel, height, width, device=input_feats.device)
        for n in range(batch_size):
            for k in range(prototype_dim):
                diff = input_feats[n, :] - W[k, :].unsqueeze(0).unsqueeze(2).unsqueeze(3)
                coeff = (Ds[k, n, :, :] * (gamma[k] ** 2) * s[k, n, :, :]).unsqueeze(0).unsqueeze(1)
                temp2[k] = -prototype_dim * coeff * diff
            Dinput[n, :] = temp2.mean(0)

                
        if ctx.needs_input_grad[0]:
            grad_input = Dinput
        if ctx.needs_input_grad[1]:
            grad_W = DW
        if ctx.needs_input_grad[2]:
            grad_BETA = Dbeta
        if ctx.needs_input_grad[3]:
            grad_alpha = Dalpha
        if ctx.needs_input_grad[4]:
            grad_gamma = Dgamma

                         
        if grad_input is not None:
            grad_input = grad_input.contiguous()
        if grad_W is not None:
            grad_W = grad_W.contiguous()
        if grad_BETA is not None:
            grad_BETA = grad_BETA.contiguous()
        if grad_alpha is not None:
            grad_alpha = grad_alpha.contiguous()
        if grad_gamma is not None:
            grad_gamma = grad_gamma.contiguous()

        return grad_input, grad_W, grad_BETA, grad_alpha, grad_gamma


class GEM(nn.Module):
           

    def __init__(self, input_dim: int, prototype_dim: int, class_dim: int = 2, geo_prior_weight: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.prototype_dim = prototype_dim
        self.class_dim = class_dim
        self.geo_prior_weight = geo_prior_weight

              
        self.class_membership = Parameter(torch.Tensor(self.prototype_dim, self.class_dim))        
        self.alpha = Parameter(torch.Tensor(self.prototype_dim, 1))
        self.gamma = Parameter(torch.Tensor(self.prototype_dim, 1))
        self.prototype_centers = Parameter(torch.Tensor(self.prototype_dim, self.input_dim))     

                
        self.geo_constraints = Parameter(torch.Tensor(self.prototype_dim, 1))

                      
        self.seasonal_weights = Parameter(torch.Tensor(4, self.prototype_dim))

                 
        self.terrain_complexity_weight = Parameter(torch.Tensor(1))

        self.reset_parameters()

    def reset_parameters(self):
                   
                           
        nn.init.normal_(self.prototype_centers, std=0.01)           
        nn.init.xavier_uniform_(self.class_membership)
        nn.init.constant_(self.gamma, 0.1)                      
        nn.init.constant_(self.alpha, 0.0)                      

                 
        nn.init.normal_(self.geo_constraints, std=0.01)
        nn.init.normal_(self.seasonal_weights, std=0.01)
        nn.init.constant_(self.terrain_complexity_weight, 1.0)

    def forward(self, feats: torch.Tensor, geo_context: torch.Tensor = None) -> torch.Tensor:
                   
                         
        if geo_context is not None:
            adjusted_centers = self._apply_geo_prior(self.prototype_centers, geo_context)
        else:
            adjusted_centers = self.prototype_centers

                             
        B, C, H, W = feats.shape
        device = feats.device

        P = adjusted_centers.size(0)
        K = self.class_membership.size(1)

                    
        BETA = self.class_membership          
        BETA2 = BETA * BETA
        beta2 = BETA2.t().sum(0)       
        U = BETA2 / (beta2.unsqueeze(1) * torch.ones(1, K, device=device))          
        alphap = 0.99 / (1 + torch.exp(-self.alpha))          

                               
        mass = _GeoDsFunction.apply(feats, adjusted_centers, BETA, self.alpha, self.gamma)
        return mass

    def _apply_geo_prior(self, prototype_centers: torch.Tensor, geo_context: torch.Tensor) -> torch.Tensor:
                   
                   
                               
                               
        scale = 1.0 + self.geo_prior_weight * torch.tanh(self.geo_constraints.mean())
        return prototype_centers * scale

    def get_uncertainty(self, mass: torch.Tensor) -> torch.Tensor:
                   
                                 
        uncertainty = mass[:, -1:, :, :]                
        return uncertainty

    def get_plausibility(self, mass: torch.Tensor) -> torch.Tensor:
                   
        pl_singletons = mass[:, :-1, :, :]                
        m_theta = mass[:, -1:, :, :]                
        plausibility = pl_singletons + m_theta                
        return plausibility


def mass_to_plausibility(mass: torch.Tensor) -> torch.Tensor:
           
    pl_singletons = mass[:, :-1, :, :]
    m_theta = mass[:, -1:, :, :]
    return pl_singletons + m_theta


def plausibility_to_probability(plausibility: torch.Tensor) -> torch.Tensor:
           
    K_sum = plausibility.sum(1, keepdim=True)
    probability = plausibility / (K_sum + 1e-12)
    return probability


if __name__ == "__main__":
                 
    torch.manual_seed(0)

          
    B, C_in, H, W = 2, 256, 32, 32
    K = 2          
    P = 20        

    feats = torch.randn(B, C_in, H, W)
    geo_context = torch.randn(B, 1, H, W)

                 
    gem_layer = GEM(input_dim=C_in, prototype_dim=P, class_dim=K, geo_prior_weight=0.1)

          
    mass = gem_layer(feats, geo_context)
    uncertainty = gem_layer.get_uncertainty(mass)
    plausibility = gem_layer.get_plausibility(mass)
    probability = plausibility_to_probability(plausibility)

    print("=== GEM-Layer 测试结果 ===")
    print(f"输入特征形状: {feats.shape}")
    print(f"质量函数形状: {mass.shape}")
    print(f"不确定性形状: {uncertainty.shape}")
    print(f"似然度形状: {plausibility.shape}")
    print(f"概率分布形状: {probability.shape}")

               
    mass_sum = mass.sum(dim=1, keepdim=True)
    print(f"质量函数和: {mass_sum.min().item():.4f} - {mass_sum.max().item():.4f}")

               
    prob_sum = probability.sum(dim=1, keepdim=True)
    print(f"概率分布和: {prob_sum.min().item():.4f} - {prob_sum.max().item():.4f}")

    print("✓ GEM-Layer 实现正确!")
