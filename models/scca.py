import torch
import torch.nn as nn


class Mlp2(nn.Module):
           

    def __init__(self, in_features: int, hidden_features: int, out_features: int, drop: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SCCA(nn.Module):
           

    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads

                                                     
        self.cross_attn_0_to_1 = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=False)
        self.cross_attn_1_to_0 = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=False)

                                                                                                 
        self.relation_judger = nn.Sequential(
            Mlp2(dim * 2, dim, dim),
            nn.Softmax(dim=-1),
        )

                                                                   
        self.k_noise = nn.Embedding(2, dim)
        self.v_noise = nn.Embedding(2, dim)

    @torch.no_grad()
    def _expand_noise_like(self, noise_vector: torch.Tensor, q_like: torch.Tensor) -> torch.Tensor:
                   
                                       
        return noise_vector.unsqueeze(0).unsqueeze(0).expand(q_like.size(0), q_like.size(1), -1)

    def forward(self, x0: torch.Tensor, x1: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, N, C = x0.shape
        assert x1.shape == (B, N, C)

        new_x0 = []
        new_x1 = []

        for b in range(B):
                    
            q_01 = x0[b].unsqueeze(0)             
            judge_in_01 = torch.cat([x0[b].unsqueeze(0), x1[b].unsqueeze(0)], dim=-1)              
            relation_01 = self.relation_judger(judge_in_01)                             

            noise_k_01 = self._expand_noise_like(self.k_noise.weight[0], q_01) + q_01
            noise_v_01 = self._expand_noise_like(self.v_noise.weight[0], q_01) + q_01

            k_01 = torch.cat([noise_k_01, q_01 * relation_01], dim=0)             
            v_01 = torch.cat([noise_v_01, x1[b].unsqueeze(0)], dim=0)              

            out_01, _ = self.cross_attn_0_to_1(q_01, k_01, v_01)
            new_x0.append(x0[b] + out_01.squeeze(0))

                    
            q_10 = x1[b].unsqueeze(0)             
            judge_in_10 = torch.cat([x1[b].unsqueeze(0), x0[b].unsqueeze(0)], dim=-1)
            relation_10 = self.relation_judger(judge_in_10)

            noise_k_10 = self._expand_noise_like(self.k_noise.weight[1], q_10) + q_10
            noise_v_10 = self._expand_noise_like(self.v_noise.weight[1], q_10) + q_10

            k_10 = torch.cat([noise_k_10, q_10 * relation_10], dim=0)
            v_10 = torch.cat([noise_v_10, x0[b].unsqueeze(0)], dim=0)

            out_10, _ = self.cross_attn_1_to_0(q_10, k_10, v_10)
            new_x1.append(x1[b] + out_10.squeeze(0))

        new_x0 = torch.stack(new_x0, dim=0)
        new_x1 = torch.stack(new_x1, dim=0)
        return new_x0, new_x1


__all__ = ["SCCA"]



if __name__ == "__main__":
                         
    torch.manual_seed(0)
    B, N, C = 2, 16, 64

    x0 = torch.randn(B, N, C)
    x1 = torch.randn(B, N, C)

    attn = SCCA(dim=C, num_heads=8, dropout=0.0)
    y0, y1 = attn(x0, x1)

    print("Input shapes:", x0.shape, x1.shape)
    print("Output shapes:", y0.shape, y1.shape)
