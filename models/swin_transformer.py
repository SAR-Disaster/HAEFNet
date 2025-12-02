                                                          
                  
                              
                                                          
                                           
                                                          

import logging
import math
import re

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from .checkpoint import load_checkpoint
from .modules import (
    ModuleParallel,
    Additional_One_ModuleParallel,
    LayerNormParallel,
    Additional_Two_ModuleParallel,
    get_num_parallel,                                                         
)


class Mlp(nn.Module):
                                

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=ModuleParallel(nn.GELU()),
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = ModuleParallel(nn.Linear(in_features, hidden_features))
        self.act = act_layer
        self.fc2 = ModuleParallel(nn.Linear(hidden_features, out_features))
        self.drop = ModuleParallel(nn.Dropout(drop))

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
           
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
           
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
           

    def __init__(
        self,
        dim,
        window_size,
        num_heads,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):

        super().__init__()
        self.dim = dim
        self.window_size = window_size          
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

                                                            
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )                       

                                                                                
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))             
        coords_flatten = torch.flatten(coords, 1)            
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]                   
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()                   
        relative_coords[:, :, 0] += self.window_size[0] - 1                         
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)                
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
                   
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )                                                       

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1],
            -1,
        )                  
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()                    
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:

            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)

            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp_2(nn.Module):
                                

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SwinTransformerBlock(nn.Module):
           

    def __init__(
        self,
        dim,
        num_heads,
        window_size=7,
        shift_size=0,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=ModuleParallel(nn.GELU()),
        norm_layer=LayerNormParallel,
        layer_idx=0,           
        fusion_params=None,        
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.num_modalities = get_num_parallel()                            
        self.layer_idx = layer_idx         
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = ModuleParallel(nn.LayerNorm(dim))
        self.attn = Additional_One_ModuleParallel(
            WindowAttention(
                dim,
                window_size=to_2tuple(self.window_size),
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                proj_drop=drop,
            )
        )

        self.drop_path = ModuleParallel(DropPath(drop_path)) if drop_path > 0.0 else ModuleParallel(nn.Identity())
        self.norm2 = ModuleParallel(nn.LayerNorm(dim))
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

        self.H = None
        self.W = None
        self.fusion_params = None

                                                                                     
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, mask_matrix):
                   
        B, L, C = x[0].shape
        H, W = self.H, self.W
        assert L == H * W, "input feature has wrong size"

        shortcut = [x_.clone() for x_ in x]                 
        x_norm = self.norm1(x)

                        
        x_view = []
        for i in range(len(x_norm)):
            x_view.append(x_norm[i].view(B, H, W, C))

                                                      
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size

                             
        x_pad = []
        for i in range(len(x_view)):
            x_pad.append(F.pad(x_view[i], (0, 0, pad_l, pad_r, pad_t, pad_b)))

        _, Hp, Wp, _ = x_pad[0].shape

                      
        if self.shift_size > 0:
            shifted_x = [
                torch.roll(x_pad[i], shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
                for i in range(len(x_pad))
            ]
            attn_mask = mask_matrix
        else:
            shifted_x = x_pad
            attn_mask = None

                           
        x_windows = [
            window_partition(shifted_x[i], self.window_size) for i in range(len(shifted_x))
        ]                                     

        for i in range(len(x_windows)):
            x_windows[i] = x_windows[i].view(
                -1, self.window_size * self.window_size, C
            )                                    

                      
        attn_windows = self.attn(x_windows, attn_mask)                                    

                       
        for i in range(len(attn_windows)):
            attn_windows[i] = attn_windows[i].view(-1, self.window_size, self.window_size, C)

        shifted_x = [
            window_reverse(attn_windows[i], self.window_size, Hp, Wp) for i in range(len(attn_windows))
        ]             

                              
        if self.shift_size > 0:
            x_shift_back = [
                torch.roll(shifted_x[i], shifts=(self.shift_size, self.shift_size), dims=(1, 2))
                for i in range(len(shifted_x))
            ]
        else:
            x_shift_back = shifted_x

                              
        if pad_r > 0 or pad_b > 0:
            x_window = [x_shift_back[i][:, :H, :W, :].contiguous() for i in range(len(x_shift_back))]
        else:
            x_window = x_shift_back

                          
        x_window_reshape = [x_window[i].view(B, H * W, C) for i in range(len(x_window))]

                                                                              
        fused_features = x_window_reshape

                                       
                              
        drop_path_results = self.drop_path(fused_features)

                            
        x_updated = []
        for i in range(len(fused_features)):
                       
            x_updated.append(shortcut[i] + drop_path_results[i])

               
        x_norm2 = self.norm2(x_updated)
        x_mlp = self.mlp(x_norm2)

                                          
        drop_path_results2 = self.drop_path(x_mlp)

                        
        output = []
        for i in range(len(x_updated)):
            output.append(x_updated[i] + drop_path_results2[i])

        return output


class PatchMerging(nn.Module):
           

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x, H, W):
                   

        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)

                 
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))

        x0 = x[:, 0::2, 0::2, :]               
        x1 = x[:, 1::2, 0::2, :]               
        x2 = x[:, 0::2, 1::2, :]               
        x3 = x[:, 1::2, 1::2, :]               
        x = torch.cat([x0, x1, x2, x3], -1)                 
        x = x.view(B, -1, 4 * C)                 

        x = self.norm(x)
        x = self.reduction(x)

        return x


class BasicLayer(nn.Module):
           

    def __init__(
        self,
        dim,
        depth,
        num_heads,
        window_size=7,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        norm_layer=LayerNormParallel,
        downsample=None,
        use_checkpoint=False,
        layer_start_idx=0,             
    ):
        super().__init__()
        self.window_size = window_size
        self.shift_size = window_size // 2
        self.depth = depth
        self.use_checkpoint = use_checkpoint

                                         
        self.blocks = nn.ModuleList(
            [
                SwinTransformerBlock(
                    dim=dim,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=0 if (i % 2 == 0) else window_size // 2,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=(drop_path[i] if isinstance(drop_path, list) else drop_path),
                    norm_layer=norm_layer,
                    layer_idx=layer_start_idx + i,         
                )
                for i in range(depth)
            ]
        )

                             
        if downsample is not None:
            self.downsample = Additional_Two_ModuleParallel(downsample(dim=dim, norm_layer=nn.LayerNorm))
        else:
            self.downsample = None

    def forward(self, x, H, W):
                   

                                             
        Hp = int(np.ceil(H / self.window_size)) * self.window_size
        Wp = int(np.ceil(W / self.window_size)) * self.window_size
        img_mask = torch.zeros((1, Hp, Wp, 1), device=x[0].device)             
        h_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        w_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)                                   
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        for blk in self.blocks:
            blk.H, blk.W = H, W
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, attn_mask)
            else:
                x = blk(x, attn_mask)
        if self.downsample is not None:
            x_down = self.downsample(x, H, W)
            Wh, Ww = (H + 1) // 2, (W + 1) // 2
            return x, H, W, x_down, Wh, Ww
        else:
            return x, H, W, x, H, W


class PatchEmbed(nn.Module):
           

    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = ModuleParallel(nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size))
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
                               
                 
        _, _, H, W = x[0].size()
        if W % self.patch_size[1] != 0:
            x = F.pad(x, (0, self.patch_size[1] - W % self.patch_size[1]))
        if H % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[0] - H % self.patch_size[0]))

        x = self.proj(x)             
        if self.norm is not None:
            Wh, Ww = x[0].size(2), x[0].size(3)
            for i in range(len(x)):
                x[i] = x[i].flatten(2).transpose(1, 2)
            x = self.norm(x)
            for i in range(len(x)):
                x[i] = x[i].transpose(1, 2).view(-1, self.embed_dim, Wh, Ww)

        return x


class SwinTransformer(nn.Module):
           

    def __init__(
        self,
        pretrain_img_size=112,
        patch_size=4,
        in_chans=3,
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.2,
        norm_layer=LayerNormParallel,
        ape=False,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        frozen_stages=-1,
        use_checkpoint=False,
    ):
        super().__init__()
                        
        self.depths = depths                 
        self.num_heads = num_heads                    
        self.window_size = window_size                      
        self.mlp_ratio = mlp_ratio                    
        self.qkv_bias = qkv_bias                   
        self.qk_scale = qk_scale                   
        self.drop_rate = drop_rate      
        self.attn_drop_rate = attn_drop_rate                         
        self.norm_layer = norm_layer                     
        self.use_checkpoint = use_checkpoint                         
        self.drop_path_rate = drop_path_rate      
        self.pretrain_img_size = pretrain_img_size      
        self.num_layers = len(depths)      
        self.embed_dim = embed_dim      
        self.ape = ape      
        self.patch_norm = patch_norm      
        self.out_indices = out_indices      
        self.frozen_stages = frozen_stages      
        self.num_modalities = get_num_parallel()      

                      

                                                  
        self.patch_embed = PatchEmbed(
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None,
        )

                                     
        if self.ape:
            pretrain_img_size = to_2tuple(pretrain_img_size)
            patch_size = to_2tuple(patch_size)
            patches_resolution = [
                pretrain_img_size[0] // patch_size[0],
                pretrain_img_size[1] // patch_size[1],
            ]

            self.absolute_pos_embed = nn.Parameter(
                torch.zeros(1, embed_dim, patches_resolution[0], patches_resolution[1])
            )
            trunc_normal_(self.absolute_pos_embed, std=0.02)

        self.pos_drop = ModuleParallel(nn.Dropout(p=drop_rate))

                          
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]                               
        self.dpr = dpr              

                                           
        self.layers = nn.ModuleList()
        layer_start_idx = 0              
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(self.embed_dim * 2**i_layer),
                depth=self.depths[i_layer],
                num_heads=self.num_heads[i_layer],
                window_size=self.window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=self.qkv_bias,
                qk_scale=self.qk_scale,
                drop=self.drop_rate,
                attn_drop=self.attn_drop_rate,
                drop_path=self.dpr[sum(self.depths[:i_layer]) : sum(self.depths[: i_layer + 1])],
                norm_layer=self.norm_layer,
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=self.use_checkpoint,
                layer_start_idx=layer_start_idx,           
            )
            self.layers.append(layer)
            layer_start_idx += self.depths[i_layer]           

        num_features = [int(embed_dim * 2**i) for i in range(self.num_layers)]
        self.num_features = num_features

                                          
        for i_layer in self.out_indices:
            layer = self.norm_layer(num_features[i_layer])
            layer_name = f"norm{i_layer}"
            self.add_module(layer_name, layer)

        self._freeze_stages()

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False

        if self.frozen_stages >= 1 and self.ape:
            self.absolute_pos_embed.requires_grad = False

        if self.frozen_stages >= 2:
            self.pos_drop.eval()
            for i in range(0, self.frozen_stages - 1):
                m = self.layers[i]
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def init_weights(self, pretrained=None):
                   

        def _init_weights(m):
                                                                                     
            pass

        if pretrained is None:
            self.apply(_init_weights)
            return

        if not isinstance(pretrained, str):
            raise TypeError("pretrained must be a str or None")

        self.apply(_init_weights)
        logger = logging.getLogger(__name__)

                                                                    
        checkpoint = torch.load(pretrained, map_location="cpu")
        if "model" in checkpoint:
            checkpoint = checkpoint["model"]

        pretrained_state = checkpoint
        current_state = self.state_dict()
        mapped_state = {}

        for k in current_state.keys():
            candidate = None
                     
            if k in pretrained_state:
                candidate = pretrained_state[k]
                                                
            elif ".module." in k:
                plain = k.replace(".module.", ".")
                if plain in pretrained_state:
                    candidate = pretrained_state[plain]
                                                        
            elif ".layers." in k:
                base = re.sub(r"\.layers\.\d+", "", k)
                if base in pretrained_state:
                    candidate = pretrained_state[base]

            if candidate is not None and candidate.shape == current_state[k].shape:
                mapped_state[k] = candidate

                    
        current_state.update(mapped_state)
        self.load_state_dict(current_state, strict=False)

        missing = set(current_state.keys()) - set(mapped_state.keys())
        if missing:
            logger.warning(f"Loaded pretrained weights with {len(mapped_state)} matched tensors; "
                           f"{len(missing)} tensors remain randomly initialized.")

    def forward(self, x):
                               
        x = self.patch_embed(x)

        Wh, Ww = x[0].size(2), x[0].size(3)
        if self.ape:
                                                                          
            absolute_pos_embed = F.interpolate(self.absolute_pos_embed, size=(Wh, Ww), mode="bicubic")
            x = (x + absolute_pos_embed).flatten(2).transpose(1, 2)             
        else:
            for i in range(len(x)):
                x[i] = x[i].flatten(2).transpose(1, 2)
        x = self.pos_drop(x)

                                                        
        outs = {}
        for i in range(self.num_layers):
            layer = self.layers[i]
            x_out, H, W, x, Wh, Ww = layer(x, Wh, Ww)

            if i in self.out_indices:
                norm_layer = getattr(self, f"norm{i}")
                x_out = norm_layer(x_out)

                                                    
                out = []
                for j in range(self.num_modalities):
                                                                 
                    if j < len(x_out):
                                                          
                        modality_out = x_out[j].view(-1, H, W, self.num_features[i])
                        modality_out = modality_out.permute(0, 3, 1, 2).contiguous()
                        out.append(modality_out)
                    else:
                                                                              
                        print(f"Warning: Expected modality {j} not found in layer {i} output")
                                                                       
                        modality_out = x_out[0].view(-1, H, W, self.num_features[i])
                        modality_out = modality_out.permute(0, 3, 1, 2).contiguous()
                        out.append(modality_out)

                outs[i] = out

                                                               
                                                                                            
        modality_outputs = []
        for m in range(self.num_modalities):
            modality_stages = []
            for i in self.out_indices:
                if i in outs and m < len(outs[i]):
                    modality_stages.append(outs[i][m])
                else:
                                                                   
                                                                       
                    print(f"Warning: Missing output for modality {m}, stage {i}")
                    if len(outs) > 0 and len(outs[list(outs.keys())[0]]) > 0:
                                                                                  
                        first_stage = list(outs.keys())[0]
                        modality_stages.append(outs[first_stage][0].clone())

                                                 
            if modality_stages:
                modality_outputs.append(modality_stages)

        return modality_outputs

    def train(self, mode=True):
                                                                             
        super(SwinTransformer, self).train(mode)
        self._freeze_stages()
