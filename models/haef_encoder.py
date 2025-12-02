import torch
import torch.nn as nn
import torch.nn.functional as F
from .conv_module import ConvModule
from .swin_transformer import SwinTransformer
from .scca import SCCA


class MLP(nn.Module):
           

    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


class SegFormerHead(nn.Module):
           

    def __init__(self, feature_strides=None, in_channels=128, embedding_dim=256, num_classes=20, **kwargs):
        super(SegFormerHead, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        assert len(feature_strides) == len(self.in_channels)
        assert min(feature_strides) == feature_strides[0]
        self.feature_strides = feature_strides

        (
            c1_in_channels,
            c2_in_channels,
            c3_in_channels,
            c4_in_channels,
        ) = self.in_channels

                                                   
                                                     

        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)
        self.dropout = nn.Dropout2d(0.1)

        self.linear_fuse = ConvModule(
            in_channels=embedding_dim * 4,
            out_channels=embedding_dim,
            kernel_size=1,
            norm_cfg=dict(type="BN", requires_grad=True),
        )

        self.linear_pred = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)

                              
        self.min_stride = min(feature_strides)

    def forward(self, x):
        c1, c2, c3, c4 = x

                                                       
        n, _, h, w = c4.shape

        _c4 = self.linear_c4(c4).permute(0, 2, 1).reshape(n, -1, c4.shape[2], c4.shape[3])
        _c4 = F.interpolate(_c4, size=c1.size()[2:], mode="bilinear", align_corners=False)

        _c3 = self.linear_c3(c3).permute(0, 2, 1).reshape(n, -1, c3.shape[2], c3.shape[3])
        _c3 = F.interpolate(_c3, size=c1.size()[2:], mode="bilinear", align_corners=False)

        _c2 = self.linear_c2(c2).permute(0, 2, 1).reshape(n, -1, c2.shape[2], c2.shape[3])
        _c2 = F.interpolate(_c2, size=c1.size()[2:], mode="bilinear", align_corners=False)

        _c1 = self.linear_c1(c1).permute(0, 2, 1).reshape(n, -1, c1.shape[2], c1.shape[3])

        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))

                                                                           
        self.last_fused = _c

                                                            
        x = self.dropout(_c)
        x = self.linear_pred(x)

                             
                                                    
        x = F.interpolate(x, scale_factor=self.min_stride, mode="bilinear", align_corners=False)

        return x


class TransformerBackbone(nn.Module):
    def __init__(
        self,
        backbone: str,
        train_backbone: bool,
        return_interm_layers: bool,
        drop_path_rate,
        pretrained_backbone_path,
    ):
        super().__init__()
        out_indices = (0, 1, 2, 3)
        if backbone == "swin_tiny":
            backbone = SwinTransformer(
                embed_dim=96,
                depths=[2, 2, 6, 2],
                num_heads=[3, 6, 12, 24],
                window_size=7,
                ape=False,
                drop_path_rate=drop_path_rate,
                patch_norm=True,
                use_checkpoint=False,
                out_indices=out_indices,
            )
            embed_dim = 96
            backbone.init_weights(pretrained_backbone_path)
        elif backbone == "swin_small":
            backbone = SwinTransformer(
                embed_dim=96,
                depths=[2, 2, 18, 2],
                num_heads=[3, 6, 12, 24],
                window_size=7,
                ape=False,
                drop_path_rate=drop_path_rate,
                patch_norm=True,
                use_checkpoint=False,
                out_indices=out_indices,
            )
            embed_dim = 96
            backbone.init_weights(pretrained_backbone_path)
        elif backbone == "swin_large":
            backbone = SwinTransformer(
                embed_dim=192,
                depths=[2, 2, 18, 2],
                num_heads=[6, 12, 24, 48],
                window_size=7,
                ape=False,
                drop_path_rate=drop_path_rate,
                patch_norm=True,
                use_checkpoint=False,
                out_indices=out_indices,
            )
            embed_dim = 192
            backbone.init_weights(pretrained_backbone_path)
        elif backbone == "swin_large_window12":
            backbone = SwinTransformer(
                pretrain_img_size=384,
                embed_dim=192,
                depths=[2, 2, 18, 2],
                num_heads=[6, 12, 24, 48],
                window_size=12,
                ape=False,
                drop_path_rate=drop_path_rate,
                patch_norm=True,
                use_checkpoint=False,
                out_indices=out_indices,
            )
            embed_dim = 192
            backbone.init_weights(pretrained_backbone_path)
        elif backbone == "swin_large_window12_to_1k":
            backbone = SwinTransformer(
                pretrain_img_size=384,
                embed_dim=192,
                depths=[2, 2, 18, 2],
                num_heads=[6, 12, 24, 48],
                window_size=12,
                ape=False,
                drop_path_rate=drop_path_rate,
                patch_norm=True,
                use_checkpoint=False,
                out_indices=out_indices,
            )
            embed_dim = 192
            backbone.init_weights(pretrained_backbone_path)
        else:
            raise NotImplementedError

        for name, parameter in backbone.named_parameters():
                                       
            if not train_backbone:
                parameter.requires_grad_(False)

        if return_interm_layers:

            self.strides = [8, 16, 32]
            self.num_channels = [
                embed_dim * 2,
                embed_dim * 4,
                embed_dim * 8,
            ]
        else:
            self.strides = [32]
            self.num_channels = [embed_dim * 8]

        self.body = backbone

    def forward(self, input):
        xs = self.body(input)

        return xs


class WeTr(nn.Module):
    def __init__(
        self,
        backbone,
        num_classes=2,
        n_heads=8,
        dpr=0.1,
        drop_rate=0.0,
        num_parallel=None,
        fusion_params=None,
    ):
        super().__init__()
                                                       
        self.num_classes = num_classes
        self.embedding_dim = 256
        self.feature_strides = [4, 8, 16, 32]
        self.num_parallel = num_parallel                        
        self.backbone = backbone
        self.use_feature_fusion = False                            

        assert num_parallel is not None, "num_parallel must be provided"
        self.num_parallel = num_parallel

                                    
        if "swin" not in backbone:
            raise NotImplementedError("Only Swin backbones are supported; MiT branch has been removed.")

        if backbone == "swin_tiny":
            pretrained_backbone_path = "pretrained/swin_tiny_patch4_window7_224.pth"
            self.in_channels = [96, 192, 384, 768]
        elif backbone == "swin_small":
            pretrained_backbone_path = "pretrained/swin_small_patch4_window7_224.pth"
            self.in_channels = [96, 192, 384, 768]
        elif backbone == "swin_large_window12":
            pretrained_backbone_path = "pretrained/swin_large_patch4_window12_384_22k.pth"
            self.in_channels = [192, 384, 768, 1536]
        elif backbone == "swin_large_window12_to_1k":
            pretrained_backbone_path = "pretrained/swin_large_patch4_window12_384_22kto1k.pth"
            self.in_channels = [192, 384, 768, 1536]
        else:
            assert backbone == "swin_large"
            pretrained_backbone_path = "pretrained/swin_large_patch4_window7_224_22k.pth"
            self.in_channels = [192, 384, 768, 1536]

        self.encoder = TransformerBackbone(backbone, True, True, dpr, pretrained_backbone_path)

                                  
        self.decoder = SegFormerHead(
            feature_strides=self.feature_strides,
            in_channels=self.in_channels,
            embedding_dim=self.embedding_dim,
            num_classes=self.num_classes,
        )

                                                                      
        self.pca_stages = nn.ModuleList([SCCA(dim=c, num_heads=n_heads, dropout=drop_rate) for c in self.in_channels])

    def get_param_groups(self):
                           
        param_groups = [[], [], []]                          

                                   
        for name, param in list(self.encoder.named_parameters()):
            if "norm" in name:
                param_groups[1].append(param)                            
            else:
                param_groups[0].append(param)                            

                                   
        for param in list(self.decoder.parameters()):
            param_groups[2].append(param)

                         
        for param in self.pca_stages.parameters():
            param_groups[2].append(param)

        return param_groups

    def forward(self, x):
                  
        original_shape = x[0].shape[2:]

                        
        assert len(x) == self.num_parallel, f"Expected {self.num_parallel} modalities, got {len(x)}"

                                                                                                
                                                                                             
        modality_features = self.encoder(x)

                                                            
        fused_features = []
        num_modalities = len(modality_features)
        num_stages = len(modality_features[0]) if num_modalities > 0 else 0

        for s in range(num_stages):
            stage_feats = [modality_features[m][s] for m in range(num_modalities)]                

            B, C, H, W = stage_feats[0].shape
            seqs = [f.permute(0, 2, 3, 1).reshape(B, H * W, C) for f in stage_feats]             

                                                                                                 
            updated_seqs = seqs
            for m in range(num_modalities):
                nxt = (m + 1) % num_modalities
                y_m, y_n = self.pca_stages[s](updated_seqs[m], updated_seqs[nxt])
                updated_seqs[m], updated_seqs[nxt] = y_m, y_n

                                                 
            avg_seq = updated_seqs[0]
            for i in range(1, num_modalities):
                avg_seq = avg_seq + updated_seqs[i]
            avg_seq = avg_seq / float(num_modalities)

            fused = avg_seq.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()                
            fused_features.append(fused)

                     
        output = self.decoder(fused_features)

        if output.shape[2:] != original_shape:
            output = F.interpolate(output, size=original_shape, mode="bilinear", align_corners=False)

        return [output], None
