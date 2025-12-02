import torch.nn as nn


class ConvModule(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        bias=None,
        norm_cfg=None,
    ):
        super().__init__()

        use_bn = norm_cfg and norm_cfg.get("type", "").upper() == "BN"
        if bias is None:
            bias = not use_bn

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_channels) if use_bn else None
        self.act = nn.ReLU(inplace=True)

        if self.bn is not None and not norm_cfg.get("requires_grad", True):
            for p in self.bn.parameters():
                p.requires_grad = False

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        return self.act(x)
