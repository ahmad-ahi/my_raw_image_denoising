"""
NAFNet: Nonlinear Activation Free Network for Image Restoration
Simplified implementation for RAW image denoising (4-channel input/output)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm2d(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        # x: [B, C, H, W]
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class SimpleGate(nn.Module):
    """Replaces activation function: splits channels and multiplies."""
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):
    def __init__(self, c, dw_expand=2, ffn_expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * dw_expand

        # Depthwise conv attention branch
        self.conv1 = nn.Conv2d(c, dw_channel, 1)
        self.conv2 = nn.Conv2d(dw_channel, dw_channel, 3, 1, 1, groups=dw_channel)
        self.conv3 = nn.Conv2d(dw_channel // 2, c, 1)

        # Channel attention
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dw_channel // 2, c // 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c // 2, dw_channel // 2, 1),
            nn.Sigmoid()
        )

        # FFN
        ffn_channel = ffn_expand * c
        self.conv4 = nn.Conv2d(c, ffn_channel, 1)
        self.conv5 = nn.Conv2d(ffn_channel // 2, c, 1)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        self.gate = SimpleGate()

        self.beta = nn.Parameter(torch.ones(1, c, 1, 1) * 1e-2)
        self.gamma = nn.Parameter(torch.ones(1, c, 1, 1) * 1e-2)

        self.dropout = nn.Dropout(drop_out_rate) if drop_out_rate > 0 else nn.Identity()

    def forward(self, inp):
        x = inp
        x = self.norm1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.gate(x)
        x = x * self.se(x)
        x = self.conv3(x)
        x = self.dropout(x)
        y = inp + x * self.beta

        x = self.norm2(y)
        x = self.conv4(x)
        x = self.gate(x)
        x = self.conv5(x)
        x = self.dropout(x)
        return y + x * self.gamma


class NAFNet(nn.Module):
    def __init__(self, in_nc=4, out_nc=4, width=32,
                 enc_blocks=[2, 2, 4, 8],
                 dec_blocks=[2, 2, 2, 2],
                 drop_out_rate=0.):
        super().__init__()

        self.intro = nn.Conv2d(in_nc, width, 3, 1, 1)
        self.ending = nn.Conv2d(width, out_nc, 3, 1, 1)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width
        for num in enc_blocks:
            self.encoders.append(
                nn.Sequential(*[NAFBlock(chan) for _ in range(num)])
            )
            self.downs.append(nn.Conv2d(chan, chan * 2, 2, 2))
            chan *= 2

        self.middle_blks = nn.Sequential(*[NAFBlock(chan) for _ in range(4)])

        for num in dec_blocks:
            self.ups.append(nn.Sequential(
                nn.Conv2d(chan, chan * 2, 1),
                nn.PixelShuffle(2)
            ))
            chan //= 2
            self.decoders.append(
                nn.Sequential(*[NAFBlock(chan) for _ in range(num)])
            )

        self.padder_size = 2 ** len(self.encoders)

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        return F.pad(x, (0, mod_pad_w, 0, mod_pad_h))

    def forward(self, inp):
        h, w = inp.shape[2:]
        inp = self.check_image_size(inp)

        x = self.intro(inp)
        encs = []

        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        x = self.middle_blks(x)

        for decoder, up, enc_skip in zip(self.decoders, self.ups, reversed(encs)):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)

        x = self.ending(x)
        x = x + inp  # residual connection
        return x[:, :, :h, :w]
