import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, t_dim: int):
        super(ResBlock, self).__init__()
        self.norm1 = nn.GroupNorm(32, num_channels=in_ch)
        self.norm2 = nn.GroupNorm(32, num_channels=out_ch)
        self.silu = nn.SiLU()
        self.dropout = nn.Dropout(p=0.1)

        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

        self.time_proj = nn.Linear(t_dim, out_ch)
        self.shortcut = (
            nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, 1)
        )

    def forward(self, x, t_emb):
        """
        input:
        x: (B, in_channels, H, W)
        t_emb: (B, t_dim)

        output:
        h: (B, out_channels, H, W)
        """

        h = self.conv1(self.silu(self.norm1(x)))
        h = h + self.time_proj(t_emb)[:, :, None, None]
        h = self.conv2(self.dropout(self.silu(self.norm2(h))))
        h = self.shortcut(x) + h

        return h


class Downsample(nn.Module):
    def __init__(self, ch: int):
        super(Downsample, self).__init__()
        self.conv = nn.Conv2d(ch, ch, 3, padding=1, stride=2)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, ch: int):
        super(Upsample, self).__init__()
        self.conv = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x):
        h = F.interpolate(x, scale_factor=2, mode="nearest")
        h = self.conv(h)
        return h


class SelfAttention(nn.Module):
    def __init__(self, in_ch: int):
        super(SelfAttention, self).__init__()
        self.norm = nn.GroupNorm(32, num_channels=in_ch)
        self.q = nn.Conv2d(in_ch, in_ch, 1)
        self.k = nn.Conv2d(in_ch, in_ch, 1)
        self.v = nn.Conv2d(in_ch, in_ch, 1)
        self.proj = nn.Conv2d(in_ch, in_ch, 1)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x):
        """
        input:
        x: (B, C, H, W)

        output:
        x: (B, C, H, W)
        """

        B, C, H, W = x.shape
        T = H * W

        h = self.norm(x)
        q = self.q(h).reshape(B, C, T).transpose(1, 2)  # (B, T, C)
        k = self.k(h).reshape(B, C, T)  # (B, C, T)
        v = self.v(h).reshape(B, C, T).transpose(1, 2)  # (B, T, C)

        scores = torch.matmul(q, k) / math.sqrt(C)  # (B, T, T)
        attn = torch.softmax(scores, dim=-1)  # (B, T, T)
        z = torch.matmul(attn, v).transpose(1, 2)  # (B, C, T)

        z = z.reshape(B, C, H, W)  # (B, C, H, W)
        out = x + self.proj(z)  # (B, C, H, W)

        return out


class UNet(nn.Module):
    def __init__(
        self,
        img_ch: int,
        t_dim: int,
        ch_base: int = 128,
        ch_multi: list = [1, 2, 2, 2],
        num_resblocks: int = 2,
    ):
        super(UNet, self).__init__()
        self.ch_base = ch_base
        self.ch_mult = ch_multi
        self.num_resolution = len(ch_multi)
        self.num_resblocks = num_resblocks
        num_ch = [ch_base * m for m in ch_multi]

        self.conv_in = nn.Conv2d(img_ch, num_ch[0], 3, padding=1)

        # Down
        hs_ch = []
        self.downblocks = nn.ModuleList()
        self.down_attn = nn.ModuleDict()
        self.downsamples = nn.ModuleList()
        in_ch = num_ch[0]
        hs_ch.append(in_ch)
        curr_res = 32
        for i in range(self.num_resolution):
            out_ch = num_ch[i]
            blocks = nn.ModuleList()
            for j in range(self.num_resblocks):
                blocks.append(ResBlock(in_ch, out_ch, t_dim))
                if curr_res == 16:
                    self.down_attn[f"d_{i}_{j}"] = SelfAttention(out_ch)
                hs_ch.append(out_ch)
                in_ch = out_ch
            self.downblocks.append(blocks)

            if i != self.num_resolution - 1:
                self.downsamples.append(Downsample(in_ch))
                hs_ch.append(in_ch)
                curr_res = curr_res // 2

        # Middle
        self.mid1 = ResBlock(in_ch, in_ch, t_dim)
        self.mid_attn = SelfAttention(in_ch)
        self.mid2 = ResBlock(in_ch, in_ch, t_dim)

        # Up
        self.upblocks = nn.ModuleList()
        self.up_attn = nn.ModuleDict()
        self.upsamples = nn.ModuleList()
        curr_res = 32 // (2 ** (self.num_resolution - 1))
        for i in reversed(range(self.num_resolution)):
            out_ch = num_ch[i]
            blocks = nn.ModuleList()
            for j in range(self.num_resblocks + 1):
                skip_ch = hs_ch.pop()
                blocks.append(ResBlock(in_ch + skip_ch, out_ch, t_dim))
                if curr_res == 16:
                    self.up_attn[f"u_{self.num_resolution - 1 - i}_{j}"] = (
                        SelfAttention(out_ch)
                    )
                in_ch = out_ch
            self.upblocks.append(blocks)

            if i != 0:
                self.upsamples.append(Upsample(in_ch))
                curr_res = curr_res * 2

        self.norm = nn.GroupNorm(32, num_ch[0])
        self.silu = nn.SiLU()
        self.conv_out = nn.Conv2d(num_ch[0], img_ch, 3, padding=1)

    def forward(self, x, t_emb):
        h = self.conv_in(x)
        hs = [h]

        # down
        for i in range(self.num_resolution):
            for j, block in enumerate(self.downblocks[i]):
                h = block(h, t_emb)
                key = f"d_{i}_{j}"
                if key in self.down_attn:
                    h = self.down_attn[key](h)
                hs.append(h)

            if i != self.num_resolution - 1:
                h = self.downsamples[i](h)
                hs.append(h)

        # midle
        h = self.mid1(h, t_emb)
        h = self.mid_attn(h)
        h = self.mid2(h, t_emb)

        # up
        for i in range(self.num_resolution):
            for j, block in enumerate(self.upblocks[i]):
                skip = hs.pop()
                h = torch.cat([h, skip], dim=1)
                h = block(h, t_emb)
                key = f"u_{i}_{j}"
                if key in self.up_attn:
                    h = self.up_attn[key](h)

            if i != self.num_resolution - 1:
                h = self.upsamples[i](h)

        h = self.conv_out(self.silu(self.norm(h)))

        return h


class DDPM(nn.Module):
    def __init__(
        self,
        img_ch: int,
        ch_base: int = 128,
        ch_multi: list = [1, 2, 2, 2],
        num_resblocks: int = 2,
    ):
        super(DDPM, self).__init__()
        self.t_dim = ch_base
        self.temb_dim = 4 * ch_base
        self.temb_mlp = nn.Sequential(
            nn.Linear(self.t_dim, self.temb_dim),
            nn.SiLU(),
            nn.Linear(self.temb_dim, self.temb_dim),
        )
        self.unet = UNet(img_ch, self.temb_dim, ch_base, ch_multi, num_resblocks)

    def get_t_emb(self, t):
        """
        input:
        t: (B,)

        output:
        emb: (B, t_dim)
        """

        assert self.t_dim % 2 == 0
        half = self.t_dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half) / (half - 1)).to(
            t.device
        )  # (half,)
        args = t.float()[:, None] * freqs[None]  # (B, half)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
        return emb

    def forward(self, x, t):
        """
        input:
        x: (B, C, H, W)

        output:
        eps: (B, C, H, W)
        """

        t_emb = self.temb_mlp(self.get_t_emb(t))
        eps = self.unet(x, t_emb)
        return eps
