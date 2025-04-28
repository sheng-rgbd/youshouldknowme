import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt

# 📸 新增：可視化工具函式
def save_attention_map(tensor, filename):
    # 假設 tensor shape: (B, C, H, W)，取第一個 batch 和第一個 channel
    attention_map = tensor[0, 0].detach().cpu().numpy()
    plt.imshow(attention_map, cmap='viridis')
    plt.colorbar()
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename)
    plt.close()

# SW-MAM 模組
class SW_MAM(nn.Module):
    def __init__(self, in_channels):
        super(SW_MAM, self).__init__()
        self.spatial_conv = nn.Conv2d(in_channels, 1, kernel_size=7, padding=3)
        self.channel_fc1 = nn.Linear(in_channels, in_channels // 4)
        self.channel_fc2 = nn.Linear(in_channels // 4, in_channels)

    def forward(self, x):
        b, c, h, w = x.size()
        avg_pool = F.adaptive_avg_pool2d(x, 1).view(b, c)
        max_pool = F.adaptive_max_pool2d(x, 1).view(b, c)

        avg_out = self.channel_fc2(F.relu(self.channel_fc1(avg_pool)))
        max_out = self.channel_fc2(F.relu(self.channel_fc1(max_pool)))
        channel_attn = torch.sigmoid(avg_out + max_out).view(b, c, 1, 1)

        spatial_attn = torch.sigmoid(self.spatial_conv(x))

        out = x * channel_attn * spatial_attn

        # 📸 可視化保存 SW-MAM attention map
        save_attention_map(spatial_attn, './attention_maps/sw_mam_attention.png')

        return out

# Coordinate Attention (CA) 模組
class CoordinateAttention(nn.Module):
    def __init__(self, in_channels, reduction=32):
        super(CoordinateAttention, self).__init__()
        reduced_channels = max(8, in_channels // reduction)
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.conv1 = nn.Conv2d(in_channels, reduced_channels, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(reduced_channels)
        self.act = nn.ReLU()
        self.conv_h = nn.Conv2d(reduced_channels, in_channels, kernel_size=1)
        self.conv_w = nn.Conv2d(reduced_channels, in_channels, kernel_size=1)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()

        x_h = self.pool_h(x).permute(0, 1, 3, 2)  # (N, C, H, 1)
        x_w = self.pool_w(x)                     # (N, C, W, 1)

        y = torch.cat([x_h, x_w], dim=3)
        y = self.act(self.bn1(self.conv1(y)))

        x_h, x_w = torch.split(y, [h, w], dim=3)

        x_h = self.conv_h(x_h).permute(0, 1, 3, 2)
        x_w = self.conv_w(x_w)

        out = identity * torch.sigmoid(x_h) * torch.sigmoid(x_w)

        # 📸 可視化保存 CA attention map
        save_attention_map(torch.sigmoid(x_h), './attention_maps/coord_att_h.png')
        save_attention_map(torch.sigmoid(x_w), './attention_maps/coord_att_w.png')

        return out

# PHA 並行混合注意力模組
class ParallelHybridAttention(nn.Module):
    def __init__(self, in_channels):
        super(ParallelHybridAttention, self).__init__()
        self.sw_mam = SW_MAM(in_channels)
        self.coord_att = CoordinateAttention(in_channels)

        self.fusion = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        sw_out = self.sw_mam(x)
        ca_out = self.coord_att(x)
        fused = sw_out + ca_out
        out = self.fusion(fused)
        return out
