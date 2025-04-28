import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from functools import partial
from timm.models.layers import trunc_normal_

# ========= ChannelEmbed =========
class ChannelEmbed(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=1, norm_layer=nn.BatchNorm2d):
        super(ChannelEmbed, self).__init__()
        self.out_channels = out_channels
        self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.channel_embed = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // reduction, kernel_size=1, bias=True),
            nn.Conv2d(out_channels // reduction, out_channels // reduction, kernel_size=3, stride=1, padding=1, bias=True, groups=out_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // reduction, out_channels, kernel_size=1, bias=True),
            norm_layer(out_channels)
        )
        self.norm = norm_layer(out_channels)

    def forward(self, x, H, W):
        B, N, _C = x.shape
        x = x.permute(0, 2, 1).reshape(B, _C, H, W).contiguous()
        residual = self.residual(x)
        x = self.channel_embed(x)
        out = self.norm(residual + x)
        return out

# ========= Dynamic CrossAttention (直接取代舊版) =========
class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None):
        super(CrossAttention, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.kv1 = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.kv2 = nn.Linear(dim, dim * 2, bias=qkv_bias)

        self.gate_mlp1 = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, dim),
            nn.Sigmoid()
        )
        self.gate_mlp2 = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, dim),
            nn.Sigmoid()
        )

    def forward(self, x1, x2):
        B, N, C = x1.shape

        q1 = x1.reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        q2 = x2.reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        k1, v1 = self.kv1(x1).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k2, v2 = self.kv2(x2).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        gate1 = self.gate_mlp1(x1).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # 正確
        gate2 = self.gate_mlp2(x2).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # 正確

        ctx1 = (k1.transpose(-2, -1) @ (v1 * gate1)) * self.scale
        ctx1 = ctx1.softmax(dim=-2)
        ctx2 = (k2.transpose(-2, -1) @ (v2 * gate2)) * self.scale
        ctx2 = ctx2.softmax(dim=-2)

        x1 = (q1 @ ctx2).permute(0, 2, 1, 3).reshape(B, N, C)
        x2 = (q2 @ ctx1).permute(0, 2, 1, 3).reshape(B, N, C)

        return x1, x2

# ========= CrossPath (直接呼叫新的 CrossAttention) =========
class CrossPath(nn.Module):
    def __init__(self, dim, reduction=1, num_heads=None, norm_layer=nn.LayerNorm):
        super().__init__()
        self.channel_proj1 = nn.Linear(dim, dim // reduction * 2)
        self.channel_proj2 = nn.Linear(dim, dim // reduction * 2)
        self.act1 = nn.ReLU(inplace=True)
        self.act2 = nn.ReLU(inplace=True)
        self.cross_attn = CrossAttention(dim // reduction, num_heads=num_heads)
        self.end_proj1 = nn.Linear(dim // reduction * 2, dim)
        self.end_proj2 = nn.Linear(dim // reduction * 2, dim)
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)

    def forward(self, x1, x2):
        y1, u1 = self.act1(self.channel_proj1(x1)).chunk(2, dim=-1)
        y2, u2 = self.act2(self.channel_proj2(x2)).chunk(2, dim=-1)

        v1, v2 = self.cross_attn(u1, u2)

        y1 = torch.cat((y1, v1), dim=-1)
        y2 = torch.cat((y2, v2), dim=-1)

        out_x1 = self.norm1(x1 + self.end_proj1(y1))
        out_x2 = self.norm2(x2 + self.end_proj2(y2))

        return out_x1, out_x2

# ========= FeatureFusionModule (舊版融合) =========
class FeatureFusionModule(nn.Module):
    def __init__(self, dim, reduction=1, num_heads=None, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.cross = CrossPath(dim=dim, reduction=reduction, num_heads=num_heads)
        self.channel_emb = ChannelEmbed(in_channels=dim * 2, out_channels=dim, reduction=reduction, norm_layer=norm_layer)

    def forward(self, x1, x2):
        B, C, H, W = x1.shape
        x1 = x1.flatten(2).transpose(1, 2)
        x2 = x2.flatten(2).transpose(1, 2)
        x1, x2 = self.cross(x1, x2)
        merge = torch.cat((x1, x2), dim=-1)
        merge = self.channel_emb(merge, H, W)
        return merge

# ========= FeatureRectifyModule =========
class FeatureRectifyModule(nn.Module):
    def __init__(self, dim, reduction=1, lambda_c=0.5, lambda_s=0.5):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(dim * 2, dim // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // reduction, 2, 1),
            nn.Sigmoid())
        self.channel_mlp = nn.Sequential(
            nn.Linear(dim * 4, dim * 2 // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(dim * 2 // reduction, dim * 2),
            nn.Sigmoid())
        self.lambda_c = lambda_c
        self.lambda_s = lambda_s

    def forward(self, x1, x2):
        B, C, H, W = x1.shape
        concat = torch.cat([x1, x2], dim=1)
        spatial = self.spatial_conv(concat).view(B, 2, 1, H, W).permute(1, 0, 2, 3, 4)
        avg = self.avg_pool(concat).view(B, -1)
        max = self.max_pool(concat).view(B, -1)
        y = self.channel_mlp(torch.cat([avg, max], dim=1)).view(B, 2, C, 1, 1).permute(1, 0, 2, 3, 4)
        out1 = x1 + self.lambda_c * y[1] * x2 + self.lambda_s * spatial[1] * x2
        out2 = x2 + self.lambda_c * y[0] * x1 + self.lambda_s * spatial[0] * x1
        return out1, out2

# ========= CoordAtt =========
class CoordAtt(nn.Module):
    def __init__(self, inp, reduction=32):
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        mip = max(8, inp // reduction)
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.ReLU(inplace=True)
        self.conv_h = nn.Conv2d(mip, inp, kernel_size=1)
        self.conv_w = nn.Conv2d(mip, inp, kernel_size=1)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        return identity * a_w * a_h

# ========= ShiftViTBlockv2 =========
class ShiftViTBlockv2(nn.Module):
    def __init__(self, dim, n_div=12, ratio=4., act_layer=nn.LeakyReLU, norm_layer=nn.BatchNorm2d, input_resolution=(64, 64)):
        super().__init__()
        self.dim = dim
        self.norm2 = norm_layer(dim)
        hidden_dim = int(dim * ratio)
        self.channel = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, hidden_dim, 1),
            act_layer(),
            nn.Conv2d(hidden_dim, dim, 1),
            nn.Sigmoid())
        self.n_div = n_div

    def forward(self, x):
        B, C, H, W = x.shape
        g = C // self.n_div
        out = torch.zeros_like(x)
        out[:, g*0:g*1, :, :-10] = x[:, g*0:g*1, :, 10:]
        out[:, g*1:g*2, :, 10:] = x[:, g*1:g*2, :, :-10]
        out[:, g*2:g*3, :-10, :] = x[:, g*2:g*3, 10:, :]
        out[:, g*3:g*4, 10:, :] = x[:, g*3:g*4, :-10, :]
        out[:, g*4:, :, :] = x[:, g*4:, :, :]
        x = out
        x = x + x * self.channel(self.norm2(x))
        return x

# ========= PHA =========
class PHA(nn.Module):
    def __init__(self, dim, out=None, input_resolution=(64, 64), n_div=12, ratio=4., act_layer=nn.LeakyReLU, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.coord_att = CoordAtt(inp=dim)
        self.shift_vit = ShiftViTBlockv2(dim=dim, n_div=n_div, ratio=ratio, act_layer=act_layer, norm_layer=norm_layer, input_resolution=input_resolution)
        self.norm2 = norm_layer(dim)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, int(dim * ratio), kernel_size=1),
            act_layer(),
            nn.Conv2d(int(dim * ratio), dim, kernel_size=1))
        self.out = nn.Conv2d(dim, out, 1) if out else nn.Identity()

    def forward(self, x):
        x_norm = self.norm1(x)
        coord_out = self.coord_att(x_norm)
        shift_out = self.shift_vit(x_norm)
        add1 = x + coord_out + shift_out
        norm_out = self.norm2(add1)
        mlp_out = self.mlp(norm_out)
        add2 = add1 + mlp_out
        return self.out(add2)

# ========= FeatureFusionPHA (升級版 FFM) =========
class FeatureFusionPHA(nn.Module):
    def __init__(self, dim, reduction=1, num_heads=None, norm_layer=nn.BatchNorm2d, input_resolution=(64, 64)):
        super().__init__()
        self.pha_rgb = PHA(dim=dim, out=dim, input_resolution=input_resolution)
        self.pha_x = PHA(dim=dim, out=dim, input_resolution=input_resolution)
        self.cross = CrossPath(dim=dim, reduction=reduction, num_heads=num_heads)
        self.channel_emb = ChannelEmbed(in_channels=dim * 2, out_channels=dim, reduction=reduction, norm_layer=norm_layer)

    def forward(self, x1, x2):
        B, C, H, W = x1.shape
        x1 = self.pha_rgb(x1)
        x2 = self.pha_x(x2)
        x1 = x1.flatten(2).transpose(1, 2)
        x2 = x2.flatten(2).transpose(1, 2)
        x1, x2 = self.cross(x1, x2)
        merge = torch.cat((x1, x2), dim=-1)
        merge = self.channel_emb(merge, H, W)
        return merge




# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from functools import partial
# import math
# from timm.models.layers import trunc_normal_

# # ========== 原始 FFM 模組 (保留) ==========
# class ChannelEmbed(nn.Module):
#     def __init__(self, in_channels, out_channels, reduction=1, norm_layer=nn.BatchNorm2d):
#         super(ChannelEmbed, self).__init__()
#         self.out_channels = out_channels
#         self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
#         self.channel_embed = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels // reduction, kernel_size=1, bias=True),
#             nn.Conv2d(out_channels // reduction, out_channels // reduction, kernel_size=3, stride=1, padding=1, bias=True, groups=out_channels // reduction),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(out_channels // reduction, out_channels, kernel_size=1, bias=True),
#             norm_layer(out_channels)
#         )
#         self.norm = norm_layer(out_channels)

#     def forward(self, x, H, W):
#         B, N, _C = x.shape
#         x = x.permute(0, 2, 1).reshape(B, _C, H, W).contiguous()
#         residual = self.residual(x)
#         x = self.channel_embed(x)
#         out = self.norm(residual + x)
#         return out

# class CrossAttention(nn.Module):
#     def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None):
#         super(CrossAttention, self).__init__()
#         self.dim = dim
#         self.num_heads = num_heads
#         head_dim = dim // num_heads
#         self.scale = qk_scale or head_dim ** -0.5
#         self.kv1 = nn.Linear(dim, dim * 2, bias=qkv_bias)
#         self.kv2 = nn.Linear(dim, dim * 2, bias=qkv_bias)

#     def forward(self, x1, x2):
#         B, N, C = x1.shape
#         q1 = x1.reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
#         q2 = x2.reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
#         k1, v1 = self.kv1(x1).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
#         k2, v2 = self.kv2(x2).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

#         ctx1 = (k1.transpose(-2, -1) @ v1) * self.scale
#         ctx1 = ctx1.softmax(dim=-2)
#         ctx2 = (k2.transpose(-2, -1) @ v2) * self.scale
#         ctx2 = ctx2.softmax(dim=-2)

#         x1 = (q1 @ ctx2).permute(0, 2, 1, 3).reshape(B, N, C)
#         x2 = (q2 @ ctx1).permute(0, 2, 1, 3).reshape(B, N, C)
#         return x1, x2

# class CrossPath(nn.Module):
#     def __init__(self, dim, reduction=1, num_heads=None, norm_layer=nn.LayerNorm):
#         super().__init__()
#         self.channel_proj1 = nn.Linear(dim, dim // reduction * 2)
#         self.channel_proj2 = nn.Linear(dim, dim // reduction * 2)
#         self.act1 = nn.ReLU(inplace=True)
#         self.act2 = nn.ReLU(inplace=True)
#         self.cross_attn = CrossAttention(dim // reduction, num_heads=num_heads)
#         self.end_proj1 = nn.Linear(dim // reduction * 2, dim)
#         self.end_proj2 = nn.Linear(dim // reduction * 2, dim)
#         self.norm1 = norm_layer(dim)
#         self.norm2 = norm_layer(dim)

#     def forward(self, x1, x2):
#         y1, u1 = self.act1(self.channel_proj1(x1)).chunk(2, dim=-1)
#         y2, u2 = self.act2(self.channel_proj2(x2)).chunk(2, dim=-1)
#         v1, v2 = self.cross_attn(u1, u2)
#         y1 = torch.cat((y1, v1), dim=-1)
#         y2 = torch.cat((y2, v2), dim=-1)
#         out_x1 = self.norm1(x1 + self.end_proj1(y1))
#         out_x2 = self.norm2(x2 + self.end_proj2(y2))
#         return out_x1, out_x2

# class FeatureFusionModule(nn.Module):
#     def __init__(self, dim, reduction=1, num_heads=None, norm_layer=nn.BatchNorm2d):
#         super().__init__()
#         self.cross = CrossPath(dim=dim, reduction=reduction, num_heads=num_heads)
#         self.channel_emb = ChannelEmbed(in_channels=dim * 2, out_channels=dim, reduction=reduction, norm_layer=norm_layer)

#     def forward(self, x1, x2):
#         B, C, H, W = x1.shape
#         x1 = x1.flatten(2).transpose(1, 2)
#         x2 = x2.flatten(2).transpose(1, 2)
#         x1, x2 = self.cross(x1, x2)
#         merge = torch.cat((x1, x2), dim=-1)
#         merge = self.channel_emb(merge, H, W)
#         return merge

# # ========== FeatureRectifyModule (補上遺失模組) ==========
# class FeatureRectifyModule(nn.Module):
#     def __init__(self, dim, reduction=1, lambda_c=0.5, lambda_s=0.5):
#         super().__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.max_pool = nn.AdaptiveMaxPool2d(1)
#         self.spatial_conv = nn.Sequential(
#             nn.Conv2d(dim * 2, dim // reduction, 1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(dim // reduction, 2, 1),
#             nn.Sigmoid())
#         self.channel_mlp = nn.Sequential(
#             nn.Linear(dim * 4, dim * 2 // reduction),
#             nn.ReLU(inplace=True),
#             nn.Linear(dim * 2 // reduction, dim * 2),
#             nn.Sigmoid())
#         self.lambda_c = lambda_c
#         self.lambda_s = lambda_s

#     def forward(self, x1, x2):
#         B, C, H, W = x1.shape
#         concat = torch.cat([x1, x2], dim=1)
#         # spatial
#         spatial = self.spatial_conv(concat).view(B, 2, 1, H, W).permute(1, 0, 2, 3, 4)
#         # channel
#         avg = self.avg_pool(concat).view(B, -1)
#         max = self.max_pool(concat).view(B, -1)
#         y = self.channel_mlp(torch.cat([avg, max], dim=1)).view(B, 2, C, 1, 1).permute(1, 0, 2, 3, 4)
#         # apply
#         out1 = x1 + self.lambda_c * y[1] * x2 + self.lambda_s * spatial[1] * x2
#         out2 = x2 + self.lambda_c * y[0] * x1 + self.lambda_s * spatial[0] * x1
#         return out1, out2

# # ========== ✅ PHA 與 FFMPHA ==========
# class CoordAtt(nn.Module):
#     def __init__(self, inp, reduction=32):
#         super().__init__()
#         self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
#         self.pool_w = nn.AdaptiveAvgPool2d((1, None))
#         mip = max(8, inp // reduction)
#         self.conv1 = nn.Conv2d(inp, mip, kernel_size=1)
#         self.bn1 = nn.BatchNorm2d(mip)
#         self.act = nn.ReLU(inplace=True)
#         self.conv_h = nn.Conv2d(mip, inp, kernel_size=1)
#         self.conv_w = nn.Conv2d(mip, inp, kernel_size=1)

#     def forward(self, x):
#         identity = x
#         n, c, h, w = x.size()
#         x_h = self.pool_h(x)
#         x_w = self.pool_w(x).permute(0, 1, 3, 2)
#         y = torch.cat([x_h, x_w], dim=2)
#         y = self.conv1(y)
#         y = self.bn1(y)
#         y = self.act(y)
#         x_h, x_w = torch.split(y, [h, w], dim=2)
#         x_w = x_w.permute(0, 1, 3, 2)
#         a_h = self.conv_h(x_h).sigmoid()
#         a_w = self.conv_w(x_w).sigmoid()
#         return identity * a_w * a_h

# class ShiftViTBlockv2(nn.Module):
#     def __init__(self, dim, n_div=12, ratio=4., act_layer=nn.LeakyReLU, norm_layer=nn.BatchNorm2d, input_resolution=(64, 64)):
#         super().__init__()
#         self.dim = dim
#         self.norm2 = norm_layer(dim)
#         hidden_dim = int(dim * ratio)
#         self.channel = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Conv2d(dim, hidden_dim, 1),
#             act_layer(),
#             nn.Conv2d(hidden_dim, dim, 1),
#             nn.Sigmoid())
#         self.n_div = n_div

#     def forward(self, x):
#         B, C, H, W = x.shape
#         g = C // self.n_div
#         out = torch.zeros_like(x)
#         out[:, g * 0:g * 1, :, :-10] = x[:, g * 0:g * 1, :, 10:]
#         out[:, g * 1:g * 2, :, 10:] = x[:, g * 1:g * 2, :, :-10]
#         out[:, g * 2:g * 3, :-10, :] = x[:, g * 2:g * 3, 10:, :]
#         out[:, g * 3:g * 4, 10:, :] = x[:, g * 3:g * 4, :-10, :]
#         out[:, g * 4:, :, :] = x[:, g * 4:, :, :]
#         x = out
#         x = x + x * self.channel(self.norm2(x))
#         return x

# class PHA(nn.Module):
#     def __init__(self, dim, out=None, input_resolution=(64, 64), n_div=12, ratio=4., act_layer=nn.LeakyReLU, norm_layer=nn.BatchNorm2d):
#         super().__init__()
#         self.norm1 = norm_layer(dim)
#         self.coord_att = CoordAtt(inp=dim)
#         self.shift_vit = ShiftViTBlockv2(dim=dim, n_div=n_div, ratio=ratio, act_layer=act_layer, norm_layer=norm_layer, input_resolution=input_resolution)
#         self.norm2 = norm_layer(dim)
#         self.mlp = nn.Sequential(
#             nn.Conv2d(dim, int(dim * ratio), kernel_size=1),
#             act_layer(),
#             nn.Conv2d(int(dim * ratio), dim, kernel_size=1))
#         self.out = nn.Conv2d(dim, out, 1) if out else nn.Identity()

#     def forward(self, x):
#         x_norm = self.norm1(x)
#         coord_out = self.coord_att(x_norm)
#         shift_out = self.shift_vit(x_norm)
#         add1 = x + coord_out + shift_out
#         norm_out = self.norm2(add1)
#         mlp_out = self.mlp(norm_out)
#         add2 = add1 + mlp_out
#         return self.out(add2)

# class FeatureFusionPHA(nn.Module):
#     def __init__(self, dim, reduction=1, num_heads=None, norm_layer=nn.BatchNorm2d, input_resolution=(64, 64)):
#         super().__init__()
#         self.pha_rgb = PHA(dim=dim, out=dim, input_resolution=input_resolution)  # ✅ RGB 分支前置 PHA
#         self.pha_x = PHA(dim=dim, out=dim, input_resolution=input_resolution)    # ✅ X 分支前置 PHA
#         self.cross = CrossPath(dim=dim, reduction=reduction, num_heads=num_heads)
#         self.channel_emb = ChannelEmbed(in_channels=dim*2, out_channels=dim, reduction=reduction, norm_layer=norm_layer)
#         self.apply(self._init_weights)

#     def _init_weights(self, m):
#         if isinstance(m, nn.Linear):
#             trunc_normal_(m.weight, std=.02)
#             if m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
#         elif isinstance(m, nn.LayerNorm):
#             nn.init.constant_(m.bias, 0)
#             nn.init.constant_(m.weight, 1.0)
#         elif isinstance(m, nn.Conv2d):
#             fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#             fan_out //= m.groups
#             m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
#             if m.bias is not None:
#                 m.bias.data.zero_()

#     def forward(self, x1, x2):
#         B, C, H, W = x1.shape

#         # ✅ 先對兩個模態分別進行 PHA 處理
#         x1 = self.pha_rgb(x1)
#         x2 = self.pha_x(x2)

#         # ✅ 接著做 Cross Attention 與融合
#         x1 = x1.flatten(2).transpose(1, 2)
#         x2 = x2.flatten(2).transpose(1, 2)
#         x1, x2 = self.cross(x1, x2)
#         merge = torch.cat((x1, x2), dim=-1)
#         merge = self.channel_emb(merge, H, W)
        
#         # print(f"[FFMPHA Output] shape: {merge.shape}")

#         return merge
    

