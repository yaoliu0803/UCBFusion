# import python package
import numpy as np
import os
from PIL import Image
import cv2

# import external package
import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia

'''
# -----------------------------------------------------------
Configure our network
# -----------------------------------------------------------
'''


class DRFE_Module(nn.Module):
    def __init__(self):
        super(DRFE_Module, self).__init__()
        nb_filter = [1, 16, 128, 64, 32, 16, 1]
        kernel_size = 3
        stride = 1
        padding = 1

        # Extraction multiply feature
        self.inlayer1 = nn.Sequential(
            *[nn.Conv2d(nb_filter[0], nb_filter[1], 1, 1, 0), nn.LeakyReLU(inplace=True)])  # [1 -> 16]
        self.inlayer2 = nn.Sequential(
            *[nn.Conv2d(nb_filter[0], nb_filter[1], 1, 1, 0), nn.LeakyReLU(inplace=True)])  # [1 -> 16]
        self.denselayer1 = DenseBlock(nb_filter[1], kernel_size, stride, padding)  # [16 -> 64]
        self.denselayer2 = DenseBlock(nb_filter[1], kernel_size, stride, padding)  # [16 -> 64]
        self.sobelconv1 = Sobelxy(nb_filter[3])              # [64 -> 64]
        self.sobelconv2 = Sobelxy(nb_filter[3])              # [64 -> 64]
        self.re1_conv = nn.Sequential(
            *[nn.Conv2d(nb_filter[3], nb_filter[4], 1, 1, 0), nn.LeakyReLU(inplace=True)])  # [64 -> 32]
        self.re2_conv = nn.Sequential(
            *[nn.Conv2d(nb_filter[3], nb_filter[4], 1, 1, 0), nn.LeakyReLU(inplace=True)])  # [64 -> 32]
        ###################### Ablation studies #####################
        # self.inlayer1 = nn.Sequential(
        #     *[nn.Conv2d(nb_filter[0], nb_filter[1], 1, 1, 0), nn.LeakyReLU(inplace=True)])  # [1 -> 64]
        # self.inlayer2 = nn.Sequential(
        #     *[nn.Conv2d(nb_filter[0], nb_filter[1], 1, 1, 0), nn.LeakyReLU(inplace=True)])  # [1 -> 64]
        # self.cnnlayer1 = nn.Sequential(
        #     *[nn.Conv2d(nb_filter[1], nb_filter[3], 1, 1, 0), nn.LeakyReLU(inplace=True)])  # [16 -> 64]
        # self.cnnlayer2 = nn.Sequential(
        #     *[nn.Conv2d(nb_filter[1], nb_filter[3], 1, 1, 0), nn.LeakyReLU(inplace=True)])  # [16 -> 64]
        # self.re1_conv = nn.Sequential(
        #     *[nn.Conv2d(nb_filter[3], nb_filter[4], 1, 1, 0), nn.LeakyReLU(inplace=True)])  # [64 -> 32]
        # self.re2_conv = nn.Sequential(
        #     *[nn.Conv2d(nb_filter[3], nb_filter[4], 1, 1, 0), nn.LeakyReLU(inplace=True)])  # [64 -> 32]

        # Fusion Block
        self.attn_cot = CoTAttention(dim=nb_filter[4], kernel_size=3)
        ###################### Ablation studies #####################
        # self.attn_cot = CoT(dim=nb_filter[4], kernel_size=3)

        # Decoder Block
        self.decov0 = ConvBnLeakyRelu2d(nb_filter[2], nb_filter[3], kernel_size=3, padding=1)   # [128->64]
        self.decov1 = ConvBnLeakyRelu2d(nb_filter[3], nb_filter[4], kernel_size=3, padding=1)   # [64->32]
        self.decov2 = ConvBnLeakyRelu2d(nb_filter[4], nb_filter[5], kernel_size=3, padding=1)   # [32->16]
        self.tanh = ConvBnTanh2d(nb_filter[5], nb_filter[6])  # 将范围转换为[0,1]     # [16->1]

    def forward(self, img1_Y, img2_Y):
        img1_Y_B, img1_Y_C, img1_Y_H, img1_Y_W = img1_Y.shape
        img2_Y_B, img2_Y_C, img2_Y_H, img2_Y_W = img2_Y.shape

        ###################### Extraction Module #####################
        img1_feature1 = self.inlayer1(img1_Y)  # [B, 16, 358, 358]
        img1_feature2 = self.denselayer1(img1_feature1)  # [B, 64, 358, 358]
        img1_sobel = self.sobelconv1(img1_feature2)  # [B, 64, 358, 358]
        img1_feature3 = self.re1_conv(img1_sobel)  # [B, 32, 358, 358]
        ###################### Ablation studies #####################
        # img1_cnnfeature1 = self.cnnlayer1(img1_feature1)  # [B, 64, 358, 358]

        img2_feature1 = self.inlayer2(img2_Y)  # [B, 16, 358, 358]
        img2_feature2 = self.denselayer2(img2_feature1)  # [B, 64, 358, 358]
        img2_sobel = self.sobelconv2(img2_feature2)  # [B, 64, 358, 358]
        img2_feature3 = self.re2_conv(img2_sobel)  # [B, 32, 358, 358]
        ###################### Ablation studies #####################
        # img1_cnnfeature2 = self.cnnlayer2(img2_feature1)  # [B, 64, 358, 358]

        ####### Adaptive Cross-domain Fusion Module ##############
        feature_layer_con = torch.cat([img1_feature1, img2_feature1], dim=1)  # [B, 32, 358, 358]
        feature_cota = self.attn_cot(feature_layer_con)  # [B, 32, 358, 358]
        ###################### Ablation studies #####################
        # feature_cota = self.attn_cot(feature_layer_con)  # [B, 32, 358, 358]
        feature1_sub = torch.cat([img1_feature3, feature_cota], dim=1)
        feature2_sub = torch.cat([img2_feature3, feature_cota], dim=1)
        feature_cat = torch.cat([feature1_sub, feature2_sub], dim=1)  # [B, 128, 358, 358]


        ###################### Reconstructor Module #####################
        feature_de0 = self.decov0(feature_cat)  # [128->64]
        feature_de1 = self.decov1(feature_de0)  # [64->32]
        feature_de2 = self.decov2(feature_de1)  # [32->16]
        feature_out = self.tanh(feature_de2)

        return feature_out

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)


class reflect_conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, pad=1):
        super(reflect_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.ReflectionPad2d(pad),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                      padding=0)
        )

    def forward(self, x):
        out = self.conv(x)
        return out


class CoTAttention(nn.Module):

    def __init__(self, dim=512, kernel_size=3):
        super().__init__()
        self.dim = dim
        self.kernel_size = kernel_size

        self.key_embed = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=4, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU()
        )
        self.value_embed = nn.Sequential(
            nn.Conv2d(dim, dim, 1, bias=False),
            nn.BatchNorm2d(dim)
        )

        factor = 4
        self.attention_embed = nn.Sequential(
            nn.Conv2d(2 * dim, 2 * dim // factor, 1, bias=False),
            nn.BatchNorm2d(2 * dim // factor),
            nn.ReLU(),
            nn.Conv2d(2 * dim // factor, kernel_size * kernel_size * dim, 1)
        )

    def forward(self, x):
        bs, c, h, w = x.shape
        k1 = self.key_embed(x)  # bs,c,h,w
        v = self.value_embed(x).view(bs, c, -1)  # bs,c,h,w

        y = torch.cat([k1, x], dim=1)  # bs,2c,h,w
        att = self.attention_embed(y)  # bs,c*k*k,h,w
        att = att.reshape(bs, c, self.kernel_size * self.kernel_size, h, w)
        att = att.mean(2, keepdim=False).view(bs, c, h, w)  # bs,c,h,w
        fusion = att + k1
        fusion = fusion.view(bs, c, -1)
        k2 = F.softmax(fusion, dim=-1) * v
        k2 = k2.view(bs, c, h, w)

        return k2

class CoT(nn.Module):

    def __init__(self, dim=512, kernel_size=3):
        super().__init__()
        self.dim = dim
        self.kernel_size = kernel_size

        self.key_embed = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=4, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU()
        )
        self.value_embed = nn.Sequential(
            nn.Conv2d(dim, dim, 1, bias=False),
            nn.BatchNorm2d(dim)
        )

        factor = 4
        self.attention_embed = nn.Sequential(
            nn.Conv2d(2 * dim, 2 * dim // factor, 1, bias=False),
            nn.BatchNorm2d(2 * dim // factor),
            nn.ReLU(),
            nn.Conv2d(2 * dim // factor, kernel_size * kernel_size * dim, 1)
        )

    def forward(self, x):
        bs, c, h, w = x.shape
        k1 = self.key_embed(x)  # bs,c,h,w
        v = self.value_embed(x).view(bs, c, -1)  # bs,c,h,w

        y = torch.cat([k1, x], dim=1)  # bs,2c,h,w
        att = self.attention_embed(y)  # bs,c*k*k,h,w
        att = att.reshape(bs, c, self.kernel_size * self.kernel_size, h, w)
        att = att.mean(2, keepdim=False).view(bs, c, -1)  # bs,c,h*
        k2 = F.softmax(att, dim=-1) * v
        k2 = k2.view(bs, c, h, w)

        return k1 + k2

class Sobelxy(nn.Module):
    def __init__(self, channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(Sobelxy, self).__init__()
        sobel_filter = np.array([[1, 0, -1],
                                 [2, 0, -2],
                                 [1, 0, -1]])
        self.convx = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding, stride=stride,
                               dilation=dilation, groups=channels, bias=False)
        self.convx.weight.data.copy_(torch.from_numpy(sobel_filter))
        self.convy = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding, stride=stride,
                               dilation=dilation, groups=channels, bias=False)
        self.convy.weight.data.copy_(torch.from_numpy(sobel_filter.T))

    def forward(self, x):
        sobelx = self.convx(x)
        sobely = self.convy(x)
        x = torch.abs(sobelx) + torch.abs(sobely)
        return x


class ConvBnTanh2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(ConvBnTanh2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride,
                              dilation=dilation, groups=groups)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return torch.tanh(self.conv(x)) / 2 + 0.5


class ConvBnLeakyRelu2d(nn.Module):
    # convolution
    # batch normalization
    # leaky relu
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(ConvBnLeakyRelu2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride,
                              dilation=dilation, groups=groups)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return F.leaky_relu(self.conv(x), negative_slope=0.2)


# https://github.com/hli1221/densefuse-pytorch/blob/master/net.py
# Dense convolution unit
# 减缓了梯度消失的现象，也使其可以在参数与计算量更少的情况下实现比ResNet更优的性能。
class DenseConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(DenseConv2d, self).__init__()
        self.dense_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.activate = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        out = self.activate(self.dense_conv(x))
        out = torch.cat([x, out], dim=1)
        return out


# Dense Block unit
class DenseBlock(nn.Module):
    def __init__(self, in_channels, kernel_size=3, stride=1, padding=1):
        super(DenseBlock, self).__init__()
        """
                self.DenseBlockLayers = nn.ModuleDict({
                    'DenseConv1': nn.Conv2d(16, 16, 3, 1, 1),
                    'DenseConv2': nn.Conv2d(32, 16, 3, 1, 1),
                    'DenseConv3': nn.Conv2d(48, 16, 3, 1, 1)
                })
          """
        denseblock = []
        denseblock += [DenseConv2d(in_channels, in_channels, kernel_size, stride, padding),
                       DenseConv2d(in_channels + in_channels, in_channels, kernel_size, stride, padding),
                       DenseConv2d(in_channels + in_channels * 2, in_channels, kernel_size, stride, padding)]
        self.denseblock = nn.Sequential(*denseblock)

    def forward(self, x):
        out = self.denseblock(x)
        return out

####################################################################
# class CotLayer(nn.Module):
#     def __init__(self, dim, kernel_size):
#         super(CotLayer, self).__init__()
#
#         self.dim = dim
#         self.kernel_size = kernel_size
#
#         self.key_embed = nn.Sequential(
#             nn.Conv2d(dim, dim, self.kernel_size, stride=1, padding=self.kernel_size // 2, groups=4, bias=False),
#             nn.BatchNorm2d(dim),
#             nn.ReLU(inplace=True)
#         )
#
#         share_planes = 8
#         factor = 4
#         self.embed = nn.Sequential(
#             nn.Conv2d(2 * dim, dim // factor, 1, bias=False),
#             nn.BatchNorm2d(dim // factor),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(dim // factor, pow(kernel_size, 2) * dim // share_planes, kernel_size=1),
#             nn.GroupNorm(num_groups=dim // share_planes, num_channels=pow(kernel_size, 2) * dim // share_planes)
#         )
#
#         self.conv1x1 = nn.Sequential(
#             nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0, dilation=1, bias=False),
#             nn.BatchNorm2d(dim)
#         )
#
#         self.local_conv = LocalConvolution(dim, dim, kernel_size=self.kernel_size, stride=1,
#                                            padding=(self.kernel_size - 1) // 2, dilation=1)
#         self.bn = nn.BatchNorm2d(dim)
#         act = get_act_layer('swish')
#         self.act = act(inplace=True)
#
#         reduction_factor = 4
#         self.radix = 2
#         attn_chs = max(dim * self.radix // reduction_factor, 32)
#         self.se = nn.Sequential(
#             nn.Conv2d(dim, attn_chs, 1),
#             nn.BatchNorm2d(attn_chs),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(attn_chs, self.radix * dim, 1)
#         )
#
#     def forward(self, x):
#         k = self.key_embed(x)
#         qk = torch.cat([x, k], dim=1)
#         b, c, qk_hh, qk_ww = qk.size()
#
#         w = self.embed(qk)
#         w = w.view(b, 1, -1, self.kernel_size * self.kernel_size, qk_hh, qk_ww)
#
#         x = self.conv1x1(x)
#         x = self.local_conv(x, w)
#         x = self.bn(x)
#         x = self.act(x)
#
#         B, C, H, W = x.shape
#         x = x.view(B, C, 1, H, W)
#         k = k.view(B, C, 1, H, W)
#         x = torch.cat([x, k], dim=2)
#
#         x_gap = x.sum(dim=2)
#         x_gap = x_gap.mean((2, 3), keepdim=True)
#         x_attn = self.se(x_gap)
#         x_attn = x_attn.view(B, C, self.radix)
#         x_attn = F.softmax(x_attn, dim=2)
#         out = (x * x_attn.reshape((B, C, self.radix, 1, 1))).sum(dim=2)
#
#         return out

