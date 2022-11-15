#!/usr/bin/env python
# _*_ coding: UTF-8 _*_
import torch
import math
import torch.nn as nn
from .network_common_module import variable_activate, DownSample2d, expend_params


class PyResNet2D(nn.Module):
    def __init__(
            self, block_num=2, in_channels=64, hidden_channels=64, pyramid_level=4, kernel_size=3, groups=32, stride=1,
            act_type='relu', expansion=1, downsample_type='norm', **other_params):
        super(PyResNet2D, self).__init__()
        self.layer_num = 4
        hidden_channels_of_layers = expend_params(value=hidden_channels, length=self.layer_num)
        pyramid_levels_of_layers = expend_params(value=pyramid_level, length=self.layer_num)
        kernels_size_of_layers = expend_params(value=kernel_size, length=self.layer_num)
        groups_of_layers = expend_params(value=groups, length=self.layer_num)
        strides_of_layers = expend_params(value=stride, length=self.layer_num)
        act_types_of_layers = expend_params(value=act_type, length=self.layer_num)
        expansions_of_layers = expend_params(value=expansion, length=self.layer_num)
        downsample_types_of_layers = expend_params(value=downsample_type, length=self.layer_num)

        in_planes = in_channels
        for layer_idx in range(self.layer_num):
            blocks = []
            for block_idx in range(expend_params(value=block_num, length=self.layer_num)[layer_idx]):
                blocks.append(PyBottleneckBlock2D(
                    in_channels=in_planes, hidden_channels=hidden_channels_of_layers[layer_idx],
                    pyramid_level=pyramid_levels_of_layers[layer_idx], kernel_size=kernels_size_of_layers[layer_idx],
                    groups=groups_of_layers[layer_idx], stride=strides_of_layers[layer_idx] if block_idx == 0 else 1,
                    act_type=act_types_of_layers[layer_idx], expansion=expansions_of_layers[layer_idx],
                    downsample_type=downsample_types_of_layers[layer_idx]))
                in_planes = int(hidden_channels_of_layers[layer_idx] * expansions_of_layers[layer_idx])
            setattr(self, 'layer{}'.format(layer_idx), nn.Sequential(*blocks))
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # default init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            else:
                pass

    def forward(self, x, length=None):
        for layer_idx in range(self.layer_num):
            x = getattr(self, 'layer_{}'.format(layer_idx))(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        return x, length


class PyBottleneckBlock2D(nn.Module):
    def __init__(
            self, in_channels, hidden_channels, pyramid_level, kernel_size, groups=1, stride=1, act_type='relu',
            expansion=1, downsample_type='norm', **other_params):
        super(PyBottleneckBlock2D, self).__init__()
        if pyramid_level == 3:
            pyramid_out_channels = [hidden_channels//4, hidden_channels//4, hidden_channels//2]
        else:
            pyramid_out_channels = [hidden_channels // pyramid_level for _ in range(pyramid_level)]

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            variable_activate(act_type=act_type, in_channels=hidden_channels))

        self.conv2 = nn.Sequential(
            PyConv2d(
                in_channels=hidden_channels, pyramid_level=pyramid_level, out_channels=pyramid_out_channels,
                kernel_size=kernel_size, groups=groups, stride=stride, dilation=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            variable_activate(act_type=act_type, in_channels=hidden_channels))

        out_channels = int(hidden_channels * expansion)
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=hidden_channels, out_channels=out_channels, kernel_size=1,
                stride=1, bias=False),
            nn.BatchNorm2d(out_channels))
        self.act3 = variable_activate(act_type=act_type, in_channels=out_channels)

        if stride != 1 or in_channels != out_channels:
            self.downsample = DownSample2d(
                in_channels=in_channels, out_channels=out_channels, stride=stride, downsample_type=downsample_type)
        else:
            pass

    def forward(self, x):
        residual = self.downsample(x) if hasattr(self, 'downsample') else x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.act3(out + residual)
        return out


class PyConv2d(nn.Module):
    def __init__(
            self, in_channels, pyramid_level, out_channels, kernel_size, groups, stride=1, dilation=1, bias=False):
        super(PyConv2d, self).__init__()
        self.pyramid_level = pyramid_level
        out_channels_of_pyramid = expend_params(value=out_channels, length=pyramid_level)
        kernels_size_of_pyramid = expend_params(value=kernel_size, length=pyramid_level)
        groups_of_pyramid = expend_params(value=groups, length=pyramid_level)
        strides_of_pyramid = expend_params(value=stride, length=pyramid_level)
        dilation_of_pyramid = expend_params(value=dilation, length=pyramid_level)
        biases_of_pyramid = expend_params(value=bias, length=pyramid_level)
        for level_idx in range(pyramid_level):
            setattr(
                self, 'pyconv_{}'.format(level_idx),
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels_of_pyramid[level_idx],
                          kernel_size=kernels_size_of_pyramid[level_idx], stride=strides_of_pyramid[level_idx],
                          padding=(kernels_size_of_pyramid[level_idx]-1)//2, groups=groups_of_pyramid[level_idx],
                          dilation=dilation_of_pyramid[level_idx], bias=biases_of_pyramid[level_idx]))

    def forward(self, x):
        out = []
        for level_idx in range(self.pyramid_level):
            out.append(getattr(self, 'pyconv_{}'.format(level_idx))(x))
        return torch.cat(out, dim=1)
