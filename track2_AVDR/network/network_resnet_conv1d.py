#!/usr/bin/env python
# _*_ coding: UTF-8 _*_
import math
import torch.nn as nn
from .network_common_module import variable_activate, DownSample1d, expend_params


class ResNet1D(nn.Module):
    def __init__(
            self, block_type='basic1d', block_num=2, in_channels=64, hidden_channels=256, stride=1, act_type='relu',
            expansion=1, downsample_type='norm', **other_params):
        super(ResNet1D, self).__init__()
        self.layer_num = 4
        self.length_retract = 1
        type2block = {'basic1d': BasicBlock1D, 'bottleneck1d': BottleneckBlock1D}
        hidden_channels_of_layers = expend_params(value=hidden_channels, length=self.layer_num)
        stride_of_layers = expend_params(value=stride, length=self.layer_num)
        act_type_of_layers = expend_params(value=act_type, length=self.layer_num)
        expansion_of_layers = expend_params(value=expansion, length=self.layer_num)
        downsample_type_of_layers = expend_params(value=downsample_type, length=self.layer_num)

        in_planes = in_channels
        for layer_idx in range(self.layer_num):
            blocks = []
            self.length_retract = self.length_retract*stride_of_layers[layer_idx]
            for block_idx in range(expend_params(value=block_num, length=self.layer_num)[layer_idx]):
                blocks.append(
                    type2block[block_type](
                        in_channels=in_planes, hidden_channels=hidden_channels_of_layers[layer_idx],
                        stride=stride_of_layers[layer_idx] if block_idx == 0 else 1,
                        act_type=act_type_of_layers[layer_idx], expansion=expansion_of_layers[layer_idx],
                        downsample_type=downsample_type_of_layers[layer_idx]))
                in_planes = int(hidden_channels_of_layers[layer_idx] * expansion_of_layers[layer_idx])
            setattr(self, 'layer{}'.format(layer_idx), nn.Sequential(*blocks))

        # default init
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            if isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        # if self.gamma_zero:
        #     for m in self.modules():
        #         if isinstance(m, BasicBlock):
        #             m.norm2.weight.data.zero_()

    def forward(self, x, length=None):
        if length is not None:
            length = (length / self.length_retract).long()
        for layer_idx in range(self.layer_num):
            x = getattr(self, 'layer{}'.format(layer_idx))(x)
        return x, length


class BasicBlock1D(nn.Module):
    def __init__(
            self, in_channels, hidden_channels, stride=1, act_type='relu', expansion=1, downsample_type='norm',
            **other_params):
        super(BasicBlock1D, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=3, stride=stride, padding=1,
                      bias=False),
            nn.BatchNorm1d(hidden_channels),
            variable_activate(act_type=act_type, in_channels=hidden_channels))

        out_channels = hidden_channels * expansion
        self.conv2 = nn.Sequential(
            nn.Conv1d(
                in_channels=hidden_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(out_channels))
        self.act2 = variable_activate(act_type=act_type, in_channels=out_channels)

        if stride != 1 or in_channels != out_channels:
            self.downsample = DownSample1d(
                in_channels=in_channels, out_channels=out_channels, stride=stride, downsample_type=downsample_type)
        else:
            pass

    def forward(self, x):
        residual = self.downsample(x) if hasattr(self, 'downsample') else x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.act2(out + residual)
        return out


class BottleneckBlock1D(nn.Module):
    def __init__(
            self, in_channels, hidden_channels, stride=1, act_type='relu', expansion=1, downsample_type='norm',
            **other_params):
        super(BottleneckBlock1D, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm1d(hidden_channels),
            variable_activate(act_type=act_type, in_channels=hidden_channels))

        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=3, stride=stride,
                      padding=1, bias=False),
            nn.BatchNorm1d(hidden_channels),
            variable_activate(act_type=act_type, in_channels=hidden_channels))

        out_channels = int(hidden_channels * expansion)
        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels=hidden_channels, out_channels=out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm1d(out_channels))
        self.act3 = variable_activate(act_type=act_type, in_channels=out_channels)

        if stride != 1 or in_channels != out_channels:
            self.downsample = DownSample1d(
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
