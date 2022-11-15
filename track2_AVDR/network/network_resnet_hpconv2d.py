#!/usr/bin/env python
# _*_ coding: UTF-8 _*_
import math
import torch
import torch.nn as nn
from .network_common_module import variable_activate, DownSample2d, expend_params


class HPResNet2D(nn.Module):
    def __init__(
            self, block_num=2, in_channels=64, hidden_channels=64, split_num=8, kernel_size=3, stride=1,
            act_type='relu', expansion=1, stride_type='1x1', downsample_type='norm', **other_params):
        super(HPResNet2D, self).__init__()
        self.layer_num = 4
        hidden_channels_of_layers = expend_params(value=hidden_channels, length=self.layer_num)
        split_num_of_layers = expend_params(value=split_num, length=self.layer_num)
        kernel_size_of_layers = expend_params(value=kernel_size, length=self.layer_num)
        stride_of_layers = expend_params(value=stride, length=self.layer_num)
        act_type_of_layers = expend_params(value=act_type, length=self.layer_num)
        expansion_of_layers = expend_params(value=expansion, length=self.layer_num)
        stride_type_of_layers = expend_params(value=stride_type, length=self.layer_num)
        downsample_type_of_layers = expend_params(value=downsample_type, length=self.layer_num)

        in_planes = in_channels
        for layer_idx in range(self.layer_num):
            blocks = []
            for block_idx in range(expend_params(value=block_num, length=self.layer_num)[layer_idx]):
                blocks.append(
                    HPBottleneckBlock2D(
                        in_channels=in_planes, hidden_channels=hidden_channels_of_layers[layer_idx],
                        stride=stride_of_layers[layer_idx] if block_idx == 0 else 1,
                        kernels_size=kernel_size_of_layers[layer_idx], act_type=act_type_of_layers[layer_idx],
                        expansion=expansion_of_layers[layer_idx], stride_type=stride_type_of_layers[layer_idx],
                        split_num=split_num_of_layers[layer_idx], downsample_type=downsample_type_of_layers[layer_idx]))
                in_planes = int(hidden_channels_of_layers[layer_idx] * expansion_of_layers[layer_idx])
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
        # if self.gamma_zero:
        #     for m in self.modules():
        #         if isinstance(m, BasicBlock):
        #             m.norm2.weight.data.zero_()

    def forward(self, x, length=None):
        for layer_idx in range(self.layer_num):
            x = getattr(self, 'layer{}'.format(layer_idx))(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        return x, length


class HPBottleneckBlock2D(nn.Module):
    def __init__(
            self, in_channels, hidden_channels, stride=1, kernels_size=3, act_type='relu', expansion=1,
            stride_type='1x1', split_num=8, downsample_type='norm', **other_params):
        super(HPBottleneckBlock2D, self).__init__()
        out_channels = int(hidden_channels * expansion)
        if stride == 1 or stride_type == '1x1':
            self.conv1 = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels, out_channels=hidden_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(hidden_channels),
                variable_activate(act_type=act_type, in_channels=hidden_channels))
        elif stride_type == '3x3':
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=3, stride=stride,
                          padding=1, bias=False),
                nn.BatchNorm2d(hidden_channels),
                variable_activate(act_type=act_type, in_channels=hidden_channels))
        elif stride_type == 'dw-3x3':
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(hidden_channels),
                variable_activate(act_type=act_type, in_channels=hidden_channels),
                nn.Conv2d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=3, stride=stride,
                          padding=1, groups=hidden_channels, bias=False),
                nn.BatchNorm2d(hidden_channels),
                variable_activate(act_type=act_type, in_channels=hidden_channels))
        else:
            raise NotImplementedError('unknown stride_type')

        assert hidden_channels % split_num == 0, 'hidden_channels needs to be divisible by split_num'
        split_in_channels = hidden_channels // split_num
        self.split_out_channels = hidden_channels // int(2 ** (split_num - 1))
        self.split_num = split_num
        for split_idx in range(1, split_num):
            in_planes = split_in_channels if split_idx == 1 else int(2*split_in_channels-self.split_out_channels)
            kernel_size = expend_params(value=kernels_size, length=split_num)[split_idx]
            setattr(self, 'conv2_{}'.format(split_idx),
                    nn.Sequential(
                        nn.Conv2d(
                            in_channels=in_planes, out_channels=split_in_channels, stride=1, padding=(kernel_size-1)//2,
                            bias=False, kernel_size=kernel_size),
                        nn.BatchNorm2d(split_in_channels),
                        variable_activate(act_type=act_type, in_channels=split_in_channels)))

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=int(2 * split_in_channels + (split_num - 2) * self.split_out_channels),
                      out_channels=out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels))
        self.act3 = variable_activate(act_type=act_type, in_channels=out_channels)

        if stride != 1 or in_channels != out_channels:
            self.downsample = DownSample2d(in_channels=in_channels, out_channels=out_channels, stride=stride,
                                           downsample_type=downsample_type)
        else:
            pass

    def forward(self, x):
        residual = self.downsample(x) if hasattr(self, 'downsample') else x
        out = self.conv1(x)

        split_inputs = out.chunk(self.split_num, dim=1)
        current_hs_conv_output = split_inputs[0]
        hs_conv_outputs = [current_hs_conv_output]
        for split_idx in range(1, self.split_num):
            if split_idx == 1:
                hs_conv_input = split_inputs[split_idx]
            else:
                hs_conv_input = torch.cat([split_inputs[split_idx], current_hs_conv_output[:, self.split_out_channels:, :, :]],
                                          dim=1)
            current_hs_conv_output = getattr(self, 'conv2_{}'.format(split_idx))(hs_conv_input)
            if split_idx == self.split_num - 1:
                hs_conv_outputs.append(current_hs_conv_output)
            else:
                hs_conv_outputs.append(current_hs_conv_output[:, :self.split_out_channels, :, :])
        hs_conv_output = torch.cat(hs_conv_outputs, dim=1)
        out = self.conv3(hs_conv_output)
        out = self.act3(out + residual)
        return out
