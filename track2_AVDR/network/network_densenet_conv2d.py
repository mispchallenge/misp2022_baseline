#!/usr/bin/env python
# _*_ coding: UTF-8 _*_
import math
import torch
import torch.nn as nn
from .network_common_module import variable_activate, expend_params


class DenseNet2D(nn.Module):
    def __init__(
            self, in_channels, out_channels, block_num, hidden_channels=12, act_type='relu', expansion=2, stride=1,
            reduction=0.5, drop_rate=0.0, **other_params):
        super(DenseNet2D, self).__init__()
        self.layer_num = 4
        block_num_of_layers = expend_params(value=block_num, length=self.layer_num)
        growth_rates_of_layers = expend_params(value=hidden_channels, length=self.layer_num)
        act_types_of_layers = expend_params(value=act_type, length=self.layer_num)
        expansions_of_layers = expend_params(value=expansion, length=self.layer_num)
        drop_rates_of_layers = expend_params(value=drop_rate, length=self.layer_num)
        strides_of_trans = expend_params(value=stride, length=self.layer_num)
        reductions_of_trans = expend_params(value=reduction, length=self.layer_num)
        in_planes = in_channels
        for layer_idx in range(self.layer_num):
            # print('layer_{}_in_planes: {}'.format(layer_idx, in_planes))
            blocks = []
            for block_idx in range(block_num_of_layers[layer_idx]):
                # print('layer_{}_block_{}_in_planes: {}'.format(layer_idx, block_idx, in_planes))
                blocks.append(
                    DenseBottleneckBlock2D(
                        in_channels=in_planes, out_channels=growth_rates_of_layers[layer_idx],
                        act_type=act_types_of_layers[layer_idx], expansion=expansions_of_layers[layer_idx],
                        drop_rate=drop_rates_of_layers[layer_idx]))
                in_planes = in_planes + growth_rates_of_layers[layer_idx]

            if layer_idx != self.layer_num - 1:
                # print(layer_idx)
                out_planes = int(in_planes*reductions_of_trans[layer_idx])
            else:
                out_planes = out_channels
            blocks.append(
                TransitionBlock2D(in_channels=in_planes, out_channels=out_planes, stride=strides_of_trans[layer_idx],
                                  act_type=act_types_of_layers[layer_idx]))
            in_planes = out_planes
            setattr(self, 'layer_{}'.format(layer_idx), nn.Sequential(*blocks))
        self.pool = nn.AdaptiveAvgPool2d(1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x, length=None):
        for layer_idx in range(self.layer_num):
            x = getattr(self, 'layer_{}'.format(layer_idx))(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return x, length


class DenseBottleneckBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, act_type='relu', expansion=1, drop_rate=0.0):
        super(DenseBottleneckBlock2D, self).__init__()
        hidden_channels = out_channels*expansion
        conv1_sequence = [nn.BatchNorm2d(in_channels), variable_activate(act_type=act_type, in_channels=in_channels),
                          nn.Conv2d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=1, stride=1,
                                    padding=0, bias=False)]
        self.conv1 = nn.Sequential(*conv1_sequence)
        conv2_sequence = [nn.BatchNorm2d(hidden_channels),
                          variable_activate(act_type=act_type, in_channels=hidden_channels),
                          nn.Conv2d(in_channels=hidden_channels, out_channels=out_channels, kernel_size=3, stride=1,
                                    padding=1, bias=False)]
        if drop_rate > 0:
            conv2_sequence.append(nn.Dropout(p=drop_rate, inplace=False))
        else:
            pass
        self.conv2 = nn.Sequential(*conv2_sequence)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        return torch.cat([x, out], dim=1)


class TransitionBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2, act_type='relu'):
        super(TransitionBlock2D, self).__init__()
        process_sequence = [
            nn.BatchNorm2d(in_channels), variable_activate(act_type=act_type, in_channels=in_channels),
            nn.Conv2d(
                in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, bias=False)]
        if stride == 1:
            pass
        else:
            process_sequence.append(
                nn.AvgPool2d(kernel_size=stride, stride=stride, ceil_mode=True, count_include_pad=False))
        self.process = nn.Sequential(*process_sequence)

    def forward(self, x):
        out = self.process(x)
        return out


# class DenseBlock(nn.Module):
#     def __init__(self, nb_layers, in_planes, growth_rate, block, dropRate=0.0):
#         super(DenseBlock, self).__init__()
#         self.layer = self._make_layer(block, in_planes, growth_rate, nb_layers, dropRate)
#     def _make_layer(self, block, in_planes, growth_rate, nb_layers, dropRate):
#         layers = []
#         for i in range(nb_layers):
#             layers.append(block(in_planes+i*growth_rate, growth_rate, dropRate))
#         return nn.Sequential(*layers)
#     def forward(self, x):
#         return self.layer(x)
