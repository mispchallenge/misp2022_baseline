#!/usr/bin/env python
# _*_ coding: UTF-8 _*_
import os
import math
import torch
import torch.nn as nn
import numpy as np
EPS = 1e-16


def prepare_device(used_gpu, **other_params):
    """
    prepare device
    :param used_gpu: gpu usage
    :return: device, cuda or cpu
    """
    if not used_gpu:
        device = torch.device('cpu')
    elif isinstance(used_gpu, list) or isinstance(used_gpu, int):
        gpu_str = ','.join(map(str, used_gpu)) if isinstance(used_gpu, list) else str(used_gpu)
        # os.environ["CUDA_VISIBLE_DEVICES"] = gpu_str
        device = torch.device('cuda')
    else:
        raise ValueError('unknown use_gpu')
    return device


def expend_params(value, length):
    if isinstance(value, list):
        if len(value) == length:
            return value
        else:
            return [value for _ in range(length)]
    else:
        return [value for _ in range(length)]


def variable_activate(act_type, in_channels=None, **other_params):
    if act_type == 'relu':
        return nn.ReLU(inplace=True)
    elif act_type == 'prelu':
        return nn.PReLU(num_parameters=in_channels)
    else:
        raise NotImplementedError('activate type not implemented')


def unify_time_dimension(*xes):
    lengths = [x.shape[2] for x in xes]
    if len([*set(lengths)]) == 1:
        outs = [*xes]
    else:
        max_length = max(lengths)
        outs = []
        for x in xes:
            if max_length // x.shape[2] != 1:
                if max_length % x.shape[2] == 0:
                    x = torch.stack([x for _ in range(max_length // x.shape[2])], dim=-1).reshape(*x.shape[:-1],
                                                                                                  max_length)
                else:
                    raise ValueError('length error, {}'.format(lengths))
            else:
                pass
            outs.append(x)
    return outs


class DownSample1d(nn.Module):
    def __init__(self, in_channels, out_channels, stride, downsample_type='norm', **others_params):
        super(DownSample1d, self).__init__()
        if downsample_type == 'norm' or stride == 1:
            self.process = nn.Sequential(
                nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels))
        elif downsample_type == 'avgpool':
            self.process = nn.Sequential(
                nn.AvgPool1d(kernel_size=stride, stride=stride, ceil_mode=True, count_include_pad=False),
                nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm1d(out_channels))
        else:
            raise ValueError('unknown downsample type')

    def forward(self, x):
        y = self.process(x)
        return y


class DownSample2d(nn.Module):
    def __init__(self, in_channels, out_channels, stride, downsample_type='norm', **others_params):
        super(DownSample2d, self).__init__()
        if downsample_type == 'norm' or stride == 1:
            self.process = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        elif downsample_type == 'avgpool':
            self.process = nn.Sequential(
                nn.AvgPool2d(kernel_size=stride, stride=stride, ceil_mode=True, count_include_pad=False),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            raise ValueError('unknown downsample type')

    def forward(self, x):
        y = self.process(x)
        return y


class Chomp1d(nn.Module):
    def __init__(self, chomp_size, symmetric_chomp):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size
        self.symmetric_chomp = symmetric_chomp
        if self.symmetric_chomp:
            assert self.chomp_size % 2 == 0, 'If symmetric chomp, chomp size needs to be even'

    def forward(self, x):
        if self.chomp_size == 0:
            return x
        if self.symmetric_chomp:
            return x[:, :, self.chomp_size // 2:-self.chomp_size // 2].contiguous()
        else:
            return x[:, :, :-self.chomp_size].contiguous()


def chose_norm(norm_type, channel_size):
    """
    The input of normalization will be (M, C, K), where M is batch size, C is channel size and K is sequence length.
    """
    if norm_type == "gLN1d":
        return GlobalLayerNorm1d(channel_size)
    elif norm_type == "cLN1d":
        return CumulativeLayerNorm1d(channel_size)
    elif norm_type == 'BN1d':
        return nn.BatchNorm1d(channel_size)
        # norm_type == "BN":
        # Given input (M, C, K), nn.BatchNorm1d(C) will accumulate statics
        # along M and K, so this BN usage is right.
    else:
        raise ValueError('unknown norm_type')


class GlobalLayerNorm1d(nn.Module):
    """Global Layer Normalization (gLN)"""
    def __init__(self, channel_size):
        super(GlobalLayerNorm1d, self).__init__()
        self.gamma = nn.Parameter(torch.ones(1, channel_size, 1), requires_grad=True)  # [1, N, 1]
        self.beta = nn.Parameter(torch.zeros(1, channel_size, 1), requires_grad=True)  # [1, N, 1]
        self.reset_parameters()

    def reset_parameters(self):
        self.gamma.data.fill_(1)
        self.beta.data.zero_()

    def forward(self, y):
        """
        Args:
            y: [M, N, K], M is batch size, N is channel size, K is length
        Returns:
            gLN_y: [M, N, K]
        """
        mean = y.mean(dim=1, keepdim=True).mean(dim=2, keepdim=True)  # [M, 1, 1]
        var = (torch.pow(y-mean, 2)).mean(dim=1, keepdim=True).mean(dim=2, keepdim=True)
        gln_y = self.gamma * (y - mean) / torch.pow(var + EPS, 0.5) + self.beta
        return gln_y


class CumulativeLayerNorm1d(nn.Module):
    def __init__(self, dimension, eps=1e-8, trainable=True):
        super(CumulativeLayerNorm1d, self).__init__()
        self.eps = eps
        if trainable:
            self.gain = nn.Parameter(torch.ones(1, dimension, 1), requires_grad=True)
            self.bias = nn.Parameter(torch.zeros(1, dimension, 1), requires_grad=True)
        else:
            self.gain = torch.ones(1, dimension, 1, requires_grad=False)
            self.bias = torch.zeros(1, dimension, 1, requires_grad=False)

    def forward(self, input0):
        # input size: (Batch, Freq, Time)
        # cumulative mean for each time step
        # batch_size = input0.size(0)
        channel = input0.size(1)
        time_step = input0.size(2)
        step_sum = input0.sum(1)  # B, T
        step_pow_sum = input0.pow(2).sum(1)  # B, T
        cum_sum = torch.cumsum(step_sum, dim=1)  # B, T
        cum_pow_sum = torch.cumsum(step_pow_sum, dim=1)  # B, T
        entry_cnt = np.arange(channel, channel * (time_step + 1), channel)
        entry_cnt = torch.from_numpy(entry_cnt).type(input0.type())
        entry_cnt = entry_cnt.view(1, -1).expand_as(cum_sum)
        cum_mean = cum_sum / entry_cnt  # B, T
        cum_var = (cum_pow_sum - 2 * cum_mean * cum_sum) / entry_cnt + cum_mean.pow(2)  # B, T
        cum_std = (cum_var + self.eps).sqrt()  # B, T
        cum_mean = cum_mean.unsqueeze(1)
        cum_std = cum_std.unsqueeze(1)
        x = (input0 - cum_mean.expand_as(input0)) / cum_std.expand_as(input0)
        return x * self.gain.expand_as(x).type(x.type()) + self.bias.expand_as(x).type(x.type())
