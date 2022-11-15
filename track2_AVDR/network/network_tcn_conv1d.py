#!/usr/bin/env python
# _*_ coding: UTF-8 _*_
import torch
import torch.nn as nn
from .network_common_module import Chomp1d, variable_activate, DownSample1d
# from network_common_module import Chomp1d, variable_activate, DownSample1d

class MultiscaleMultibranchTCN(nn.Module):
    def __init__(
            self, in_channels, hidden_channels, num_classes, kernel_size, dropout, act_type, dwpw=False,
            consensus_type='mean', consensus_setting=None, **other_params):
        super(MultiscaleMultibranchTCN, self).__init__()
        self.consensus_type = consensus_type
        self.kernel_sizes = kernel_size
        self.num_kernels = len(self.kernel_sizes)
        self.mb_ms_tcn = MultibranchTemporalConv1DNet(
            in_channels=in_channels, hidden_channels=hidden_channels, kernels_size=kernel_size, dropout=dropout,
            act_type=act_type, dwpw=dwpw, **other_params)
        if self.consensus_type == 'none':
            pass
        # elif self.consensus_type == 'mean':
        #     self.consensus_func = mean_consensus
        # elif self.consensus_type == 'attention':
        #     consensus_setting.update({'d_input': hidden_channels[-1]})
        #     self.consensus_func = MultiHeadAttention(**consensus_setting)
        else:
            raise NotImplementedError('unknown consensus type')
        self.tcn_output = nn.Linear(hidden_channels[-1], num_classes)

    def forward(self, x, length=None):
        # x needs to have dimension (N, C, L) in order to be passed into CNN
        # x_trans = x.transpose(1, 2)
        x_trans = x
        out, length = self.mb_ms_tcn(x_trans, length)
        if self.consensus_type == 'none':
            # out needs to have dimension (N, L, C) in order to be passed into fc
            out = self.tcn_output(out.transpose(1, 2))
            return out, length
        # elif self.consensus_type == 'mean':
        #     out = self.consensus_func(out, length)
        #     out = self.tcn_output(out)
        #     return out
        # elif self.consensus_type == 'attention':
        #     # out needs to have dimension (N, L, C) in order to be passed into attention block
        #     y = out.transpose(1, 2)
        #     out, attention_weight = self.consensus_func(q=y, k=y, v=y, boundary=length)
        #     out = self.tcn_output(out)
        #     return out, attention_weight
        else:
            raise NotImplementedError('unknown consensus type')


class MultibranchTemporalConv1DNet(nn.Module):
    def __init__(
            self, in_channels, hidden_channels, kernels_size, dropout=0.2, act_type='relu', dwpw=False,
            downsample_type='norm', **other_params):
        super(MultibranchTemporalConv1DNet, self).__init__()
        self.kernels_size = kernels_size
        self.blocks_num = len(hidden_channels)
        for block_idx in range(self.blocks_num):
            dilation_size = 2 ** block_idx
            in_planes = in_channels if block_idx == 0 else hidden_channels[block_idx - 1]
            out_planes = hidden_channels[block_idx]
            padding = [(kernel_size - 1) * dilation_size for kernel_size in self.kernels_size]
            setattr(self, 'block_{}'.format(block_idx),
                    MultibranchTemporalConvolution1DBlock(
                        in_channels=in_planes, out_channels=out_planes, kernels_size=self.kernels_size, stride=1,
                        dilation=dilation_size, padding=padding, dropout=dropout, act_type=act_type, dwpw=dwpw,
                        downsample_type=downsample_type))

    def forward(self, x, length=None):
        for block_idx in range(self.blocks_num):
            x = getattr(self, 'block_{}'.format(block_idx))(x)
        return x, length


class MultibranchTemporalConvolution1DBlock(nn.Module):
    def __init__(
            self, in_channels, out_channels, kernels_size, stride, dilation, padding, conv_num=2, dropout=0.2,
            act_type='relu', dwpw=False, downsample_type='norm', **other_params):
        # conv_num = 2
        super(MultibranchTemporalConvolution1DBlock, self).__init__()
        self.kernels_size = kernels_size if isinstance(kernels_size, list) else [kernels_size]
        self.conv_num = conv_num
        self.branches_num = len(kernels_size)
        assert out_channels % self.branches_num == 0, "out_channels needs to be divisible by branches_num"
        self.branch_out_channels = out_channels // self.branches_num

        for conv_idx in range(self.conv_num):
            for kernel_idx, kernel_size in enumerate(self.kernels_size):
                setattr(
                    self, 'conv{}_kernel{}'.format(conv_idx, kernel_size),
                    Conv1dBN1dChomp1dRelu(
                        in_channels=in_channels if conv_idx == 0 else out_channels, act_type=act_type,
                        out_channels=self.branch_out_channels, kernel_size=kernel_size, stride=stride,
                        dilation=dilation, padding=padding[kernel_idx], dwpw=dwpw))
            setattr(self, 'dropout{}'.format(conv_idx), nn.Dropout(dropout))

        if stride != 1 or (in_channels//self.branches_num) != out_channels:
            self.downsample = DownSample1d(
                in_channels=in_channels, out_channels=out_channels, stride=stride, downsample_type=downsample_type)
        else:
            pass
        # final act
        self.final_act = variable_activate(act_type=act_type, in_channels=out_channels)

    def forward(self, x):
        residual = self.downsample(x) if hasattr(self, 'downsample') else x
        y = x
        for conv_idx in range(self.conv_num):
            outputs = [
                getattr(self, 'conv{}_kernel{}'.format(conv_idx, kernel_size))(y) for kernel_size in self.kernels_size]
            y = torch.cat(outputs, dim=1)
            y = getattr(self, 'dropout{}'.format(conv_idx))(y)
        return self.final_act(y + residual)


class Conv1dBN1dChomp1dRelu(nn.Module):
    def __init__(
            self, in_channels, out_channels, kernel_size, stride, dilation, padding, act_type, dwpw=False,
            **other_params):
        super(Conv1dBN1dChomp1dRelu, self).__init__()
        if dwpw:
            self.conv = nn.Sequential(
                nn.Conv1d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, stride=stride,
                          padding=padding, dilation=dilation, groups=in_channels, bias=False),
                nn.BatchNorm1d(in_channels),
                Chomp1d(chomp_size=padding, symmetric_chomp=True),
                variable_activate(act_type=act_type, in_channels=in_channels),
                nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0,
                          bias=False),
                nn.BatchNorm1d(out_channels),
                variable_activate(act_type=act_type, in_channels=out_channels))
        else:
            self.conv = nn.Sequential(
                nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                          padding=padding, dilation=dilation),
                nn.BatchNorm1d(out_channels),
                Chomp1d(padding, True),
                variable_activate(act_type=act_type, in_channels=out_channels))

    def forward(self, x):
        return self.conv(x)


def mean_consensus(x, lengths=None):
    if lengths is None:
        return torch.mean(x, dim=2)
    elif len(lengths.shape) == 1:
        #  begin from 0
        return torch.stack([torch.mean(x[index, :, :length], dim=1) for index, length in enumerate(lengths)], dim=0)
    elif len(lengths.shape) == 2 and lengths.shape[-1] == 2:
        # [begin, end]
        return torch.stack(
            [torch.mean(x[index, :, window[0]:window[1]], dim=1) for index, window in enumerate(lengths)], dim=0)
    elif len(lengths.shape) == 2 and lengths.shape[-1] == x.shape[2]:
        # weight
        return torch.stack(
            [torch.sum(x[index, :, :]*weight, dim=1) for index, weight in enumerate(lengths)], dim=0)
    else:
        raise ValueError('unknown lengths')


if __name__ == '__main__':
    network = MultibranchTemporalConv1DNet(
        in_channels=512, hidden_channels=[256*3, 256*3, 256*3], kernels_size=[3, 5, 7], dropout=0.2, act_type='relu',
        dwpw=False)
    print(network)
    # in_data = [torch.ones(16, 300, 201)]
    output,_ = network(torch.ones(16, 512, 29))
    print(output.size())#[16, 768, 29]
