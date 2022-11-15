#!/usr/bin/env python
# _*_ coding: UTF-8 _*_
import torch
import torch.nn as nn

EPS = torch.finfo(torch.float).eps


class BatchCalSNR(nn.Module):
    def __init__(self, scale_invariability=True):
        super(BatchCalSNR, self).__init__()
        self.si = scale_invariability

    def forward(self, net_output, label, length=None):
        """
        net_output: [B, T], B is batch size
        label: [B, T]
        length: [B], each item is between [0, T]
        """
        assert label.size() == net_output.size()
        # mask
        mask = label.new_ones(label.shape)
        if length is not None:
            for i in range(mask.shape[0]):
                mask[i, length[i]:] = 0.

        net_output *= mask

        # Step 1. Zero-mean norm
        num_samples = length.to(label.device).view(-1, 1).float()  # [B, 1]
        mean_label = torch.sum(label, dim=-1, keepdim=True) / num_samples
        mean_net_output = torch.sum(net_output, dim=-1, keepdim=True) / num_samples
        zero_mean_label = label - mean_label
        zero_mean_net_output = net_output - mean_net_output
        # mask padding position along T
        zero_mean_label *= mask
        zero_mean_net_output *= mask

        # Step 2. cal SNR
        # scale invariability transform
        if self.si:
            # s_target = <s', s>s / ||s||^2
            net_output_dot_label = torch.sum(zero_mean_label*zero_mean_net_output, dim=-1, keepdim=True)  # [B, 1]
            label_energy = torch.sum(zero_mean_label ** 2, dim=-1, keepdim=True) + EPS # [B, 1]
            zero_mean_label = net_output_dot_label * zero_mean_label / label_energy

        noise = zero_mean_net_output - zero_mean_label
        # SNR = 10 * log_10(||s_target||^2 / ||e_noise||^2)
        snr_per_sample = 10 * torch.log10(
            torch.sum(zero_mean_label ** 2, dim=-1) / (torch.sum(noise ** 2, dim=-1) + EPS) + EPS)
        snr = torch.mean(snr_per_sample)
        return -snr, torch.tensor([snr.item()])
