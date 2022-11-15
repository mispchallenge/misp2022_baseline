#!/usr/bin/env python
# _*_ coding: UTF-8 _*_
import torch
import torch.nn as nn

from .loss_mse import BatchCalMSE
from .loss_snr import BatchCalSNR
from .loss_ce import BatchCalCE


class BatchCalCEplusSNRplusMSE(nn.Module):
    def __init__(self, weights=[1, 1, 1], mode='correct', scale_invariability=True, label_type='token', cal_acc=True, ignore_idx=None, reduction='mean'):
        super(BatchCalCEplusSNRplusMSE, self).__init__()
        self.weights = weights
        assert len(self.weights) == 3
        assert self.weights[0] != 0 or self.weights[1] != 0 or self.weights[2] != 0
        
        if self.weights[0] != 0:
            self.ce_computer = BatchCalCE(label_type=label_type, cal_acc=cal_acc, ignore_idx=ignore_idx, reduction=reduction)
        
        if self.weights[1] != 0:
            self.snr_computer = BatchCalSNR(scale_invariability=scale_invariability)
            
        if self.weights[2] != 0:
            self.mse_computer = BatchCalMSE(mode=mode)
        
        
        
    def forward(self, net_output_ce=None, net_output_snr=None, net_output_mse=None, label_ce=None, label_snr=None, label_mse=None, length_ce=None, length_snr=None, length_mse=None):
        loss_sum = 0
        all_out = []
        
        if self.weights[0] != 0:
            ce_loss, ce_out = self.ce_computer(net_output_ce, label_ce, length_ce)
            loss_sum += self.weights[0]*ce_loss
            all_out.append(ce_out)
        
        if self.weights[1] != 0:
            snr_loss, snr_out = self.snr_computer(net_output_snr, label_snr, length_snr)
            loss_sum += self.weights[1]*snr_loss
            all_out.append(snr_out)
        
        if self.weights[2] != 0:
            mse_loss, mse_out = self.mse_computer(net_output_mse, label_mse, length_mse)
            loss_sum += self.weights[2]*mse_loss
            all_out.append(mse_out)
        
 
        all_out = torch.cat([torch.tensor([loss_sum.item()]), *all_out], dim=0)
        return loss_sum, all_out
        