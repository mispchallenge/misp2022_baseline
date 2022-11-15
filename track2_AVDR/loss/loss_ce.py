#!/usr/bin/env python
# _*_ coding: UTF-8 _*_
import torch
import torch.nn as nn
import torch.nn.functional as nf

eps = torch.finfo(torch.float).eps


class BatchCalCE(nn.Module):
    def __init__(self, label_type='token', cal_acc=True, ignore_idx=None, reduction='mean',
                 **other_params):
        super(BatchCalCE, self).__init__()
        self.ignore_idx = ignore_idx
        self.label_type = label_type
        self.cal_acc = cal_acc
        self.reduction = reduction

    def forward(self, net_output, label, length=None):
        if self.label_type == 'token':
            packed_label = label.view(-1, ).detach()
            packed_net_output = net_output.view(-1, net_output.shape[-1])
            ce_loss = nf.cross_entropy(
                input=packed_net_output, target=packed_label, weight=None, size_average=None,
                ignore_index=self.ignore_idx, reduce=None, reduction=self.reduction)
            ce_value = ce_loss.numpy() if self.reduction == 'none' else ce_loss.item()
            packed_net_output_token = nf.softmax(packed_net_output, dim=-1).argmax(dim=-1).detach()
            packed_label_token = packed_label.detach()
        elif self.label_type in ['probability', 'representation']:
            label = nf.softmax(label, dim=-1).detach() if self.label_type == 'representation' else label.detach()
            net_output = nf.softmax(net_output, dim=-1)
            padded_ce_sequence = torch.sum(-label * torch.log(net_output + eps), dim=-1)
            padded_label_token = label.argmax(dim=-1).detach()
            padded_net_output_token = net_output.argmax(dim=-1).detach()
            if length is not None:
                mask = padded_ce_sequence.new_ones(padded_ce_sequence.shape)
                for i in range(mask.shape[0]):
                    mask[i, length[i]:] = 0
                packed_ce_sequence = padded_ce_sequence.masked_select(mask.bool())
                packed_net_output_token = padded_net_output_token.masked_select(mask.bool())
                packed_label_token = padded_label_token.masked_select(mask.bool())
            else:
                packed_ce_sequence = padded_ce_sequence.view(-1,)
                packed_net_output_token = padded_net_output_token.view(-1, )
                packed_label_token = padded_label_token.view(-1, )

            if self.reduction == 'none':
                ce_loss = packed_ce_sequence
                ce_value = ce_loss.numpy()
            elif self.reduction == 'mean':
                ce_loss = torch.mean(packed_ce_sequence)
                ce_value = ce_loss.item()
            elif self.reduction == 'sum':
                ce_loss = torch.sum(packed_ce_sequence)
                ce_value = ce_loss.item()
            else:
                raise ValueError('unknown reduction')
        else:
            raise ValueError('unknown label type: {}'.format(self.label_type))

        if self.cal_acc:
            num_correct = (packed_net_output_token == packed_label_token).float().sum().item()
            num_all = (packed_label_token != self.ignore_idx).float().sum().item()
            acc_value = num_correct / num_all
            return ce_loss, torch.tensor([ce_value, acc_value])

        return ce_loss, torch.tensor([ce_value])
