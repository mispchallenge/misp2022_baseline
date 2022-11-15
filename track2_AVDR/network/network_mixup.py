#!/usr/bin/env python
# _*_ coding: UTF-8 _*_
import random
import torch
import numpy as np
import torch.nn as nn


class MixUp(nn.Module):
    def __init__(self, mix_probability, alpha, mix_types):
        super(MixUp, self).__init__()
        self.mix_probability = mix_probability
        self.alpha = alpha
        self.mix_types = mix_types
    
    def forward(self, *x):
        assert len(x) == len(self.mix_types), '{} but {}'.format(len(x), len(self.mix_types))
        if self.training and random.uniform(0, 1) < self.mix_probability:
            y = []
            lam = np.random.beta(self.alpha, self.alpha)
            for sub_x, mix_type in zip(x, self.mix_types):
                if sub_x is None:
                    y.append(sub_x)
                else:
                    if sub_x.shape[0] % 2 != 0:
                        padded_sub_x = torch.cat([sub_x, sub_x[-1:]], dim=0)
                    else:
                        padded_sub_x = sub_x
                    padded_sub_x_a, padded_sub_x_b = padded_sub_x.chunk(chunks=2, dim=0)
                    if mix_type == 'sum':
                        y.append(lam * padded_sub_x_a + (1. - lam) * padded_sub_x_b)
                    elif mix_type == 'max':
                        y.append(torch.stack([padded_sub_x_a, padded_sub_x_b], dim=0).max(dim=0)[0])
                    else:
                        raise NotImplementedError('Unknown mix_type: {}'.format(mix_type))
            return y
        return [*x]
