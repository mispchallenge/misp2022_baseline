#!/usr/bin/env python
# _*_ coding: UTF-8 _*_
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as nf


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention module
    """
    def __init__(self, n_head, d_input, d_k, d_v, window=None, dropout=0.1, attn_dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.window = window
        self.w_qs = nn.Linear(d_input, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_input, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_input, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_input, bias=False)
        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5, attn_dropout=attn_dropout)
        self.dropout = nn.Dropout(dropout)
        # self.layer_norm = nn.LayerNorm(d_input, eps=1e-6)

    def forward(self, q, k, v, boundary=None):
        d_k, d_v, n_head, window = self.d_k, self.d_v, self.n_head, self.window
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)
        # Pass through the pre-attention projection: b x l x (n*d)
        # Separate different heads: b x l x n x d and transpose for attention dot product: b x n x l x d
        multi_head_q = self.w_qs(q).view(sz_b, len_q, n_head, d_k).transpose(1, 2)
        multi_head_k = self.w_ks(k).view(sz_b, len_k, n_head, d_k).transpose(1, 2)
        multi_head_v = self.w_vs(v).view(sz_b, len_v, n_head, d_v).transpose(1, 2)
        if (isinstance(window, bool) or window is None) and boundary is None:
            assert not window, 'bool only used to express no window'
            residual_v = torch.mean(v, dim=1, keepdim=True)
            attn_mask = None
            multi_head_q_mean = torch.mean(multi_head_q, dim=2, keepdim=True)
        elif isinstance(window, (int, float, list, tuple)) and not isinstance(window, bool):
            assert boundary is None, 'window is at odds with boundary'
            residual_v = torch.mean(v, dim=1, keepdim=True)
            attn_mask = None
            if isinstance(window, (int, float)):
                start_point = int((len_q-window)//2)
                end_point = int((len_q-window)//2+window)
            else:
                start_point, end_point = window
            start_point = max(start_point, 0)
            start_point = 0 if start_point > len_q - 1 else start_point
            end_point = min(end_point, len_q)
            end_point = len_q if end_point < 0 else end_point
            multi_head_q_mean = torch.mean(multi_head_q[:, :, start_point: end_point, :], dim=2, keepdim=True)
        elif boundary is not None:
            assert not window, 'boundary is at odds with window'
            residual_v = torch.stack(
                [torch.sum(v[index, :, :]*weight.unsqueeze(-1), dim=0, keepdim=True)
                 for index, weight in enumerate(boundary)], dim=0)
            attn_mask = torch.stack([boundary.unsqueeze(dim=1) for _ in range(n_head)], dim=1)
            multi_head_q_mean = torch.stack(
                [torch.sum(multi_head_q[index, :, :, :]*weight.unsqueeze(0).unsqueeze(-1), dim=1, keepdim=True)
                 for index, weight in enumerate(boundary)], dim=0)
        else:
            raise ValueError('unknown window or boundary')
        final_v, attn = self.attention(q=multi_head_q_mean, k=multi_head_k, v=multi_head_v, mask=attn_mask)
        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        final_v = final_v.transpose(1, 2).contiguous().view(sz_b, 1, -1)
        final_v = self.dropout(self.fc(final_v))
        final_v += residual_v
        # q = self.layer_norm(q)
        return final_v[:, 0, :], attn


class ScaledDotProductAttention(nn.Module):
    def __init__(self, temperature, attn_dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        attn = self.dropout(nf.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)
        return output, attn


if __name__ == '__main__':
    test_input = torch.ones(2, 5, 1)
    test_boundary = torch.from_numpy(np.array([[0., 1./3., 1./3., 1./3., 0.], [0., 1./2., 0., 1./2., 0.]], dtype='float32'))
    test_window_1 = 13
    test_window_2 = [8, 21]
    test_model = MultiHeadAttention(n_head=8, d_input=1, d_k=64, d_v=64, dropout=0.1, window=None, attn_dropout=0.0)
    test_output = test_model(q=test_input, k=test_input, v=test_input, boundary=test_boundary)
    print(test_output[0])
    print(test_output[1])

