#!/usr/bin/env python
# _*_ coding: UTF-8 _*_
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as nf

eps = torch.finfo(torch.float).eps


def generate_stft_forward_weight(win_len, n_fft, win_type='hamming'):
    win_len = n_fft if win_len is None else win_len
    fourier_basis = torch.fft.fft(torch.eye(n_fft))
    cutoff = int(n_fft / 2 + 1)
    fourier_basis = torch.vstack([torch.real(fourier_basis[:cutoff, :]),
                                  torch.imag(fourier_basis[:cutoff, :])])
    forward_weight = torch.FloatTensor(fourier_basis[:, None, :])

    if win_type == 'hamming':
        fft_window = torch.hamming_window(window_length=win_len)
    else:
        raise NotImplementedError('unknown win_type')
    fft_window = torch.cat([torch.zeros(n_fft - win_len), fft_window], dim=0)

    forward_weight = forward_weight * fft_window
    return forward_weight


def generate_mel_filter_banks(num_mel_bins=40, num_fft_bins=400, sample_freq=16000, low_freq=0, high_freq=8000,
                              norm='slaney', vtln=False, vtln_low=0, vtln_high=8000, vtln_warp_factor=1.):
    nyquist = 0.5 * sample_freq
    high_freq = high_freq + nyquist if high_freq <= 0.0 else high_freq
    if not (0.0 <= low_freq < nyquist and (0.0 < high_freq <= nyquist) and (low_freq < high_freq)):
        raise ValueError('Bad values in options: low-freq {} and high-freq {} vs. nyquist {}'.format(
            low_freq, high_freq, nyquist))
    if vtln:
        # fft-bin width [think of it as Nyquist-freq / half-window-length]
        fft_bin_width = sample_freq / num_fft_bins
        mel_low_freq = 1127.0 * np.log(1.0 + low_freq / 700.0)
        mel_high_freq = 1127.0 * np.log(1.0 + high_freq / 700.0)
        # divide by num_bins+1 in next line because of end-effects where the bins spread out to the sides.
        mel_freq_delta = (mel_high_freq - mel_low_freq) / (num_mel_bins + 1)
        vtln_high = vtln_high + nyquist if vtln_high <= 0.0 else vtln_high
        if not (vtln_warp_factor == 1.0
                or ((low_freq < vtln_low < high_freq) and (0.0 < vtln_high < high_freq) and (vtln_low < vtln_high))):
            raise ValueError(
                'Bad values in options: vtln-low {} and vtln-high {}, versus low-freq {} and high-freq {}'.format(
                    vtln_low, vtln_high, low_freq, high_freq))
        fbank_bin = torch.arange(num_mel_bins).unsqueeze(1)
        left_mel = mel_low_freq + fbank_bin * mel_freq_delta  # size(num_bins, 1)
        center_mel = mel_low_freq + (fbank_bin + 1.0) * mel_freq_delta  # size(num_bins, 1)
        right_mel = mel_low_freq + (fbank_bin + 2.0) * mel_freq_delta  # size(num_bins, 1)

        if vtln_warp_factor != 1.0:
            left_mel = 1127.0 * (1.0 + vtln_warp_freq(
                vtln_low_cutoff=vtln_low, vtln_high_cutoff=vtln_high, low_freq=low_freq, high_freq=high_freq,
                vtln_warp_factor=vtln_warp_factor, freq=left_mel) / 700.0).log()
            center_mel = 1127.0 * (1.0 + vtln_warp_freq(
                vtln_low_cutoff=vtln_low, vtln_high_cutoff=vtln_high, low_freq=low_freq, high_freq=high_freq,
                vtln_warp_factor=vtln_warp_factor, freq=center_mel) / 700.0).log()
            right_mel = 1127.0 * (1.0 + vtln_warp_freq(
                vtln_low_cutoff=vtln_low, vtln_high_cutoff=vtln_high, low_freq=low_freq, high_freq=high_freq,
                vtln_warp_factor=vtln_warp_factor, freq=right_mel) / 700.0).log()

        center_freq = 700.0 * ((center_mel / 1127.0).exp() - 1.0)  # size (num_bins)
        mel = 1127.0 * (1.0 + fft_bin_width * torch.arange((num_fft_bins // 2 + 1)) / 700.0).log().unsqueeze(0)
        # size (num_bins, num_fft_bins)
        up_slope = (mel - left_mel) / (center_mel - left_mel)
        down_slope = (right_mel - mel) / (right_mel - center_mel)
        if vtln_warp_factor == 1.0:
            # left_mel < center_mel < right_mel so we can min the two slopes and clamp negative values
            bins = torch.max(torch.zeros(1), torch.min(up_slope, down_slope))
        else:
            # warping can move the order of left_mel, center_mel, right_mel anywhere
            bins = torch.zeros_like(up_slope)
            up_idx = torch.gt(mel, left_mel) & torch.le(mel, center_mel)  # left_mel < mel <= center_mel
            down_idx = torch.gt(mel, center_mel) & torch.lt(mel, right_mel)  # center_mel < mel < right_mel
            bins[up_idx] = up_slope[up_idx]
            bins[down_idx] = down_slope[down_idx]
        return bins
    else:
        all_freqs = torch.linspace(0, sample_freq // 2, num_fft_bins // 2 + 1)
        # calculate mel freq bins
        # hertz to mel(f) is 2595. * math.log10(1. + (f / 700.))
        m_min = 2595.0 * np.log10(1.0 + (low_freq / 700.0))
        m_max = 2595.0 * np.log10(1.0 + (high_freq / 700.0))
        m_pts = torch.linspace(m_min, m_max, num_mel_bins + 2)
        # mel to hertz(mel) is 700. * (10**(mel / 2595.) - 1.)
        f_pts = 700.0 * (10 ** (m_pts / 2595.0) - 1.0)
        # calculate the difference between each mel point and each stft freq point in hertz
        f_diff = f_pts[1:] - f_pts[:-1]  # (n_mels + 1)
        slopes = f_pts.unsqueeze(0) - all_freqs.unsqueeze(1)  # (n_freqs, n_mels + 2)
        # create overlapping triangles
        zero = torch.zeros(1)
        down_slopes = (-1.0 * slopes[:, :-2]) / f_diff[:-1]  # (n_freqs, n_mels)
        up_slopes = slopes[:, 2:] / f_diff[1:]  # (n_freqs, n_mels)
        fb = torch.max(zero, torch.min(down_slopes, up_slopes))
        if norm is not None and norm == 'slaney':
            # Slaney-style mel is scaled to be approx constant energy per channel
            enorm = 2.0 / (f_pts[2:num_mel_bins + 2] - f_pts[:num_mel_bins])
            fb *= enorm.unsqueeze(0)
        return fb.T


def vtln_warp_freq(vtln_low_cutoff, vtln_high_cutoff, low_freq, high_freq, vtln_warp_factor, freq):
    """
    This computes a VTLN warping function that is not the same as HTK's one, but has similar inputs (this function has
    the advantage of never producing empty bins).

    This function computes a warp function F(freq), defined between low_freq and high_freq inclusive, with the following
    properties:
        F(low_freq) == low_freq
        F(high_freq) == high_freq
    The function is continuous and piecewise linear with two inflection points.
    The lower inflection point (measured in terms of the unwarped frequency) is at frequency l, determined as described
    below. The higher inflection point is at a frequency h, determined as described below.
    If l <= f <= h, then F(f) = f/vtln_warp_factor.
    If the higher inflection point (measured in terms of the unwarped frequency) is at h, then max(h, F(h)) ==
    vtln_high_cutoff. Since (by the last point) F(h) == h/vtln_warp_factor, then max(h, h/vtln_warp_factor) ==
    vtln_high_cutoff, so
        h = vtln_high_cutoff / max(1, 1/vtln_warp_factor) = vtln_high_cutoff * min(1, vtln_warp_factor).
    If the lower inflection point (measured in terms of the unwarped frequency) is at l, then min(l, F(l)) ==
    vtln_low_cutoff. This implies that
        l = vtln_low_cutoff / min(1, 1/vtln_warp_factor) = vtln_low_cutoff * max(1, vtln_warp_factor)

    Args:
        vtln_low_cutoff (float): Lower frequency cutoffs for VTLN
        vtln_high_cutoff (float): Upper frequency cutoffs for VTLN
        low_freq (float): Lower frequency cutoffs in mel computation
        high_freq (float): Upper frequency cutoffs in mel computation
        vtln_warp_factor (float): Vtln warp factor
        freq (Tensor): given frequency in Hz
    Returns:
        Tensor: Freq after vtln warp
    """
    assert vtln_low_cutoff > low_freq, 'be sure to set the vtln_low option higher than low_freq'
    assert vtln_high_cutoff < high_freq, 'be sure to set the vtln_high option lower than high_freq [or negative]'
    low = vtln_low_cutoff * max(1.0, vtln_warp_factor)
    high = vtln_high_cutoff * min(1.0, vtln_warp_factor)
    scale = 1.0 / vtln_warp_factor
    f_low = scale * low  # F(l)
    f_high = scale * high  # F(h)
    assert low > low_freq and high < high_freq
    # slope of left part of the 3-piece linear function
    scale_left = (f_low - low_freq) / (low - low_freq)
    # [slope of center part is just "scale"]
    # slope of right part of the 3-piece linear function
    scale_right = (high_freq - f_high) / (high_freq - high)
    res = torch.empty_like(freq)
    outside_low_high_freq = torch.lt(freq, low_freq) | torch.gt(freq, high_freq)  # freq < low_freq || freq > high_freq
    before_l = torch.lt(freq, low)  # freq < l
    before_h = torch.lt(freq, high)  # freq < h
    after_h = torch.ge(freq, high)  # freq >= h
    # order of operations matter here (since there is overlapping frequency regions)
    res[after_h] = high_freq + scale_right * (freq[after_h] - high_freq)
    res[before_h] = scale * freq[before_h]
    res[before_l] = low_freq + scale_left * (freq[before_l] - low_freq)
    res[outside_low_high_freq] = freq[outside_low_high_freq]
    return res


class FilterBankCNN(nn.Module):
    def __init__(self, n_fft=512, hop_length=160, win_type='hamming', win_length=None, cmvn=None, f_min=0, f_max=8000,
                 n_mels=40, sample_rate=16000, norm='slaney', preemphasis_coefficient=0.97, vtln=False, vtln_low=0,
                 vtln_high=8000, vtln_warp_factor=1.):
        super(FilterBankCNN, self).__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.preemphasis_coefficient = preemphasis_coefficient
        self.cutoff = int((self.n_fft / 2) + 1)
        self.stft_forward_weight = generate_stft_forward_weight(win_len=win_length, n_fft=n_fft, win_type=win_type)
        self.mel_filter_banks = generate_mel_filter_banks(
            num_mel_bins=n_mels, num_fft_bins=n_fft, sample_freq=sample_rate, low_freq=f_min, high_freq=f_max,
            norm=norm, vtln=vtln, vtln_low=vtln_low, vtln_high=vtln_high, vtln_warp_factor=vtln_warp_factor)
        if cmvn is not None:
            cmvn = torch.load(cmvn, map_location=lambda storage, loc: storage)
            self.mean = cmvn['mean'].unsqueeze(-1)
            self.std = cmvn['std'].unsqueeze(-1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, length=None):
        if length is not None:
            length = torch.ceil(length / self.hop_length).long()

        emphasized_wave = torch.cat([x[:, :1], x[:, 1:] - self.preemphasis_coefficient * x[:, :-1]], dim=-1)
        pad_width = [self.n_fft - self.hop_length]
        if emphasized_wave.shape[-1] % self.hop_length == 0:
            pad_width.append(0)
        else:
            pad_width.append(self.hop_length - emphasized_wave.shape[-1] % self.hop_length)
        emphasized_wave = nf.pad(input=emphasized_wave, pad=pad_width, mode='constant', value=0)
        emphasized_wave = emphasized_wave.unsqueeze(1)
        spectrum = nf.conv1d(emphasized_wave, self.stft_forward_weight.to(emphasized_wave.device), stride=self.hop_length,
                             padding=0)
        real_spectrum = spectrum[:, :self.cutoff, :]
        imag_spectrum = spectrum[:, self.cutoff:, :]
        power_spectrum = (real_spectrum**2 + imag_spectrum**2) / self.n_fft  # .permute(0, 2, 1)
        fbank = torch.matmul(self.mel_filter_banks.to(power_spectrum.device), power_spectrum)
        fbank = 20 * torch.log10(fbank + eps)
        if hasattr(self, 'mean') and hasattr(self, 'std'):
             fbank = (fbank - self.mean.to(fbank.device)) / (self.std + eps).to(fbank.device)
        return fbank, length
