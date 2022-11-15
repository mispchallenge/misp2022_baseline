#!/usr/bin/env python
# _*_ coding: UTF-8 _*_
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as nf
from torchvision.transforms import CenterCrop, RandomCrop, RandomHorizontalFlip, Compose
import warnings

from .network_extract_fbank import FilterBankCNN
from .network_specaug import SpecAug

warnings.filterwarnings('ignore')
eps = torch.finfo(torch.float).eps


class FeatureExtractor(nn.Module):
    def __init__(self, extractor_type, extractor_setting, **other_params):
        super(FeatureExtractor, self).__init__()
        self.extractor_type = extractor_type
        if self.extractor_type == 'fbank':
            default_extractor_setting = {
                'n_fft': 512, 'hop_length': 160, 'win_type': 'hamming', 'win_length': None, 'cmvn': None, 'f_min': 0,
                'f_max': 8000, 'n_mels': 40, 'sample_rate': 16000, 'norm': 'slaney', 'preemphasis_coefficient': 0.97,
                'vtln': False, 'vtln_low': 0, 'vtln_high': 8000, 'vtln_warp_factor': 1.}
            extractor_setting = {**default_extractor_setting, **extractor_setting}
            self.output_size = extractor_setting['n_mels']
            self.extractor = FilterBank(**extractor_setting)
        elif self.extractor_type == 'fbank_cnn':
            default_extractor_setting = {
                'n_fft': 512, 'hop_length': 160, 'win_type': 'hamming', 'win_length': None, 'cmvn': None, 'f_min': 0,
                'f_max': 8000, 'n_mels': 40, 'sample_rate': 16000, 'norm': 'slaney', 'preemphasis_coefficient': 0.97,
                'vtln': False, 'vtln_low': 0, 'vtln_high': 8000, 'vtln_warp_factor': 1.}
            extractor_setting = {**default_extractor_setting, **extractor_setting}
            self.output_size = extractor_setting['n_mels']
            self.extractor = FilterBankCNN(**extractor_setting)
        elif self.extractor_type == 'lps':
            default_extractor_setting = {
                'n_fft': 512, 'hop_length': 160, 'win_type': 'hamming', 'win_length': None, 'cmvn': None}
            extractor_setting = {**default_extractor_setting, **extractor_setting}
            self.extractor = LogPowerSpectrum(**extractor_setting)
            self.output_size = extractor_setting['n_fft'] // 2 + 1
        elif self.extractor_type == 'mask':
            default_extractor_setting = {
                'mask_type': 'irm', 'n_fft': 512, 'hop_length': 160, 'win_type': 'hamming', 'win_length': None}
            extractor_setting = {**default_extractor_setting, **extractor_setting}
            self.extractor = IdealMask(**extractor_setting)
            self.output_size = extractor_setting['n_fft'] // 2 + 1
        elif self.extractor_type == 'gray_crop_flip':
            default_extractor_setting = {'channel_input': 'bgr', 'size': None, 'random': False, 'skip_gray': False}
            extractor_setting = {**default_extractor_setting, **extractor_setting}
            self.extractor = GrayCropFlip(**extractor_setting)
            self.output_size = extractor_setting['size']
        else:
            raise NotImplementedError('unknown extractor_type: {}'.format(self.extractor_type))

    def forward(self, x, length=None):
        y, length = self.extractor(x, length)
        return y, length


#  audio feature
class FilterBank(nn.Module):
    def __init__(self, n_fft=512, hop_length=160, win_type='hamming', win_length=None, cmvn=None, f_min=0, f_max=8000,
                 n_mels=40, sample_rate=16000, norm='slaney', preemphasis_coefficient=0.97, vtln=False, vtln_low=0, vtln_high=8000, vtln_warp_factor=1., 
                 specaug=True, apply_time_warp=True, time_warp_window=5, apply_freq_mask=True, freq_mask_width_range=(0, 20), 
                 num_freq_mask=2, apply_time_mask=False, time_mask_width_range=None, time_mask_width_ratio_range=None, num_time_mask=2, **other_params):
        super(FilterBank, self).__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.preemphasis_coefficient = preemphasis_coefficient
        self.stft = ShortTimeFourierTransform(
            n_fft=n_fft, hop_length=hop_length, win_type=win_type, win_length=win_length, is_complex=False)
        self.mel_banks = nn.Parameter(get_mel_banks(
            num_mel_bins=n_mels, num_fft_bins=n_fft, sample_freq=sample_rate, low_freq=f_min, high_freq=f_max,
            norm=norm, vtln=vtln, vtln_low=vtln_low, vtln_high=vtln_high, vtln_warp_factor=vtln_warp_factor),
            requires_grad=False)
        if specaug:
            self.specaug = SpecAug(apply_time_warp=apply_time_warp, time_warp_window=time_warp_window, 
                                   apply_freq_mask=apply_freq_mask, freq_mask_width_range=freq_mask_width_range, num_freq_mask=num_freq_mask,
                                   apply_time_mask=apply_time_mask, time_mask_width_range=time_mask_width_range, 
                                   time_mask_width_ratio_range=time_mask_width_ratio_range, num_time_mask=num_time_mask)
        
        
        self.mean = 0.
        self.std = 1.
        if cmvn is not None:
            cmvn = torch.load(cmvn, map_location=lambda storage, loc: storage)
            self.mean = nn.Parameter(cmvn['mean'].unsqueeze(-1), requires_grad=False)
            self.std = nn.Parameter(cmvn['std'].unsqueeze(-1), requires_grad=False)

    def forward(self, x, length=None):
        if length is not None:
            length = torch.ceil(length / self.hop_length).long()
        if len(x.shape) == 2:
            emphasized_wave = torch.cat([x[:, :1], x[:, 1:]-self.preemphasis_coefficient*x[:, :-1]], dim=-1)
        else:
            emphasized_wave = torch.cat([x[:1], x[1:] - self.preemphasis_coefficient * x[:-1]], dim=-1)
        spectrum = self.stft(emphasized_wave)
        magnitude = spectrum[:, :, :, 0] if len(spectrum.size()) == 4 else spectrum[:, :, 0]
        power = (magnitude ** 2) / self.n_fft
        fbank = torch.matmul(self.mel_banks, power)
        if hasattr(self, 'specaug') and self.training:
            # fbank, length = self.specaug(fbank.transpose(1, 2), length)
            fbank, length = self.specaug(fbank, length)
        fbank = (20 * torch.log10(fbank + eps) - self.mean) / (self.std + eps)
        return fbank, length


def get_mel_banks(num_mel_bins=40, num_fft_bins=400, sample_freq=16000, low_freq=0, high_freq=8000, norm='slaney',
                  vtln=False, vtln_low=0, vtln_high=8000, vtln_warp_factor=1.):
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


class LogPowerSpectrum(nn.Module):
    def __init__(self, n_fft=512, hop_length=160, win_type='hamming', win_length=None, cmvn=None, **other_params):
        super(LogPowerSpectrum, self).__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.stft = ShortTimeFourierTransform(
            n_fft=n_fft, hop_length=hop_length, win_type=win_type, win_length=win_length, is_complex=False)
        self.mean = 0.
        self.std = 1.
        if cmvn is not None:
            cmvn = torch.load(cmvn, map_location=lambda storage, loc: storage)
            self.mean = nn.Parameter(cmvn['mean'].unsqueeze(-1), requires_grad=False)
            self.std = nn.Parameter(cmvn['std'].unsqueeze(-1), requires_grad=False)

    def forward(self, x, length=None):
        if length is not None:
            length = torch.ceil(length / self.hop_length)
        wave = x / (2. ** 15)
        spectrum = self.stft(wave)
        magnitude = spectrum[:, :, :, 0] if len(spectrum.size()) == 4 else spectrum[:, :, 0]
        power = (magnitude ** 2) / self.n_fft
        # power = torch.where(power == 0, power + torch.finfo(torch.float).eps, power)
        log_power = (20 * torch.log10(power + eps) - self.mean) / (self.std + eps)
        return log_power, length


class IdealMask(nn.Module):
    def __init__(self, mask_type='irm', n_fft=512, hop_length=160, win_type='hamming', win_length=None, **other_params):
        super(IdealMask, self).__init__()
        self.mask_type = mask_type
        self.hop_length = hop_length
        self.stft = ShortTimeFourierTransform(
            n_fft=n_fft, hop_length=hop_length, win_type=win_type, win_length=win_length, is_complex=False)

    def forward(self, x, length=None):
        if length is not None:
            length = torch.ceil(length / self.hop_length)
        if self.mask_type in ['irm']:
            powers = []
            for wave in x:
                spectrum = self.stft(wave)
                magnitude = spectrum[:, :, :, 0] if len(spectrum.size()) == 4 else spectrum[:, :, 0]
                powers.append((magnitude ** 2))
            power_sum = torch.stack(powers, dim=0).sum(dim=0)
            # power_sum = torch.where(power_sum == 0, power_sum + torch.finfo(torch.float).eps, power_sum)
            return torch.sqrt(powers[0] / (power_sum + eps) + eps), length
        else:
            raise NotImplementedError('unknown mask_type')


class ShortTimeFourierTransform(nn.Module):
    def __init__(
            self, n_fft=512, hop_length=160, win_type='hamming', win_length=None, is_complex=True):
        super(ShortTimeFourierTransform, self).__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = self.n_fft if win_length is None else win_length
        self.is_complex = is_complex

        if win_type == 'hamming':
            self.window = nn.Parameter(torch.hamming_window(window_length=self.win_length), requires_grad=False)
        else:
            raise NotImplementedError('unknown win_type')

    def forward(self, x, inverse=False, length=None):
        if inverse:
            assert len(x.shape) in [3, 4], 'unknown data shape'
            input_dic = {'window': self.window, 'n_fft': self.n_fft, 'hop_length': self.hop_length,
                         'win_length': self.win_length, 'length': length, 'is_complex': self.is_complex}
            if len(x.shape) == 3:
                return batch_inverse_short_time_fourier_transform(**{**input_dic, **{'batch_x': x.unsqueeze(0)}})[0]
            return batch_inverse_short_time_fourier_transform(**{**input_dic, **{'batch_x': x}})
        else:
            assert len(x.shape) in [1, 2], 'unknown data shape'
            input_dic = {'window': self.window, 'n_fft': self.n_fft, 'hop_length': self.hop_length,
                         'win_length': self.win_length, 'is_complex': self.is_complex}
            if len(x.shape) == 1:
                return batch_short_time_fourier_transform(**{**input_dic, **{'batch_x': x.unsqueeze(0)}})[0]
            return batch_short_time_fourier_transform(**{**input_dic, **{'batch_x': x}})


def batch_short_time_fourier_transform(batch_x, window, n_fft=512, hop_length=160, win_length=None, is_complex=True):
    batch_wave = batch_x.float()
    win_length = n_fft if win_length is None else win_length
    if batch_wave.shape[-1] % hop_length == 0:
        pad_width = [n_fft - hop_length, 0]
    else:
        pad_width = [n_fft - hop_length, hop_length - batch_wave.shape[-1] % hop_length]
    batch_wave = nf.pad(input=batch_wave, pad=pad_width, mode='constant', value=0)
    batch_spectrum = torch.stft(
        input=batch_wave, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window,
        center=False, pad_mode='reflect', normalized=False, onesided=True)
    if not is_complex:
        batch_magnitude = torch.sqrt(batch_spectrum[:, :, :, 0] ** 2 + batch_spectrum[:, :, :, 1] ** 2)
        batch_phase = torch.atan2(batch_spectrum[:, :, :, 1].data, batch_spectrum[:, :, :, 0].data)
        return torch.stack(tensors=[batch_magnitude, batch_phase], dim=-1)
    return batch_spectrum


def batch_inverse_short_time_fourier_transform(
        batch_x, window, n_fft=512, hop_length=160, win_length=None, length=None, is_complex=True):
    batch_spectrum = batch_x
    win_length = n_fft if win_length is None else win_length
    if not is_complex:
        batch_spectrum = torch.stack(tensors=[batch_x[:, :, :, 0] * torch.cos(batch_x[:, :, :, 1]),
                                              batch_x[:, :, :, 0] * torch.sin(batch_x[:, :, :, 1])], dim=-1)
    wave_length = length + win_length - hop_length if length is not None else None
    batch_wave = torch.istft(
        input=batch_spectrum, n_fft=n_fft, hop_length=hop_length, win_length=win_length, center=False,
        window=window, normalized=False, length=wave_length, onesided=True)
    batch_wave = batch_wave[:, win_length - hop_length:]
    return batch_wave


# visual feature
class GrayCropFlip(nn.Module):
    def __init__(self, channel_input='bgr', size=None, random=False, skip_gray=False, **other_params):
        super(GrayCropFlip, self).__init__()
        self.skip_gray = skip_gray
        if not self.skip_gray:
            self.channel2idx = {channel_input[i]: i for i in range(len(channel_input))}
        if size is not None:
            self.random = random
            self.train_transform = Compose([
                RandomCrop(size=size, padding=None, pad_if_needed=False, fill=0, padding_mode='constant'),
                RandomHorizontalFlip(p=0.5)])
            self.eval_transform = Compose([CenterCrop(size=size)])

    def forward(self, x, length=None):
        if not self.skip_gray:
            assert x.shape[-1] == 3, 'shape error: input must have r,g,b 3 channels, but got {}'.format(x.shape)
            x_split = x.split(1, dim=-1)
            gray_frames = 0.114 * x_split[self.channel2idx['b']] + 0.587 * x_split[
                self.channel2idx['g']] + 0.299 * x_split[self.channel2idx['r']]
            x = gray_frames.sum(dim=-1)
        if hasattr(self, 'random'):
            x = self.train_transform(x) if self.training and self.random else self.eval_transform(x)
        return x, length
