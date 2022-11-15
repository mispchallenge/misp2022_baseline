#!/usr/bin/env python
# _*_ coding: UTF-8 _*_
import torch
import torch.nn as nn
from .network_feature_extract import ShortTimeFourierTransform


class ReconstructMask2Wave(nn.Module):
    def __init__(self, n_fft, hop_length, win_type='hamming', win_length=None, rescale=False, fit_wav=True):
        super(ReconstructMask2Wave, self).__init__()
        self.stft = ShortTimeFourierTransform(n_fft=n_fft, hop_length=hop_length, win_type=win_type,
                                              win_length=win_length, is_complex=False)
        self.fit_wav = fit_wav
        self.rescale = rescale

    def forward(self, predicted_mask, mixture_wave, *clean_wave_and_length):
        if self.rescale and len(clean_wave_and_length) == 2:
            length = clean_wave_and_length[1]
        elif not self.rescale and len(clean_wave_and_length) == 1:
            length = clean_wave_and_length[0]
        else:
            length = None
        mixture_wave = mixture_wave / (2. ** 15)
        mixture_spectrum = self.stft(x=mixture_wave, inverse=False)
        if mixture_spectrum.ndim == 4:
            mixture_magnitude = mixture_spectrum[:, :, :, 0]
            mixture_angle = mixture_spectrum[:, :, :, 1]
            predicted_mask = predicted_mask.transpose(1, 2)
        else:
            mixture_magnitude = mixture_spectrum[:, :, 0]
            mixture_angle = mixture_spectrum[:, :, 1]
            predicted_mask = predicted_mask.transpose(0, 1)
        reconstruct_magnitude = mixture_magnitude * predicted_mask
        reconstructed_wave = self.stft(x=torch.stack([reconstruct_magnitude, mixture_angle], dim=-1), inverse=True,
                                       length=mixture_wave.shape[-1])
        reconstructed_wave = reconstructed_wave * (2. ** 15)
        reconstructed_wave = reconstructed_wave[:mixture_wave.shape[0]]

        if self.fit_wav:
            if reconstructed_wave.ndim == 1:
                if self.rescale:
                    clean_wave = clean_wave_and_length[0].to(torch.float32)
                    rescale = torch.sum(clean_wave * reconstructed_wave) / torch.sum(
                        reconstructed_wave * reconstructed_wave)
                    reconstructed_wave = rescale * reconstructed_wave
                value_max = reconstructed_wave.abs().max()
                if value_max > 2. ** 15:
                    reconstructed_wave = reconstructed_wave * (2. ** 14) / value_max

            if reconstructed_wave.ndim == 2:
                if self.rescale:
                    clean_wave = clean_wave_and_length[0].to(torch.float32)
                    rescale = (clean_wave * reconstructed_wave).sum(dim=1, keepdim=True) / (
                            reconstructed_wave * reconstructed_wave).sum(dim=1, keepdim=True)
                    reconstructed_wave = rescale * reconstructed_wave
                value_max = reconstructed_wave.abs().max(dim=1, keepdim=True)[0]
                value_clip = torch.where(value_max > 2. ** 15, (2. ** 14) * value_max.new_ones(value_max.shape),
                                         value_max)
                reconstructed_wave = reconstructed_wave * value_clip / value_max
            reconstructed_wave = reconstructed_wave.to(torch.int16)
        return reconstructed_wave, length
