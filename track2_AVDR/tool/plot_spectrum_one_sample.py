#!/usr/bin/env python
# _*_ coding: UTF-8 _*_
import os
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import librosa
import argparse
from scipy.io import wavfile

user_path = os.path.expanduser('~')
plt.switch_backend('agg')


def stft(x, hop_length, n_fft=None, length=None, is_complex=True):
    window = 'hanning'
    if n_fft is None:
        if not is_complex:
            x = x[0]*x[1]
        n_fft = 2*(x.shape[1] - 1)
        x = x.T
        if length is None:
            y = librosa.core.istft(x, hop_length=hop_length, win_length=n_fft, window=window, center=False)
        else:
            y = librosa.core.istft(x, hop_length=hop_length, win_length=n_fft, window=window, center=False,
                                   length=length+n_fft-hop_length)
        return y[n_fft-hop_length:]
    else:
        x = x.astype('float')
        x_len = x.shape[0]
        if x_len % hop_length == 0:
            pad_width = (n_fft - hop_length, 0)
        else:
            pad_width = (n_fft-hop_length, hop_length - x_len % hop_length)
        x = np.pad(x, pad_width=pad_width, mode='constant', constant_values=(0, 0))
        y = librosa.core.stft(x, n_fft=n_fft, hop_length=hop_length, win_length=n_fft, window=window, center=False)
        y = y.T
        if is_complex:
            return y
        else:
            return librosa.core.magphase(y, power=1)


def spectrum_plotter(pt_path, fig_path, nfft=400, nshift=160):
    os.makedirs(os.path.split(fig_path)[0], exist_ok=True)
    figure = plt.figure()
    wave_np = torch.load(pt_path).numpy().astype('float')
    wavfile.write(fig_path.replace('.png', '.wav'), 16000, wave_np.astype('int16'))
    wave_np = wave_np/(max(abs(wave_np)))
    spectrum_np, _ = stft(x=wave_np, hop_length=nshift, n_fft=nfft, length=None, is_complex=False)
    spectrum_np = spectrum_np[1:][:]
    spectrum_np = spectrum_np.transpose((1, 0))
    spectrum_np = spectrum_np[::-1][:]
    spectrum_np = np.log10(np.where(spectrum_np == 0, np.finfo(float).eps, spectrum_np))
    plt.imshow(spectrum_np, cmap='hot')
    # plt.specgram(wave_np, Fs=16000, scale_by_freq=True, sides='default')
    plt.ylabel('Frequency')
    plt.xlabel('Time')
    plt.close('all')
    figure.savefig(fig_path, dpi=330, bbox_inches='tight')
    return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser('train')
    parser.add_argument('pt_path', type=str, default=None, help='yaml file about plot config')
    parser.add_argument('fig_path', type=str, default='spectrum_0', help='select plot name')
    args = parser.parse_args()
    spectrum_plotter(pt_path=args.pt_path, fig_path=args.fig_path)
