#!/usr/bin/env python
# _*_ coding: UTF-8 _*_
import time
from tqdm import tqdm
import torch
import torch.nn as nn
import argparse

from tool.data_io import safe_store
from network.network_feature_extract import FeatureExtractor
from loader_audio_visual_pdf import get_data_loader


def lps_cmvn(annotate, n_fft=400, hop_length=160, cmvn=None, **other_params):
    extractor = FeatureExtractor(extractor_type='lps', extractor_setting={
        'n_fft': n_fft, 'hop_length': hop_length, 'win_type': 'hamming', 'win_length': n_fft, 'cmvn': cmvn})
    extractor = nn.DataParallel(extractor).cuda()
    checkout_data_loader, _ = get_data_loader(
        annotate=annotate, items=['far_wave'], batch_size=96, max_batch_size=512, repeat=1, max_duration=100, pad_value=[0],
        hop_duration=6, key_output=False, dynamic=True, bucket_length_multiplier=1.1, shuffle=False, drop_last=False,
        num_workers=4, pin_memory=False, seed=123456, epoch=0, logger='print', distributed=False, target_shape=[[0]])
    lps_mean, lps_std, frames_count = 0., 0., 0.
    for data, length in tqdm(checkout_data_loader):
        lps, frame_num = extractor(data.cuda(), length)
        frame_num = frame_num.long()
        for sample_idx in range(mixture_lps.size(0)):
            avail_lps = lps[sample_idx, :, :frame_num[sample_idx]]
            updated_count = frames_count + frame_num[sample_idx]
            lps_mean = lps_mean * (frames_count/updated_count) + avail_lps.sum(dim=1) / updated_count
            lps_std = lps_std * (frames_count/updated_count) + (avail_lps ** 2).sum(dim=1) / updated_count
            frames_count = updated_count
    lps_std = torch.sqrt(lps_std - lps_mean ** 2)
    return frames_count, lps_mean, lps_std


def fbank_cmvn(annotate, n_fft=512, hop_length=160, cmvn=None, f_min=0, f_max=8000, n_mels=40, sample_rate=16000,
               norm='slaney', preemphasis_coefficient=0.97, vtln=False, vtln_low=0, vtln_high=8000,
               vtln_warp_factor=1.):
    extractor = FeatureExtractor(extractor_type='fbank', extractor_setting={
        'n_fft': n_fft, 'hop_length': hop_length, 'win_type': 'hamming', 'win_length': n_fft, 'cmvn': cmvn,
        'f_min': f_min, 'f_max': f_max, 'n_mels': n_mels, 'sample_rate': sample_rate, 'norm': norm,
        'preemphasis_coefficient': preemphasis_coefficient, 'vtln': vtln, 'vtln_low': vtln_low,
        'vtln_high': vtln_high, 'vtln_warp_factor': vtln_warp_factor})
    # extractor = FilterBank(
    #     n_fft=n_fft, hop_length=hop_length, win_type='hamming', win_length=None, cmvn=cmvn, f_min=f_min, f_max=f_max,
    #     n_mels=n_mels, sample_rate=sample_rate, norm=norm, preemphasis_coefficient=preemphasis_coefficient,
    #     vtln=vtln, vtln_low=vtln_low, vtln_high=vtln_high, vtln_warp_factor=vtln_warp_factor)
    extractor = nn.DataParallel(extractor).cuda()
    checkout_data_loader, _ = get_data_loader(
        annotate=annotate, items=['far_wave'], batch_size=96, max_batch_size=512, repeat=1, max_duration=100, pad_value=[0],
        hop_duration=6, key_output=False, dynamic=True, bucket_length_multiplier=1.1, shuffle=False, drop_last=False,
        num_workers=4, pin_memory=False, seed=123456, epoch=0, logger='print', distributed=False, target_shape=[[0]])
    fbank_mean, fbank_std, frames_count = 0., 0., 0.
    for data, length in tqdm(checkout_data_loader):
        fbank, frame_num = extractor(data.cuda(), length)
        frame_num = frame_num.long()
        for sample_idx in range(fbank.size(0)):
            avail_fbank = fbank[sample_idx, :, :frame_num[sample_idx]]
            updated_count = frames_count + frame_num[sample_idx]
            fbank_mean = fbank_mean * (frames_count / updated_count) + avail_fbank.sum(dim=1) / updated_count
            fbank_std = fbank_std * (frames_count / updated_count) + (avail_fbank ** 2).sum(dim=1) / updated_count
            frames_count = updated_count
    fbank_std = torch.sqrt(fbank_std - fbank_mean ** 2)
    return frames_count, fbank_mean, fbank_std


if __name__ == '__main__':
    parser = argparse.ArgumentParser('feature_cmvn')
    parser.add_argument('step', type=int, nargs='+', default=[0], help='select run step')
    args = parser.parse_args()
    stage = args.step

    if 0 in stage:
        annotate = ['feature/misp2022_avsr/train_far.json']
        start_time = time.time()
        cmvn_output = fbank_cmvn(
            annotate=annotate, n_fft=400, hop_length=160, cmvn=None, f_min=0,
            f_max=8000, n_mels=40, sample_rate=16000, norm='slaney', preemphasis_coefficient=0.97, vtln=False,
            vtln_low=0, vtln_high=8000, vtln_warp_factor=1.)
        safe_store(file='feature/misp2022_avsr/cmvn_wave/cmvn_fbank_htk_far.pt',
                   data={'mean': cmvn_output[1].cpu(), 'std': cmvn_output[2].cpu()}, ftype='torch', mode='cover')
        print('frames: {} time: {}s'.format(cmvn_output[0], time.time() - start_time))
        start_time = time.time()
        cmvn_output = fbank_cmvn(
            annotate=annotate, n_fft=400, hop_length=160, cmvn='feature/misp2022_avsr/cmvn_wave/cmvn_fbank_htk_far.pt', f_min=0,
            f_max=8000, n_mels=40, sample_rate=16000, norm='slaney', preemphasis_coefficient=0.97, vtln=False,
            vtln_low=0, vtln_high=8000, vtln_warp_factor=1.)
        print('frames: {} time: {}s'.format(cmvn_output[0], time.time() - start_time))
        print('mean:', cmvn_output[1])
        print('std: ', cmvn_output[2])

    # if 1 in stage:
    #     snr = '-5_0_5_10_15'
    #     annotate = [
    #         '/yrfs1/intern/hangchen2/feature/TCD-TIMIT_sr_16000_fps_25_noisy_35h/train_snr_{}.json'.format(snr)]
    #     start_time = time.time()
    #     cmvn_output = lps_cmvn(annotate=annotate, n_fft=400, hop_length=160, cmvn=None)
    #     safe_store(file='/yrfs1/intern/hangchen2/feature/TCD-TIMIT_sr_16000_fps_25_noisy_35h/'
    #                     'cmvn/cmvn_lps_snr_{}.pt'.format(snr),
    #                data={'mean': cmvn_output[1].cpu(), 'std': cmvn_output[2].cpu()}, ftype='torch', mode='cover')
    #     print('frames: {} time: {}s'.format(cmvn_output[0], time.time() - start_time))
    #     start_time = time.time()
    #     cmvn_output = lps_cmvn(
    #         annotate=annotate, n_fft=400, hop_length=160,
    #         cmvn='/yrfs1/intern/hangchen2/feature/TCD-TIMIT_sr_16000_fps_25_noisy_35h/'
    #              'cmvn/cmvn_lps_snr_{}.pt'.format(snr))
    #     print('frames: {} time: {}s'.format(cmvn_output[0], time.time() - start_time))
    #     print('mean:', cmvn_output[1])
    #     print('std: ', cmvn_output[2])