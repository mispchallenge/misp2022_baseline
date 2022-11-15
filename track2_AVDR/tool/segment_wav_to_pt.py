#!/usr/bin/env python
# _*_ coding: UTF-8 _*_
import os
import codecs
import argparse
import torch
from tqdm import tqdm
from scipy.io import wavfile
from multiprocessing import Pool


def segment_wav2pt_worker(overall2segment, processing_id=None, processing_num=None):
    overall_wav_keys = sorted([*overall2segment.keys()])
    unavailable_segments_files = []
    for wav_idx in tqdm(range(len(overall_wav_keys)), leave=True, desc='0' if processing_id is None else str(processing_id)):
        if processing_id is None:
            processing_token = True
        else:
            if wav_idx % processing_num == processing_id:
                processing_token = True
            else:
                processing_token = False
        if processing_token:
            sample_rate, data = wavfile.read(overall_wav_keys[wav_idx])
            for segment_info in overall2segment[overall_wav_keys[wav_idx]]:
                if not os.path.exists(segment_info[2]):
                    if not os.path.exists(os.path.split(segment_info[2])[0]):
                        os.makedirs(os.path.split(segment_info[2])[0], exist_ok=True)
                    start = int(round(sample_rate * segment_info[0]))
                    end = int(round(sample_rate * segment_info[1]))
                    if start < end <= len(data):
                        torch.save(torch.from_numpy(data[start: end]), segment_info[2])
                    else:
                        unavailable_segments_files.append('{} {} {}, {} {}'.format(segment_info[2], start, end, overall_wav_keys[wav_idx], len(data)))
    return unavailable_segments_files


def segment_wav2pt_manager(data_dir, output_dir, processing_num=1):
    # overall2segment
    with codecs.open(os.path.join(data_dir, 'wav.scp'), 'r') as handle:
            lines_content = handle.readlines()
    wav_lines = [*map(lambda x: x[:-1] if x[-1] in ['\n'] else x, lines_content)]
    wav_dic = {}
    for wav_line in wav_lines:
        name, path = wav_line.split(' ')
        wav_dic[name] = path
    
    with codecs.open(os.path.join(data_dir, 'segments'), 'r') as handle:
            lines_content = handle.readlines()
    segment_lines = [*map(lambda x: x[:-1] if x[-1] in ['\n'] else x, lines_content)]
    overall2segment = {}
    for segment_line in segment_lines:
        segment_name, wav_name, start, end = segment_line.split(' ')
        start, end = float(start), float(end)
        if wav_dic[wav_name] in overall2segment:
            overall2segment[wav_dic[wav_name]].append([start, end, os.path.join(output_dir, '{}.pt'.format(segment_name))])
        else:
            overall2segment[wav_dic[wav_name]] = [[start, end, os.path.join(output_dir, '{}.pt'.format(segment_name))]]
    
    if processing_num > 1:
        all_result = []
        pool = Pool(processes=processing_num)
        for i in range(processing_num):
            part_result = pool.apply_async(segment_wav2pt_worker, kwds={'overall2segment': overall2segment, 'processing_id': i, 'processing_num': processing_num})
            all_result.append(part_result)
        pool.close()
        pool.join()
        unavailable_segments_files = []
        for item in all_result:
            unavailable_segments_files += item.get()
    else:
        unavailable_segments_files = segment_wav2pt_worker(overall2segment=overall2segment)
    
    segment_log = [
        'There are {} wavs, generating {} segments, success {}, fail {}:'.format(len(wav_dic), len(segment_lines), 
                                                                                 len(segment_lines) - len(unavailable_segments_files), len(unavailable_segments_files)),
        *unavailable_segments_files
        ]
    with codecs.open(os.path.join(output_dir, 'segment.log'), 'w') as handle:
        handle.write(''.join([*map(lambda x: x if x[-1] in ['\n'] else '{}\n'.format(x), segment_log)]))
    return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser('')
    parser.add_argument('data_dir', type=str, default='./local/tmp/wpe.scp', help='list file of wav, format is scp')
    parser.add_argument('output_dir', type=str, default='wpe', help='output wpe data root')
    parser.add_argument('-nj', type=int, default='1', help='number of process')
    args = parser.parse_args()
    segment_wav2pt_manager(data_dir=args.data_dir, output_dir=args.output_dir, processing_num=args.nj)
 