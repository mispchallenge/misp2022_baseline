#!/usr/bin/env python
# -- coding: UTF-8 
import os
import glob
import argparse
from tqdm import tqdm
from multiprocessing import Pool


from tool.data_io import safe_load, safe_store


# wav.scp <recording-id> <extended-filename>
def prepare_wav_scp(file_pattern, store_dir):
    all_file_lines = []
    file_path_list = glob.glob(file_pattern)
    for file_path in file_path_list:
        record_id = os.path.splitext(os.path.split(file_path)[-1])[0]
        all_file_lines.append('{} {}'.format(record_id, file_path))
    safe_store(file=os.path.join(store_dir, 'wav.scp'), data=sorted(all_file_lines), mode='cover', ftype='txt')
    return None


# segments <utterance-id> <recording-id> <segment-begin> <segment-end>
# text <utterance-id> <words>
# utt2spk <utterance-id> <speaker-id>
def prepare_segments_text_utt2spk_worker(transcription_pattern, store_dir, processing_id=None, processing_num=None):    
    segments_lines = []
    text_sentence_lines = []
    utt2spk_lines = []
    tier_name = '内容层'
    rejected_text_list = ['<NOISE>', '<DEAF>']
    punctuation_list = [' ','。','.','?','？','!','！','，',',','、']
    sound_list = ['呃', '啊', '噢', '嗯', '唉']
    min_duration = 0.04

    file_lines = sorted(safe_load(file=os.path.join(store_dir, 'wav.scp'), ftype='txt'))
    file_ids = []
    for file_line in file_lines:
        file_ids.append(file_line.split(' ')[0])
    file_ids = sorted(file_ids)
    transcription_files = sorted(glob.glob(transcription_pattern))
    # 
    transcription2id_dic = {}
    # if len(file_ids) == len(transcription_files):
    #     for transcription_file, file_id in zip(transcription_files, file_ids):
    #         transcription2id_dic[transcription_file] = file_id
    # else:
    transcription_pool = []
    while file_ids:
        file_id = file_ids.pop(0)
        while transcription_files or transcription_pool:
            if len(transcription_pool) == 0:
                transcription_file = transcription_files.pop(0)
            else:
                transcription_file = transcription_pool.pop(0)
            file_id_split = file_id.split('_')
            file_id_split.pop(4)
            transcription_filename_split = os.path.splitext(os.path.split(transcription_file)[-1])[0].split('_')
            transcription_filename_split.pop(4)
            if '_'.join(file_id_split) in '_'.join(transcription_filename_split):
                transcription2id_dic[transcription_file] = file_id
            else:
                transcription_pool.append(transcription_file)
                break            
        
    for transcription_idx in tqdm(range(len(sorted([*transcription2id_dic.keys()])))):
        if processing_id is None:
            processing_token = True
        else:
            if transcription_idx % processing_num == processing_id:
                processing_token = True
            else:
                processing_token = False
        if processing_token:
            transcription_file = sorted([*transcription2id_dic.keys()])[transcription_idx]
            file_id = transcription2id_dic[transcription_file]
            transcription_filename = os.path.splitext(os.path.split(transcription_file)[-1])[0]
            speaker = transcription_filename.split('_')[-1]
            
            tg = safe_load(file=transcription_file, ftype='textgrid')
            target_tier = False
            for tier in tg.tiers:
                if tier.name == tier_name:
                    target_tier = tier
            if not target_tier:
                raise ValueError('no tier: {}'.format(tier_name))
            for interval in target_tier.intervals:
                if interval.text not in rejected_text_list and interval.xmax - interval.xmin >= min_duration:
                    start_stamp = interval.xmin - interval.xmin % 0.04
                    start_stamp = round(start_stamp, 2)
                    end_stamp = interval.xmax + 0.04 - interval.xmax % 0.04 if interval.xmax % 0.04 != 0 else interval.xmax
                    end_stamp = round(end_stamp, 2)
                    utterance_id = 'S{}_{}'.format(speaker, '_'.join(transcription_filename.split('_')[:-2])) + '_' + \
                                    '{0:06d}'.format(int(round(start_stamp*100, 0))) + '-' + \
                                    '{0:06d}'.format(int(round(end_stamp*100, 0)))
                    text = interval.text
                    for punctuation in punctuation_list:
                        text = text.replace(punctuation, '')
                    if text not in sound_list:
                        segments_lines.append('{} {} {} {}'.format(utterance_id, file_id, start_stamp, end_stamp))
                        text_sentence_lines.append('{} {}'.format(utterance_id, text))
                        utt2spk_lines.append('{} S{}'.format(utterance_id, speaker)) 
    return [segments_lines, text_sentence_lines, utt2spk_lines]


def prepare_segments_text_utt2spk_manager(transcription_pattern, store_dir, processing_num=1):
    if processing_num > 1:
        pool = Pool(processes=processing_num)
        all_result = []
        for i in range(processing_num):
            part_result = pool.apply_async(prepare_segments_text_utt2spk_worker, kwds={
                'transcription_pattern': transcription_pattern, 'store_dir': store_dir, 'processing_id': i, 
                'processing_num': processing_num})
            all_result.append(part_result)
        pool.close()
        pool.join()
        segments_lines, text_sentence_lines, utt2spk_lines = [], [], []
        for item in all_result:
            part_segments_lines, part_text_sentence_lines, part_utt2spk_lines = item.get()
            segments_lines += part_segments_lines
            text_sentence_lines += part_text_sentence_lines
            utt2spk_lines += part_utt2spk_lines
    else:
        segments_lines, text_sentence_lines, utt2spk_lines = prepare_segments_text_utt2spk_worker(
            transcription_pattern=transcription_pattern, store_dir=store_dir)

    safe_store(file=os.path.join(store_dir, 'segments'), data=sorted([*set(segments_lines)]), mode='cover', ftype='txt')
    safe_store(file=os.path.join(store_dir, 'text_sentence'), data=sorted([*set(text_sentence_lines)]), mode='cover', ftype='txt')
    safe_store(file=os.path.join(store_dir, 'utt2spk'), data=sorted([*set(utt2spk_lines)]), mode='cover', ftype='txt')
    return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser('')
    parser.add_argument('file_pattern', type=str, default='', help='directory of wav')
    parser.add_argument('transcription_pattern', type=str, default='', help='directory of transcription')
    parser.add_argument('store_dir', type=str, default='data/train_far', help='set types')
    parser.add_argument('-nj', type=int, default=15, help='number of process')
    args = parser.parse_args()

    print('Preparing wav.scp with {}'.format(args.file_pattern))
    prepare_wav_scp(file_pattern=args.file_pattern, store_dir=args.store_dir)
    print('Preparing segments,text_sentence,utt2spk with {}'.format(args.transcription_pattern))
    prepare_segments_text_utt2spk_manager(transcription_pattern=args.transcription_pattern, store_dir=args.store_dir, processing_num=args.nj)

    # parser = argparse.ArgumentParser('')
    # parser.add_argument('file_pattern', type=str, default='', help='directory of wav')
    # parser.add_argument('store_dir', type=str, default='data/train_far', help='set types')
    # parser.add_argument('-nj', type=int, default=15, help='number of process')
    # args = parser.parse_args()

    # print('Preparing wav.scp with {}'.format(args.file_pattern))
    # prepare_wav_scp(file_pattern=args.file_pattern, store_dir=args.store_dir)