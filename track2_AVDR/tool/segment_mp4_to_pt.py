#!/usr/bin/env python
# _*_ coding: UTF-8 _*_
import os
import codecs
import argparse
import torch
import cv2
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool


def segment_mp42pt(mp4_path, segments_path, segments_start, segments_end):
    segments_num = len(segments_start)
    assert segments_num > 0
    assert segments_num == len(segments_end)

    video_capture = cv2.VideoCapture(mp4_path)
    fps = video_capture.get(5)
    total_frames_num = int(video_capture.get(7))
    
    unavailable_segments_files = []
    frame2segment = {}
    for i, segment_path in enumerate(segments_path):
        if not os.path.exists(segment_path):
            if not os.path.exists(os.path.split(segment_path)[0]):
                os.makedirs(os.path.split(segment_path)[0])
            segment_start, segment_end = int(round(segments_start[i]*fps)), int(round(segments_end[i]*fps))
            if segment_start < segment_end <= total_frames_num:
                segment_len = segment_end - segment_start
                for local_frame_idx in range(segment_len):
                    global_frame_idx = segment_start + local_frame_idx
                    if global_frame_idx in frame2segment:
                        frame2segment[global_frame_idx].append([segment_path, local_frame_idx, segment_len])
                    else:
                        frame2segment[global_frame_idx] = [[segment_path, local_frame_idx, segment_len]]
            else:
                unavailable_segments_files.append('{} {} {}, {} {}'.format(segment_path, segment_start, segment_end, mp4_path, total_frames_num))

    if frame2segment:
        segments_buffer = {}
        # segment_video_writer = None
        frames_idx = 0
        # frames_bar = tqdm(total=total_frames_num, leave=True, desc='Frame')
        while video_capture.isOpened():
            ret, frame = video_capture.read()
            if ret and frame2segment:
                if frames_idx in frame2segment:
                    frame_info_list = frame2segment.pop(frames_idx)
                    for frame_info in frame_info_list:
                        if frame_info[1] == 0:
                            assert frame_info[0] not in segments_buffer
                            segments_buffer[frame_info[0]] = [frame]
                        else:
                            segments_buffer[frame_info[0]].append(frame)

                        if frame_info[1] == frame_info[2] - 1:
                            torch.save(torch.from_numpy(np.array(segments_buffer.pop(frame_info[0]))), frame_info[0])
                frames_idx += 1
                # frames_bar.update(1)
            else:
                break
        # frames_bar.close()
        assert not frame2segment
        video_capture.release()
    return unavailable_segments_files


def segment_mp42pt_worker(overall2segment, processing_id=None, processing_num=None):
    unavailable_segments_files = []
    overall_mp4_keys = sorted([*overall2segment.keys()])
    for mp4_idx in tqdm(range(len(overall_mp4_keys)), leave=True, desc='0' if processing_id is None else str(processing_id)):
    # for mp4_idx in range(len(overall_mp4_keys)):
        if processing_id is None:
            processing_token = True
        else:
            if mp4_idx % processing_num == processing_id:
                processing_token = True
            else:
                processing_token = False
        if processing_token:
            unavailable_segments_files += segment_mp42pt(mp4_path=overall_mp4_keys[mp4_idx], **overall2segment[overall_mp4_keys[mp4_idx]])                
    return unavailable_segments_files


def segment_mp42pt_manager(data_dir, output_dir, processing_num=1):
    # overall2segment
    with codecs.open(os.path.join(data_dir, 'wav.scp'), 'r') as handle:
            lines_content = handle.readlines()
    mp4_lines = [*map(lambda x: x[:-1] if x[-1] in ['\n'] else x, lines_content)]
    mp4_dic = {}
    for mp4_line in mp4_lines:
        name, path = mp4_line.split(' ')
        mp4_dic[name] = path
    
    with codecs.open(os.path.join(data_dir, 'segments'), 'r') as handle:
            lines_content = handle.readlines()
    segment_lines = [*map(lambda x: x[:-1] if x[-1] in ['\n'] else x, lines_content)]
    overall2segment = {}
    for segment_line in segment_lines:
        segment_name, mp4_name, start, end = segment_line.split(' ')
        start, end = float(start), float(end)
        if mp4_dic[mp4_name] in overall2segment:
            overall2segment[mp4_dic[mp4_name]]['segments_path'].append(os.path.join(output_dir, '{}.pt'.format(segment_name)))
            overall2segment[mp4_dic[mp4_name]]['segments_start'].append(start)
            overall2segment[mp4_dic[mp4_name]]['segments_end'].append(end)
        else:
            overall2segment[mp4_dic[mp4_name]] = {'segments_path': [os.path.join(output_dir, '{}.pt'.format(segment_name))], 
                                                  'segments_start': [start], 'segments_end': [end]}

    if processing_num > 1:
        all_result = []
        pool = Pool(processes=processing_num)
        for i in range(processing_num):
            part_result = pool.apply_async(segment_mp42pt_worker, kwds={'overall2segment': overall2segment, 'processing_id': i, 'processing_num': processing_num})
            all_result.append(part_result)
        pool.close()
        pool.join()
        unavailable_segments_files = []
        for item in all_result:
            unavailable_segments_files += item.get()
    else:
        unavailable_segments_files = segment_mp42pt_worker(overall2segment=overall2segment)
    
    segment_log = [
        'There are {} mp4s, generating {} segments, success {}, fail {}:'.format(len(mp4_dic), len(segment_lines), 
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
    segment_mp42pt_manager(data_dir=args.data_dir, output_dir=args.output_dir, processing_num=args.nj)
