#!/usr/bin/env python
# _*_ coding: UTF-8 _*_
import os
import cv2
import json
import time
import codecs
import torch
import argparse
import numpy as np


def crop_frame_roi(frame, roi_bound, roi_size=(96, 96)):
    bound_l = max(roi_bound[3] - roi_bound[1], roi_bound[2] - roi_bound[0], *roi_size)
    
    bound_w_extend = (bound_l - roi_bound[3] + roi_bound[1]) / 2
    x_start = max(int(roi_bound[1] - bound_w_extend), 0)
    x_end = min(int(roi_bound[3] + bound_w_extend), frame.shape[0])
    
    bound_h_extend = (bound_l - roi_bound[2] + roi_bound[0]) / 2
    y_start = max(int(roi_bound[0] - bound_h_extend), 0)
    y_end = min(int(roi_bound[2] + bound_h_extend), frame.shape[1])
    
    roi_frame = frame[x_start: x_end, y_start: y_end, :]
    if x_end - x_start != roi_size[1] or y_end - y_start != roi_size[0] :
        roi_frame = cv2.resize(roi_frame, tuple(roi_size), interpolation=cv2.INTER_AREA)
        # return cv2.cvtColor(resized_roi_frame, cv2.COLOR_BGR2GRAY)
    return roi_frame


def crop_roi(frames_array, roi_bound, roi_size=(96, 96), roi_sum=False):
    frames_num = frames_array.shape[0]
    assert frames_num == roi_bound.shape[0]
    roi_array = []
    if roi_sum:
        min_x_start, max_x_end, min_y_start, max_y_end = roi_bound[:, 1].min(), roi_bound[:, 3].max(), roi_bound[:, 0].min(), roi_bound[:, 2].max()
        for frame_idx in range(frames_num):
            roi_array.append(crop_frame_roi(frame=frames_array[frame_idx], roi_bound=[min_y_start, min_x_start, max_y_end, max_x_end], roi_size=roi_size))
    else:
        for frame_idx in range(frames_num):
            roi_array.append(crop_frame_roi(frame=frames_array[frame_idx], roi_bound=roi_bound[frame_idx], roi_size=roi_size))
    return torch.from_numpy(np.stack(roi_array, axis=0))


def segment_roi_json(roi_json_path, segments_name, segments_start, segments_end, total_frames_num, roi_type='lip', roi_sum=False, segments_speaker=None):
    alpha = 0.5
    with codecs.open(roi_json_path, 'r') as handle:
        roi_dic = json.load(handle)

    def get_from_frame_detection(frame_i, target_id=1):
        if str(frame_i) in roi_dic:
            for roi_info in roi_dic[str(frame_i)]:
                if roi_info['id'] == target_id:
                    if roi_type == 'head':
                        return [roi_info['x1'], roi_info['y1'], roi_info['x2'], roi_info['y2']]
                    elif roi_type == 'lip':
                        return roi_info['lip']
                    else:
                        raise NotImplementedError('unknown roi type: {}'.format(roi_type))         
        return []

    delete_segments_key = []
    segments_roi_bound = {}
    # for _, (name, speaker_id, frame_start, frame_end) in enumerate(zip(segments_name, segments_speaker, segments_start,
    #                                                                    segments_end)):
    for segment_idx, segment_name in enumerate(segments_name):
        segment_start = segments_start[segment_idx]
        segment_end = segments_end[segment_idx]
        if segment_end > total_frames_num:
            delete_segments_key.append(segment_name)
            print('{}: sengment end cross the line, {} but {}, skip'.format(segment_name, segment_end, total_frames_num))
        else:
            segment_roi_bound = []
            segment_roi_idx = []
            for frame_idx in range(segment_start, segment_end):
                if segments_speaker is None:
                    segment_roi_bound.append(get_from_frame_detection(frame_i=frame_idx))
                else:
                    segment_roi_bound.append(get_from_frame_detection(frame_i=frame_idx, target_id=segments_speaker[segment_idx]))
                segment_roi_idx.append(frame_idx)

            frame_roi_exist_num = np.sum([*map(bool, segment_roi_bound)]).item()

            if float(segment_end - segment_start) == 0:
                # zhewang18
                delete_segments_key.append(segment_name)
                print('{}: {}/{} frames have detection result, skip'.format(segment_name, frame_roi_exist_num, segment_end - segment_start))
            elif float(frame_roi_exist_num) / float(segment_end - segment_start) < alpha:
                delete_segments_key.append(segment_name)
                print('{}: {}/{} frames have detection result, skip'.format(segment_name, frame_roi_exist_num, segment_end - segment_start))
            elif frame_roi_exist_num == segment_end - segment_start:
                segment_roi_bound = np.array(segment_roi_bound)
                if roi_sum:
                    segment_roi_bound[:, 1] = segment_roi_bound[:, 1].min()
                    segment_roi_bound[:, 3] = segment_roi_bound[:, 3].max()
                    segment_roi_bound[:, 0] = segment_roi_bound[:, 0].min()
                    segment_roi_bound[:, 2] = segment_roi_bound[:, 2].max()
                segments_roi_bound[segment_name] = segment_roi_bound
                print('{}: {}/{} frames have detection result, prefect'.format(segment_name, frame_roi_exist_num, segment_end - segment_start))
            else:
                print('{}: {}/{} frames have detection result, insert'.format(segment_name, frame_roi_exist_num, segment_end - segment_start))
                i = 1
                forward_buffer = []
                forward_buffer_idx = -1
                while segment_start - i >= 0:
                    if segments_speaker is None:
                        first_frame_detection = get_from_frame_detection(frame_i=segment_start - i)
                    else:
                        first_frame_detection = get_from_frame_detection(frame_i=segment_start - i, target_id=segments_speaker[segment_idx])
                    if first_frame_detection:
                        forward_buffer = first_frame_detection
                        forward_buffer_idx = segment_start - i
                        break
                    else:
                        i += 1

                need_insert_idxes = []
                for i, (frame_idx, frame_roi_bound) in enumerate(zip(segment_roi_idx, segment_roi_bound)):
                    if frame_roi_bound:
                        while need_insert_idxes:
                            need_insert_idx = need_insert_idxes.pop(0)
                            if forward_buffer_idx == -1:
                                segment_roi_bound[need_insert_idx] = frame_roi_bound
                                print(need_insert_idx, segment_roi_bound[need_insert_idx], segment_roi_idx[need_insert_idx], frame_roi_bound, frame_idx)
                            else:
                                segment_roi_bound[need_insert_idx] = (
                                        np.array(forward_buffer) +
                                        (segment_roi_idx[need_insert_idx] - forward_buffer_idx) *
                                        (np.array(frame_roi_bound) - np.array(forward_buffer)) /
                                        (frame_idx - forward_buffer_idx)).astype(np.int64).tolist()
                                print(need_insert_idx, segment_roi_bound[need_insert_idx], segment_roi_idx[need_insert_idx], frame_roi_bound, frame_idx, forward_buffer, forward_buffer_idx)
                        forward_buffer = frame_roi_bound
                        forward_buffer_idx = frame_idx
                    else:
                        need_insert_idxes.append(i)

                if need_insert_idxes:
                    i = 0
                    backward_buffer = []
                    backward_buffer_idx = -1
                    while segment_end + i < total_frames_num:
                        if segments_speaker is None:
                            last_frame_detection = get_from_frame_detection(frame_i=segment_end + i) 
                        else:
                            last_frame_detection = get_from_frame_detection(frame_i=segment_end + i, target_id=segments_speaker[segment_idx])
                        if last_frame_detection:
                            backward_buffer = last_frame_detection
                            backward_buffer_idx = segment_end + i
                            break
                        else:
                            i += 1
                    while need_insert_idxes:
                        need_insert_idx = need_insert_idxes.pop(0)
                        if forward_buffer_idx == -1 and backward_buffer_idx == -1:
                            raise ValueError('no context cannot pad')
                        elif forward_buffer_idx == -1:
                            segment_roi_bound[need_insert_idx] = backward_buffer
                            print(need_insert_idx, segment_roi_bound[need_insert_idx], segment_roi_idx[need_insert_idx], backward_buffer, backward_buffer_idx)
                        elif backward_buffer_idx == -1:
                            segment_roi_bound[need_insert_idx] = forward_buffer
                            print(need_insert_idx, segment_roi_bound[need_insert_idx], segment_roi_idx[need_insert_idx], forward_buffer, forward_buffer_idx)
                        else:
                            segment_roi_bound[need_insert_idx] = (
                                    np.array(forward_buffer) +
                                    (segment_roi_idx[need_insert_idx] - forward_buffer_idx) *
                                    (np.array(backward_buffer) - np.array(forward_buffer)) /
                                    (backward_buffer_idx - forward_buffer_idx)).astype(np.int64).tolist()
                            print(need_insert_idx, segment_roi_bound[need_insert_idx], segment_roi_idx[need_insert_idx], backward_buffer, backward_buffer_idx, forward_buffer, forward_buffer_idx)
                assert not need_insert_idxes
                segment_roi_bound = np.array(segment_roi_bound)
                if roi_sum:
                    segment_roi_bound[:, 1] = segment_roi_bound[:, 1].min()
                    segment_roi_bound[:, 3] = segment_roi_bound[:, 3].max()
                    segment_roi_bound[:, 0] = segment_roi_bound[:, 0].min()
                    segment_roi_bound[:, 2] = segment_roi_bound[:, 2].max()
                segments_roi_bound[segment_name] = segment_roi_bound
    return segments_roi_bound, delete_segments_key


def segment_video_roi_json(video_path, roi_json_path, roi_store_dir, segments_name, segments_start, segments_end, roi_type='head', segments_speaker=None, roi_size=[96, 96], roi_sum=False):
    segments_num = len(segments_start)
    assert segments_num > 0
    assert segments_num == len(segments_end)
    video_capture = cv2.VideoCapture(video_path)
    total_frames_num = int(video_capture.get(7))
    print('using roi info from {}, all {} frames, generating {} segments'.format(roi_json_path, total_frames_num, segments_num))

    segments_roi_bound, delete_segments_key = segment_roi_json(roi_json_path=roi_json_path, segments_name=segments_name, segments_start=segments_start, segments_end=segments_end, 
                                                               total_frames_num=total_frames_num, roi_type=roi_type, roi_sum=roi_sum, segments_speaker=segments_speaker)
    frame2segment_roi_bound = {}
    for i, segment_name in enumerate(segments_name):
        if segment_name not in delete_segments_key:
            segment_path = os.path.join(os.path.abspath(roi_store_dir), '{}.pt'.format(segment_name))
            if not os.path.exists(segment_path):
                for in_frame_idx in range(segments_end[i] - segments_start[i]):
                    if segments_start[i] + in_frame_idx in frame2segment_roi_bound:
                        frame2segment_roi_bound[segments_start[i] + in_frame_idx].append(
                            [segment_path, in_frame_idx, segments_end[i] - segments_start[i], segments_roi_bound[segment_name][in_frame_idx]])
                    else:
                        frame2segment_roi_bound[segments_start[i] + in_frame_idx] = [
                            [segment_path, in_frame_idx, segments_end[i] - segments_start[i], segments_roi_bound[segment_name][in_frame_idx]]]
    if not os.path.exists(roi_store_dir):
            os.makedirs(roi_store_dir, exist_ok=True)

    if frame2segment_roi_bound:
        segments_roi_frames_buffer = {}
        # segment_video_writer = None
        frames_idx = 0
        # frames_bar = tqdm(total=total_frames_num, leave=True, desc='Frame')
        while video_capture.isOpened():
            ret, frame = video_capture.read()
            if ret and frame2segment_roi_bound:
                if frames_idx in frame2segment_roi_bound:
                    frame_info_list = frame2segment_roi_bound.pop(frames_idx)
                    for frame_info in frame_info_list:
                        if frame_info[1] == 0:
                            assert frame_info[0] not in segments_roi_frames_buffer
                            segments_roi_frames_buffer[frame_info[0]] = [crop_frame_roi(frame=frame, roi_bound=frame_info[3], roi_size=roi_size)]
                        else:
                            segments_roi_frames_buffer[frame_info[0]].append(crop_frame_roi(frame=frame, roi_bound=frame_info[3], roi_size=roi_size))

                        if frame_info[1] == frame_info[2] - 1:
                            torch.save(torch.from_numpy(np.array(segments_roi_frames_buffer.pop(frame_info[0]))), frame_info[0])

                frames_idx += 1
                # frames_bar.update(1)
            else:
                break
        # frames_bar.close()
        assert not frame2segment_roi_bound
        video_capture.release()
        print('skip {} segments: {}'.format(len(delete_segments_key), ','.join(delete_segments_key)))
    return None


def input_interface(data_root, roi_json_dir, need_speaker=True, roi_type='head', roi_size=[96, 96], roi_sum=False):
    fps = 25

    video_dic = {}
    roi_json_dic = {}
    with codecs.open(os.path.join(data_root, 'wav.scp'), 'r') as handle:
        lines_content = handle.readlines()
    for video_line in [*map(lambda x: x[:-1] if x[-1] in ['\n'] else x, lines_content)]:
        name, path = video_line.split(' ')
        video_dic[name] = path
        roi_json_dic[name] = os.path.join(roi_json_dir, '{}.json'.format(os.path.splitext(os.path.split(path)[-1])[0]))
    vid2spk_dic = {}
    if need_speaker:    
        with codecs.open(os.path.join(data_root, 'utt2spk'), 'r') as handle:
            lines_content = handle.readlines()
        for vid2spk_line in [*map(lambda x: x[:-1] if x[-1] in ['\n'] else x, lines_content)]:
            name, speaker = vid2spk_line.split(' ')
            vid2spk_dic[name] = int(speaker[1:])  
    segments_dic = {}
    # dict_keys(['roi_json_path', 'segments_name', 'segments_start', 'segments_end', 'roi_type', 'roi_size', 'roi_sum', 'segments_speaker'])
    with codecs.open(os.path.join(data_root, 'segments'), 'r') as handle:
        lines_content = handle.readlines()
    for segment_line in [*map(lambda x: x[:-1] if x[-1] in ['\n'] else x, lines_content)]:
        segment_name, name, start, end = segment_line.split(' ')
        if video_dic[name] not in segments_dic:
            segments_dic[video_dic[name]] = {
                'roi_json_path': roi_json_dic[name],
                'segments_name': [segment_name], 
                'segments_start': [int(np.around(float(start) * fps))],
                'segments_end': [int(np.around(float(end) * fps))],
                'roi_type': roi_type,
                'roi_size': roi_size,
                'roi_sum': roi_sum}
            if need_speaker:
                segments_dic[video_dic[name]]['segments_speaker'] = [vid2spk_dic[segment_name]]
        else:
            segments_dic[video_dic[name]]['segments_name'].append(segment_name)
            segments_dic[video_dic[name]]['segments_start'].append(int(np.around(float(start) * fps)))
            segments_dic[video_dic[name]]['segments_end'].append(int(np.around(float(end) * fps)))
            if need_speaker:
                segments_dic[video_dic[name]]['segments_speaker'].append(vid2spk_dic[segment_name])
    # import pdb; pdb.set_trace()
    return segments_dic


if __name__ == '__main__':
    parser = argparse.ArgumentParser('prepare_far_video_roi')
    parser.add_argument('data_root', type=str, default='data/test_far_video', help='root directory of dataset')
    parser.add_argument('roi_json_dir', type=str, default='/path/roi', help='directory of roi json')
    parser.add_argument('roi_store_dir', type=str, default='data/test_far_video', help='store directory of roi npz')
    parser.add_argument('-ji', type=int, default=0, help='index of process')
    parser.add_argument('-nj', type=int, default=1, help='number of process')
    parser.add_argument('--roi_type', type=str, default='lip', help='roi type')
    parser.add_argument('--roi_size', type=int, nargs="+", default=[96, 96], help='roi size')
    parser.add_argument('--need_speaker', default=True, action='store_true', help='need speaker')
    parser.add_argument('--roi_sum', default=False, action='store_true', help='roi sum')
    
    args = parser.parse_args()

    all_input_params = input_interface(data_root=args.data_root, roi_json_dir=args.roi_json_dir, need_speaker=args.need_speaker, roi_type=args.roi_type, roi_size=args.roi_size, roi_sum=args.roi_sum)
    all_sentences = sorted([*all_input_params.keys()])
    nj = args.nj
    ji = args.ji if nj > 1 else 0
    start_time = time.time()

    for sentence_idx, sentence_path in enumerate(all_sentences):
        if sentence_idx % nj == ji:
            print('#'*72)
            print('processing {}'.format(sentence_path))
            segment_video_roi_json(video_path=sentence_path, roi_store_dir=args.roi_store_dir, **all_input_params[sentence_path])
            # video_path, roi_store_dir, roi_json_path, segments_name, segments_start, segments_end, roi_type='head', segments_speaker=None, roi_size=[96, 96]
            in_len = (len(all_sentences) - ji) // nj  # 32
            in_index = (sentence_idx - ji) // nj
            current_dur = round((time.time() - start_time) / 60., 2)
            print('{}/{} {}/{} min'.format(in_index, in_len, current_dur, round(current_dur * (in_len + 1) / (in_index + 1), 2)))

# AVSR
# python local/segment_video_roi.py --roi_type lip --roi_size 96 96 --need_speaker --roi_sum /raw7/cv1/hangchen2/misp2022/AVSR/eval_far_video /raw7/cv1/hangchen2/misp2022/Released/roi_coordinates/eval/far/lip /raw7/cv1/hangchen2/misp2022/Released/roi_coordinates/eval/far/lip_segment/pt
# python local/segment_video_roi.py --roi_type lip --roi_size 96 96 --need_speaker --roi_sum /raw7/cv1/hangchen2/misp2022/AVSR/dev_far_video /raw7/cv1/hangchen2/misp2022/Released/roi_coordinates/dev/far/lip /raw7/cv1/hangchen2/misp2022/Released/roi_coordinates/dev/far/lip_segment/pt
# python local/generate_key2shape.py /raw7/cv1/hangchen2/misp2022/Released/roi_coordinates/dev/far/lip_segment/pt/'*.pt'

# AVSDR
# python local/segment_video_roi.py --roi_type lip --roi_size 96 96 --need_speaker --roi_sum /raw7/cv1/hangchen2/misp2022/AVSD_AVSR/dev_far_video /raw7/cv1/hangchen2/misp2022/Released/roi_coordinates/dev/far/lip /raw7/cv1/hangchen2/misp2022/Released/roi_coordinates/dev/far/lip_AVSDR_segment/pt
# python local/generate_key2shape.py /raw7/cv1/hangchen2/misp2022/Released/roi_coordinates/dev/far/lip_AVSDR_segment/pt/'*.pt'
# python local/segment_video_roi.py --roi_type lip --roi_size 96 96 --need_speaker --roi_sum /raw7/cv1/hangchen2/misp2022/AVSD_AVSR/eval_far_video /raw7/cv1/hangchen2/misp2022/Released/roi_coordinates/eval/far/lip /raw7/cv1/hangchen2/misp2022/Released/roi_coordinates/eval/far/lip_AVSDR_segment/pt
# python local/generate_key2shape.py /raw7/cv1/hangchen2/misp2022/Released/roi_coordinates/eval/far/lip_AVSDR_segment/pt/'*.pt'

# VSD_ASR/AVSR
# python local/segment_video_roi.py --roi_type lip --roi_size 96 96 --need_speaker --roi_sum /yrfs1/intern/zhewang18/zhewang/misp2021_avsr/data/misp2022/VSD_AVSR/dev_far_video /yrfs1/intern/zhewang18/zhewang/misp2021_avsr/data/misp2022/Released/roi_coordinates/dev/far/lip /yrfs1/intern/zhewang18/zhewang/misp2021_avsr/data/misp2022/Released/roi_coordinates/dev/far/lip_VSD_segment/pt
# python local/generate_key2shape.py /yrfs1/intern/zhewang18/zhewang/misp2021_avsr/data/misp2022/Released/roi_coordinates/dev/far/lip_VSD_segment/pt/'*.pt'
# python local/segment_video_roi.py --roi_type lip --roi_size 96 96 --need_speaker --roi_sum /yrfs1/intern/zhewang18/zhewang/misp2021_avsr/data/misp2022/VSD_AVSR/eval_far_video /yrfs1/intern/zhewang18/zhewang/misp2021_avsr/data/misp2022/Released/roi_coordinates/eval/far/lip /yrfs1/intern/zhewang18/zhewang/misp2021_avsr/data/misp2022/Released/roi_coordinates/eval/far/lip_VSD_segment/pt
# python local/generate_key2shape.py /yrfs1/intern/zhewang18/zhewang/misp2021_avsr/data/misp2022/Released/roi_coordinates/eval/far/lip_VSD_segment/pt/'*.pt'

# ASD
# python local/segment_video_roi.py --roi_type lip --roi_size 96 96 --need_speaker --roi_sum /yrfs1/intern/zhewang18/zhewang/misp2021_avsr/data/misp2022/ASD_ASR/dev_far_video /yrfs1/intern/zhewang18/zhewang/misp2021_avsr/data/misp2022/Released/roi_coordinates/dev/far/lip /yrfs1/intern/zhewang18/zhewang/misp2021_avsr/data/misp2022/Released/roi_coordinates/dev/far/lip_ASD_segment/pt
# python local/generate_key2shape.py /yrfs1/intern/zhewang18/zhewang/misp2021_avsr/data/misp2022/Released/roi_coordinates/dev/far/lip_ASD_segment/pt/'*.pt'
# python local/segment_video_roi.py --roi_type lip --roi_size 96 96 --need_speaker --roi_sum /yrfs1/intern/zhewang18/zhewang/misp2021_avsr/data/misp2022/ASD_ASR/eval_far_video /yrfs1/intern/zhewang18/zhewang/misp2021_avsr/data/misp2022/Released/roi_coordinates/eval/far/lip /yrfs1/intern/zhewang18/zhewang/misp2021_avsr/data/misp2022/Released/roi_coordinates/eval/far/lip_ASD_segment/pt
# python local/generate_key2shape.py /yrfs1/intern/zhewang18/zhewang/misp2021_avsr/data/misp2022/Released/roi_coordinates/eval/far/lip_ASD_segment/pt/'*.pt'