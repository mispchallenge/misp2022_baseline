#!/usr/bin/env python
# _*_ coding: UTF-8 _*_
import os
import cv2
import json
import torch
import codecs
import numpy as np
from scipy.io import wavfile


def crop_resize_gray_frame(frame, roi_bound=None, roi_size=(96, 96), gray=True):
    if roi_bound is None:
        cropped_frame = frame
    else:
        assert len(roi_bound) == 4
        bound_l = max(roi_bound[3] - roi_bound[1], roi_bound[2] - roi_bound[0])
        bound_h_extend = (bound_l - roi_bound[2] + roi_bound[0]) / 2
        bound_w_extend = (bound_l - roi_bound[3] + roi_bound[1]) / 2
        x_start, x_end = int(roi_bound[1] - bound_w_extend), int(roi_bound[3] + bound_w_extend)
        if x_start < 0:
            x_start = 0
        if x_end > frame.shape[0]:
            x_end = frame.shape[0]
        y_start, y_end = int(roi_bound[0] - bound_h_extend), int(roi_bound[2] + bound_h_extend)
        if y_start < 0:
            y_start = 0
        if y_end > frame.shape[1]:
            y_end = frame.shape[1]
        cropped_frame = frame[x_start: x_end, y_start: y_end, :]
    if roi_size is None:
        resized_frame = cropped_frame
    else:
        assert len(roi_size) == 2
        resized_frame = cv2.resize(cropped_frame, roi_size, interpolation=cv2.INTER_AREA)
    if gray:
        return cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
    return resized_frame


def segment_crop_video(video_path, segments_name, segments_start, segments_end, store_dir, store_postfix, 
                       segments_roi_bound=None, roi_size=None, gray=False):
    segments_num = len(segments_start)
    assert segments_num > 0
    assert segments_num == len(segments_end)

    video_capture = cv2.VideoCapture(video_path)
    total_frames_num = int(video_capture.get(7))
    # print('all {} frames, generating {} segments'.format(total_frames_num, segments_num))
    # if segments_roi_bound is not None:
    #     print('crop roi by the given roi bound')

    frame2segment_info = {}
    for i, segment_name in enumerate(segments_name):
        if segments_roi_bound is not None and segment_name not in segments_roi_bound:
            pass
        else:
            if segments_end[i] < total_frames_num:
                for in_frame_idx in range(segments_end[i] - segments_start[i]):
                    segment_info = [segment_name, in_frame_idx, segments_end[i] - segments_start[i]]
                    if segments_roi_bound is not None:
                        segment_info.append(segments_roi_bound[segment_name][in_frame_idx])
                    else:
                        segment_info.append(None)
                    
                    if segments_start[i] + in_frame_idx in frame2segment_info:
                        frame2segment_info[segments_start[i] + in_frame_idx].append(segment_info)
                    else:
                        frame2segment_info[segments_start[i] + in_frame_idx] = [segment_info]

    segments_roi_frames_buffer = {}

    # segment_video_writer = None
    frames_idx = 0
    if not os.path.exists(store_dir):
        os.makedirs(store_dir, exist_ok=True)

    # frames_bar = tqdm(total=total_frames_num, leave=True, desc='Frame')
    while video_capture.isOpened():
        ret, frame = video_capture.read()
        if ret:
            if frames_idx in frame2segment_info:
                for frame_info in frame2segment_info[frames_idx]:
                    name, idx, length, bound = frame_info
                    if idx == 0:
                        assert name not in segments_roi_frames_buffer
                        segments_roi_frames_buffer[name] = [
                            crop_resize_gray_frame(frame=frame, roi_bound=bound, roi_size=roi_size, gray=gray)]
                    else:
                        segments_roi_frames_buffer[name].append(
                            crop_resize_gray_frame(frame=frame, roi_bound=bound, roi_size=roi_size, gray=gray))

                    if idx == length - 1:
                        # store as pt file
                        torch.save(torch.tensor(segments_roi_frames_buffer.pop(name)), 
                                   os.path.join(store_dir, '{}{}.pt'.format(name, store_postfix)))
            frames_idx += 1
            # frames_bar.update(1)
        else:
            break
    # frames_bar.close()
    video_capture.release()
    return None


def segment_audio(audio_path, segments_name, segments_start, segments_end, store_dir, store_postfix):
    segments_num = len(segments_start)
    assert segments_num > 0
    assert segments_num == len(segments_end)
    _, audio_data = wavfile.read(audio_path)
    total_samples_num = audio_data.shape[0]
    # print('all {} samples, generating {} segments'.format(total_samples_num, segments_num))
    if not os.path.exists(store_dir):
        os.makedirs(store_dir, exist_ok=True)
    for i, segment_name in enumerate(segments_name):
        if segments_end[i] < total_samples_num:
            segment_array = audio_data[segments_start[i]: segments_end[i]]
            torch.save(torch.from_numpy(segment_array), os.path.join(store_dir, '{}{}.pt'.format(segment_name, store_postfix)))
    return None


def segment_roi_json(roi_json_path, segments_name, segments_speaker, segments_start, segments_end):
    
    with codecs.open(roi_json_path, 'r') as handle:
        roi_dic = json.load(handle)
    total_frames_num = sorted([*map(int, [*roi_dic.keys()])])[-1] + 1

    def get_from_frame_detection(frame_i, target_id):
        if str(frame_i) in roi_dic:
            for roi_info in roi_dic[str(frame_i)]:
                if roi_info['id'] == target_id:
                    return [roi_info['x1'], roi_info['y1'], roi_info['x2'], roi_info['y2']]
        return []

    segments_roi_bound = {}
    for _, (name, speaker_id, frame_start, frame_end) in enumerate(zip(segments_name, segments_speaker, segments_start,
                                                                       segments_end)):
        if frame_end >= total_frames_num:
            # print('{}: segment cross the line, {} but {}, skip'.format(name, frame_end, total_frames_num))
            pass
        else:
            segment_roi_bound = []
            segment_roi_idx = []
            for frame_idx in range(frame_start, frame_end):
                segment_roi_bound.append(get_from_frame_detection(frame_idx, speaker_id))
                segment_roi_idx.append(frame_idx)

            frame_roi_exist_num = np.sum([*map(bool, segment_roi_bound)]).item()

            if float(frame_roi_exist_num) / float(frame_end - frame_start) < 0.5:
                # print('{}: {}/{} frames have detection result, skip'.format(name, frame_roi_exist_num,
                #                                                             frame_end - frame_start))
                pass
            elif frame_roi_exist_num == frame_end - frame_start:
                segments_roi_bound[name] = segment_roi_bound
                # print('{}: {}/{} frames have detection result, prefect'.format(name, frame_roi_exist_num,
                #                                                             frame_end - frame_start))
            else:
                # print('{}: {}/{} frames have detection result, insert'.format(name, frame_roi_exist_num,
                #                                                             frame_end - frame_start))
                i = 1
                forward_buffer = []
                forward_buffer_idx = -1
                while frame_start - i >= 0:
                    if get_from_frame_detection(frame_start - i, speaker_id):
                        forward_buffer = get_from_frame_detection(frame_start - i, speaker_id)
                        forward_buffer_idx = frame_start - i
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
                                # print(need_insert_idx, segment_roi_bound[need_insert_idx], segment_roi_idx[need_insert_idx], frame_roi_bound, frame_idx)
                            else:
                                segment_roi_bound[need_insert_idx] = (
                                        np.array(forward_buffer) +
                                        (segment_roi_idx[need_insert_idx] - forward_buffer_idx) *
                                        (np.array(frame_roi_bound) - np.array(forward_buffer)) /
                                        (frame_idx - forward_buffer_idx)).astype(np.int64).tolist()
                                # print(need_insert_idx, segment_roi_bound[need_insert_idx], segment_roi_idx[need_insert_idx], frame_roi_bound, frame_idx, forward_buffer, forward_buffer_idx)
                        forward_buffer = frame_roi_bound
                        forward_buffer_idx = frame_idx
                    else:
                        need_insert_idxes.append(i)

                if need_insert_idxes:
                    i = 0
                    backward_buffer = []
                    backward_buffer_idx = -1
                    while frame_end + i < total_frames_num:
                        if get_from_frame_detection(frame_end + i, speaker_id):
                            backward_buffer = get_from_frame_detection(frame_end + i, speaker_id)
                            backward_buffer_idx = frame_end + i
                            break
                        else:
                            i += 1
                    while need_insert_idxes:
                        need_insert_idx = need_insert_idxes.pop(0)
                        if forward_buffer_idx == -1 and backward_buffer_idx == -1:
                            raise ValueError('no context cannot pad')
                        elif forward_buffer_idx == -1:
                            segment_roi_bound[need_insert_idx] = backward_buffer
                            # print(need_insert_idx, segment_roi_bound[need_insert_idx], segment_roi_idx[need_insert_idx], backward_buffer, backward_buffer_idx)
                        elif backward_buffer_idx == -1:
                            segment_roi_bound[need_insert_idx] = forward_buffer
                            # print(need_insert_idx, segment_roi_bound[need_insert_idx], segment_roi_idx[need_insert_idx], forward_buffer, forward_buffer_idx)
                        else:
                            segment_roi_bound[need_insert_idx] = (
                                    np.array(forward_buffer) +
                                    (segment_roi_idx[need_insert_idx] - forward_buffer_idx) *
                                    (np.array(backward_buffer) - np.array(forward_buffer)) /
                                    (backward_buffer_idx - forward_buffer_idx)).astype(np.int64).tolist()
                            # print(need_insert_idx, segment_roi_bound[need_insert_idx], segment_roi_idx[need_insert_idx], backward_buffer, backward_buffer_idx, forward_buffer, forward_buffer_idx)
                assert not need_insert_idxes
                segments_roi_bound[name] = segment_roi_bound
    return segments_roi_bound
                  