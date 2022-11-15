#!/usr/bin/env python
# _*_ coding: UTF-8 _*_
import os
import cv2
import codecs
import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt
plt.switch_backend('agg')

def show_video(video_path, store_dir, xstart=0., xend=None):
    video_capture = cv2.VideoCapture(video_path)
    fps = int(video_capture.get(5))
    total_frames = int(video_capture.get(7))
    if xend is None:
        frames_end = total_frames
    else:
        frames_end = int(np.around(xend * fps))
    frames_start = int(np.around(xstart * fps))
    frames_end = min(total_frames, frames_end)
    
    os.makedirs(store_dir, exist_ok=True)
    frames_idx = 0
    while video_capture.isOpened():
        ret, frame = video_capture.read()
        if ret:
            if frames_end > frames_idx >= frames_start:
                cv2.imwrite(os.path.join(store_dir, '{}.jpg'.format(frames_idx)), frame)
            frames_idx += 1
        else:
            break
    video_capture.release()
    return None


def show_audio(audio_path, store_path, xstart=0., xend=None):
    sample_rate, sample_data = wavfile.read(audio_path)
    if xend is None:
        frames_end = sample_data.shape[0]
    else:
        frames_end = int(np.around(xend * sample_rate))
    frames_start = int(np.around(xstart * sample_rate))
    frames_end = min(sample_data.shape[0], frames_end)
    
    store_data = sample_data[frames_start: frames_end]

    os.makedirs(os.path.split(store_path)[0], exist_ok=True)
    wavfile.write(store_path, sample_rate, store_data)
    figure = plt.figure()
    plt.plot(np.arange(0, store_data.shape[0]) * (1.0 / sample_rate), store_data)
    plt.xlabel('time/s')
    plt.ylabel('amplitude')
    figure.savefig(store_path.replace('.wav', '.jpg'), dpi=600, bbox_inches='tight')
    return None


if __name__ == '__main__':
    segment_name = 'S245
    _R16_S242243244245_C02_I1_057384-057540'
    dataset = 'eval'
    speaker, *utt_name, duration = segment_name.split('_')
    utt_name = '_'.join(utt_name)
    start, end = duration.split('-')
    start = int(start) / 100.
    end = int(end) / 100.
    
    store_dir = 'sample_{}'.format(segment_name)
    
    # far audio
    far_audio_root = 'released_data/misp2021_avsr/{}_far_audio/wav'.format(dataset)
    for i in range(6):
        show_audio(audio_path=os.path.join(far_audio_root, '{}_Far_{}.wav'.format(utt_name, i)), store_path=os.path.join(store_dir, '{}_Far_{}.wav'.format(utt_name, i)), xstart=start, xend=end)
    # far wpe audio
    far_wpe_audio_root = 'feature/misp2021_avsr/{}_far_audio_wpe/wav'.format(dataset)
    for i in range(6):
        show_audio(audio_path=os.path.join(far_wpe_audio_root, '{}_Far_{}.wav'.format(utt_name, i)), store_path=os.path.join(store_dir, '{}_Far_{}_wpe.wav'.format(utt_name, i)), xstart=start, xend=end)
    # far beamformit audio
    far_beamformit_audio_root = 'feature/misp2021_avsr/{}_far_audio_wpe_beamformit/wav'.format(dataset)
    show_audio(audio_path=os.path.join(far_beamformit_audio_root, '{}_Far.wav'.format(utt_name)), store_path=os.path.join(store_dir, '{}_Far_wpe_beamformit.wav'.format(utt_name)), xstart=start, xend=end)
    # middle audio
    middle_audio_root = 'released_data/misp2021_avsr/{}_middle_audio/wav'.format(dataset)
    for i in range(2):
        show_audio(audio_path=os.path.join(middle_audio_root, '{}_Middle_{}.wav'.format(utt_name, i)), store_path=os.path.join(store_dir, '{}_Middle_{}.wav'.format(utt_name, i)), xstart=start, xend=end)
    # middle wpe audio
    middle_wpe_audio_root = 'feature/misp2021_avsr/{}_middle_audio_wpe/wav'.format(dataset)
    for i in range(2):
        show_audio(audio_path=os.path.join(middle_wpe_audio_root, '{}_Middle_{}.wav'.format(utt_name, i)), store_path=os.path.join(store_dir, '{}_Middle_{}_wpe.wav'.format(utt_name, i)), xstart=start, xend=end)
    # middle beamformit audio
    middle_beamformit_audio_root = 'feature/misp2021_avsr/{}_middle_audio_wpe/beamformit/wav'.format(dataset)
    show_audio(audio_path=os.path.join(middle_beamformit_audio_root, '{}_Middle.wav'.format(utt_name)), store_path=os.path.join(store_dir, '{}_Middle_wpe_beamformit.wav'.format(utt_name)), xstart=start, xend=end)
    # near audio
    near_audio_root = 'released_data/misp2021_avsr/{}_near_audio/wav'.format(dataset)
    show_audio(audio_path=os.path.join(near_audio_root, '{}_Near_{}.wav'.format(utt_name, speaker[1:])), store_path=os.path.join(store_dir, '{}_Near_{}.wav'.format(utt_name, speaker[1:])), xstart=start, xend=end)
    
    # far video
    far_video_root = 'released_data/misp2021_avsr/{}_far_video/mp4'.format(dataset)
    show_video(video_path=os.path.join(far_video_root, '{}_Far.mp4'.format(utt_name)), store_dir=os.path.join(store_dir, 'far_video'), xstart=60., xend=62)
    
    # middle video
    middle_video_root = 'released_data/misp2021_avsr/{}_middle_video/mp4'.format(dataset)
    show_video(video_path=os.path.join(middle_video_root, '{}_Middle_{}.mp4'.format(utt_name, speaker[1:])), store_dir=os.path.join(store_dir, 'middle_video'), xstart=60., xend=62.)
