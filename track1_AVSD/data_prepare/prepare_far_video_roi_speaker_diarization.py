#!/usr/bin/env python
# _*_ coding: UTF-8 _*_
from itertools import accumulate
import os
import cv2
import json
import codecs
import numpy as np
import multiprocessing
import HTK
import tqdm
import time
import argparse


parser = argparse.ArgumentParser() 
parser.add_argument('--set')
parser.add_argument('--video_dir')
parser.add_argument('--roi_json_dir')
parser.add_argument('--roi_store_dir')
args = parser.parse_args()

def crop_frame_roi(frame, roi_bound, roi_size=(96, 96)):
    bound_l = max(roi_bound[3] - roi_bound[1], roi_bound[2] - roi_bound[0], *roi_size)
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
    roi_frame = frame[x_start: x_end, y_start: y_end, :]
    if x_end - x_start != roi_size[1] or y_end - y_start != roi_size[0] :
        resized_roi_frame = cv2.resize(roi_frame, roi_size, interpolation=cv2.INTER_AREA)
        return cv2.cvtColor(resized_roi_frame, cv2.COLOR_BGR2GRAY)
    else:
        return cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)


def segment_video_roi_json(video_path, roi_json_path, roi_store_dir):

    name = video_path.split('/')[-1].split('.')[0]
    print(f"processing {name}")
    video_capture = cv2.VideoCapture(video_path)

    if not os.path.exists(roi_store_dir+"/lip"):
        os.makedirs(roi_store_dir+"/lip", exist_ok=True)
    if not os.path.exists(roi_store_dir+"/face"):
        os.makedirs(roi_store_dir+"/face", exist_ok=True)
    
    s = name.split("_")[1][1:]
    roi_spk = {}
    while s != "":
        roi_spk[s[:3]] = {}
        s = s[3:]
    with codecs.open(roi_json_path, 'r') as handle:
        roi_dic = json.load(handle)
        for f in roi_dic.keys():
            for spk in roi_spk.keys():
                roi_spk[spk][f] = {}
                roi_spk[spk][f]["face"] = []
                roi_spk[spk][f]["lip"] = []
            for item in roi_dic[f]:
                spk = "{:03d}".format(item['id'])
                if spk in roi_spk.keys():
                    roi_spk[spk][f]["face"] = [item['x1'], item['y1'], item['x2'], item['y2']]
                    roi_spk[spk][f]["lip"] = item["lip"]

    start_frame = {}
    frame_idx = 0
    segments_roi_frames_buffer = {}
    for spk in roi_spk.keys():
        segments_roi_frames_buffer[spk] = {}
        start_frame[spk] = {}
        segments_roi_frames_buffer[spk]["face"] = []
        segments_roi_frames_buffer[spk]["lip"] = []
        start_frame[spk]["face"] = 0
        start_frame[spk]["lip"] = 0
    while video_capture.isOpened():
        ret, frame = video_capture.read()
        if ret:
            for spk in roi_spk.keys():
                if str(frame_idx) not in roi_spk[spk].keys():
                    continue
                for roi_type in ["face", "lip"]:
                    segment_roi_bound = roi_spk[spk][str(frame_idx)][roi_type]
                    if segment_roi_bound != []:
                        segments_roi_frames_buffer[spk][roi_type].append(crop_frame_roi(frame, segment_roi_bound, (96, 96)))
                    else:
                        if segments_roi_frames_buffer[spk][roi_type] != []:
                            output_path = os.path.join(roi_store_dir, roi_type, "{}_{}-{}-{}.htk".format(name, spk, start_frame[spk][roi_type], frame_idx-1))
                            segments_roi = np.stack(segments_roi_frames_buffer[spk][roi_type])
                            HTK.writeHtk3D(output_path, segments_roi)
                            segments_roi = 0
                        start_frame[spk][roi_type] = frame_idx + 1
                        segments_roi_frames_buffer[spk][roi_type] = []
            frame_idx += 1
        else:
            break
    for spk in segments_roi_frames_buffer.keys():
        for roi_type in segments_roi_frames_buffer[spk].keys():
            if segments_roi_frames_buffer[spk][roi_type] != []:
                output_path = os.path.join(roi_store_dir, roi_type, "{}_{}-{}-{}.htk".format(name, spk, start_frame[spk][roi_type], frame_idx-1))
                segments_roi = np.stack(segments_roi_frames_buffer[spk][roi_type])
                HTK.writeHtk3D(output_path, segments_roi)
                segments_roi = 0
    video_capture.release()

    return None

def process(video_item, video_dir, roi_json_dir, roi_store_dir):
    for i in video_item:
        point = time.time()
        video_path = os.path.join(video_dir, i+".mp4")
        roi_json_path = os.path.join(roi_json_dir, i+".json")
        if os.path.isfile(video_path) and os.path.isfile(roi_json_path):
            segment_video_roi_json(video_path, roi_json_path, roi_store_dir)
        else:
            print(roi_json_path)
        print(f"Process {i}, take {time.time() - point}s ")

if __name__ == '__main__':
    for set in [args.set]:
        video_dir = args.video_dir
        roi_json_dir = args.roi_json_dir
        roi_store_dir = args.roi_store_dir
        video_item = [ i.split('.')[0] for i in os.listdir(video_dir) if i.endswith(".mp4") ]
        print(len(video_item))
        print(video_item)
        for v in video_item:
            os.system(f"rm -f {roi_store_dir}/face/{v}*")
            os.system(f"rm -f {roi_store_dir}/lip/{v}*")
        num_thread = 1
        video_items = [ l.tolist() for l in np.array_split(video_item, num_thread) ]

        processes = []
        for id in range(num_thread):
            p = multiprocessing.Process(target=process, 
                args=(video_items[id], video_dir, roi_json_dir, roi_store_dir,))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
