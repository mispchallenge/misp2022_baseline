import os
import json
import codecs
import argparse

def get_utt2spk_segments_from_rttm_file(path, path_utt2spk, path_segments, **other_params):
    content = []
    content_utt2spk = []
    content_segments = []
    with open(path, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            _, name, _, start_time, duration, _, _, speaker_ID, _, _ = line.split(" ")
            if  float(duration) < 0.04:
                continue
            information = [name, start_time, duration, speaker_ID]
            assert len(information) == 4, "Length Error! {}".format(information)
            content.append(information)
    for element in content:
        start_time = element[1]
        end_time = str(round((float(element[1])+float(element[2])), 2))
        start_time_str = "00000{}".format(str(int(float(element[1])*100)))[-6:]
        end_time_str = "00000{}".format(str(int((float(element[1])+float(element[2]))*100)))[-6:]
        speaker_ID = "S" + "00000{}".format(str(int(element[3])))[-3:]
        time_str = start_time_str + "-" + end_time_str
        file_name = "_".join([speaker_ID, element[0], time_str])
        information_utt2spk = " ".join([file_name, speaker_ID])
        information_segments = " ".join([file_name, element[0] + "_Far", start_time, end_time])
        content_utt2spk.append(information_utt2spk)
        content_segments.append(information_segments)
    
    with open(path_utt2spk, 'w') as f_utt2spk:
        f_utt2spk.write("\n".join(sorted(content_utt2spk))+"\n")
    with open(path_segments, 'w') as f_segments:
        f_segments.write("\n".join(sorted(content_segments))+"\n")
    
    return None

if __name__ == '__main__':
    parser = argparse.ArgumentParser('prepare_far_video_roi')
    parser.add_argument('rttm_path', type=str, default='data/dev_far.rttm', help='the path of the RTTM file')
    parser.add_argument('utt2spk_path', type=str, default='data/dev_far_audio_inference/utt2spk', help='storage path of the utt2spk file')
    parser.add_argument('segments_path', type=str, default='data/dev_far_audio_inference/segments', help='storage path of the segments file')
    args = parser.parse_args()

    get_utt2spk_segments_from_rttm_file(args.rttm_path, args.utt2spk_path, args.segments_path)