#!/usr/bin/env python
# _*_ coding: UTF-8 _*_
import os
import codecs
import json
import argparse
import numpy as np
from tqdm import tqdm
from prettytable import PrettyTable
import matplotlib.pyplot as plt
plt.switch_backend('agg')


def json2dic(jsonpath, dic=None):
    """
    read dic from json or write dic to json
    :param jsonpath: filepath of json
    :param dic: content dic or None, None means read
    :return: content dic for read while None for write
    """
    if dic is None:
        with codecs.open(jsonpath, 'r') as handle:
            output = json.load(handle)
        return output
    else:
        assert isinstance(dic, dict)
        with codecs.open(jsonpath, 'w') as handle:
            json.dump(dic, handle)
        return None


def overlsp2cer(result_json):
    id2class = {'C07': 'nonoverlap', 'C08': 'nonoverlap', 'C12': 'nonoverlap', 'C01': 'nonoverlap+tv', 'C02': 'nonoverlap+tv', 
                'C09': 'nonoverlap+tv', 'C04': 'overlap', 'C03': 'overlap', 'C10': 'overlap', 'C05': 'overlap+tv', 'C06': 'overlap+tv', 
                'C11': 'overlap+tv'}
    result_dic = json2dic(result_json)
    keys_list = result_dic['keys']
    key2path = result_dic['key2path']
    # key2overlap
    duration_factor = 4
    segment2utt = {}
    utt2vad_array = {}
    for key in keys_list:
        speaker_id, *utt_id, stamp = key.split('_')
        utt_id = '_'.join(utt_id)
        # print(utt_id)
        start, end = stamp.split('-')
        start = int(round(int(start) / duration_factor))
        end = int(round(int(end) / duration_factor))
        segment2utt[key] = [utt_id, start, end]
        if utt_id in utt2vad_array:
            vad_array = utt2vad_array[utt_id]
            if end > vad_array.shape[0]:
                current_vad_array = np.zeros((end, ))
                current_vad_array[start: end] = 1
                current_vad_array[:vad_array.shape[0]] = current_vad_array[:vad_array.shape[0]]+ vad_array
                utt2vad_array[utt_id] = current_vad_array
            else:
                vad_array[start: end] = vad_array[start: end] + 1
                utt2vad_array[utt_id] = vad_array
        else:
            current_vad_array = np.zeros((end, ))
            current_vad_array[start: end] = 1
            utt2vad_array[utt_id] = current_vad_array
    max_speaker_num_list = []
    overlap_class2c = {}
    overlap_class2s = {}
    overlap_class2i = {}
    overlap_class2d = {}    
    for key in keys_list:
        _, _, _, config_id, _, _ = key.split('_')
        class_id = id2class[config_id]
        if class_id in ['nonoverlap', 'nonoverlap+tv', 'overlap', 'overlap+tv']:
            vad_array = utt2vad_array[segment2utt[key][0]][segment2utt[key][1]: segment2utt[key][2]]
            max_speaker_num = vad_array.max()
            c, s, i, d = key2path[key]['csid']
            c, s, i, d = int(c), int(s), int(i), int(d)
            
            if max_speaker_num in max_speaker_num_list:
                overlap_class2c[max_speaker_num] += c
                overlap_class2s[max_speaker_num] += s
                overlap_class2i[max_speaker_num] += i
                overlap_class2d[max_speaker_num] += d
            else:
                max_speaker_num_list.append(max_speaker_num)
                overlap_class2c[max_speaker_num] = c
                overlap_class2s[max_speaker_num] = s
                overlap_class2i[max_speaker_num] = i
                overlap_class2d[max_speaker_num] = d
    max_speaker_num_list = sorted(max_speaker_num_list)
    for i in max_speaker_num_list:
        t = overlap_class2c[i] + overlap_class2s[i] + overlap_class2d[i]
        print(i, 'all:', t, 'ser:', overlap_class2s[i] / t * 100, 'der:', overlap_class2d[i] / t * 100, 'ier:', overlap_class2i[i] / t * 100)
    return None


def noise2cer(result_json):
    id2class = {'C07': 'nonoverlap', 'C08': 'nonoverlap', 'C12': 'nonoverlap', 'C01': 'nonoverlap+tv', 'C02': 'nonoverlap+tv', 
                'C09': 'nonoverlap+tv', 'C04': 'overlap', 'C03': 'overlap', 'C10': 'overlap', 'C05': 'overlap+tv', 'C06': 'overlap+tv', 
                'C11': 'overlap+tv'}
    class2tv = {'nonoverlap': 'tv_off', 'nonoverlap+tv': 'tv_on', 'overlap': 'tv_off', 'overlap+tv': 'tv_on'}
    result_dic = json2dic(result_json)
    keys_list = result_dic['keys']
    key2path = result_dic['key2path']
    overlap_class2c = {'tv_off': 0., 'tv_on': 0.}
    overlap_class2s = {'tv_off': 0., 'tv_on': 0.}
    overlap_class2i = {'tv_off': 0., 'tv_on': 0.}
    overlap_class2d = {'tv_off': 0., 'tv_on': 0.}    
    for key in keys_list:
        _, _, _, config_id, _, _ = key.split('_')
        class_id = id2class[config_id]
        if class_id in ['nonoverlap', 'nonoverlap+tv', 'overlap', 'overlap+tv']:
            c, s, i, d = key2path[key]['csid']
            c, s, i, d = int(c), int(s), int(i), int(d)
            overlap_class2c[class2tv[class_id]] += c
            overlap_class2s[class2tv[class_id]] += s
            overlap_class2i[class2tv[class_id]] += i
            overlap_class2d[class2tv[class_id]] += d

    for i in ['tv_off', 'tv_on']:
        t = overlap_class2c[i] + overlap_class2s[i] + overlap_class2d[i]
        print(i, 'all:', t, 'ser:', overlap_class2s[i] / t *100, 'der:', overlap_class2d[i] / t *100, 'ier:', overlap_class2i[i] / t *100)
    return None


if __name__ == '__main__':
    print('far wave')
    overlsp2cer(result_json='exp/0_1_MISP2021_far_asr/predict_best_eval_addition/result_after_exp_tri3_far_audio_decode_log_likelihoods/result_cer.json')
    # print('far wave + far lip')
    # noise2cer(result_json='exp/1_3_MISP2021_far_wave_far_lip_avsr/predict_best_eval_addition/result_after_exp_tri3_far_audio_decode_log_likelihoods/result_cer.json')
    print('far wave + middle lip')
    overlsp2cer(result_json='exp/1_5_MISP2021_far_wave_middle_lip_avsr/predict_best_eval_addition/result_after_exp_tri3_far_audio_decode_log_likelihoods/result_cer.json')
    
    