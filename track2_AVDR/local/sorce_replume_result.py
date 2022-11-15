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


def text2lines(textpath, lines_content=None):
    """
    read lines from text or write lines to txt
    :param textpath: filepath of text
    :param lines_content: list of lines or None, None means read
    :return: processed lines content for read while None for write
    """
    if lines_content is None:
        with codecs.open(textpath, 'r') as handle:
            lines_content = handle.readlines()
        processed_lines = [*map(lambda x: x[:-1] if x[-1] in ['\n'] else x, lines_content)]
        return processed_lines
    else:
        processed_lines = [*map(lambda x: x if x[-1] in ['\n'] else '{}\n'.format(x), lines_content)]
        with codecs.open(textpath, 'w') as handle:
            handle.write(''.join(processed_lines))
        return None


def merge_result(utt_result, data_json, result_json):
    result_lines = text2lines(utt_result)
    data_dic = json2dic(data_json)
    result_keys = []
    for line in result_lines:
        # import pdb; pdb.set_trace()
        content_list = [i for i in line.split(' ') if i != '']
        utt_key, item, *content = content_list
        result_keys.append(utt_key)
        if item == '#csid':
            item = 'csid'
            content = [int(i) for i in content]
        data_dic['key2path'][utt_key][item] = content
    print(set(data_dic['keys']) - set(result_keys))
    json2dic(result_json, dic=data_dic)
    return None


def statistics_per(result_json):
    result_dic = json2dic(result_json)
    statistic_table = result_json.replace('.json', '.table')
    keys_list = result_dic['keys']
    duration_list = result_dic['duration']
    key2item = result_dic['key2path']
    # classification
    metric_type = 'csid'
    class_types = ['nonoverlap', 'nonoverlap+tv', 'overlap', 'overlap+tv', 'all']
    id2class = {'C07': 'nonoverlap', 'C08': 'nonoverlap', 'C12': 'nonoverlap', 'C01': 'nonoverlap+tv', 'C02': 'nonoverlap+tv', 
                'C09': 'nonoverlap+tv', 'C04': 'overlap', 'C03': 'overlap', 'C10': 'overlap', 'C05': 'overlap+tv', 'C06': 'overlap+tv', 
                'C11': 'overlap+tv'}
    class2duration = {'nonoverlap': 0., 'nonoverlap+tv': 0., 'overlap': 0., 'overlap+tv': 0., 'all': 0.}
    # statistic
    statistic_dic = {'all': {'c': [], 's': [], 'i': [], 'd': []}}
    for key_idx in tqdm(range(len(keys_list)), leave=False):
        key = keys_list[key_idx]
        duration = duration_list[key_idx]
        _, _, _, config_id, _, _ = key.split('_')
        for i, v in enumerate(['c', 's', 'i', 'd']):
            # import pdb; pdb.set_trace()
            statistic_dic['all'][v].append(int(key2item[key][metric_type][i]))
        utt_class = id2class[config_id]
        class2duration[utt_class] += duration
        class2duration['all'] += duration
        if utt_class in statistic_dic:
            for i, v in enumerate(['c', 's', 'i', 'd']):
                statistic_dic[utt_class][v].append(int(key2item[key][metric_type][i]))
        else:
            statistic_dic[utt_class] =  {}
            for i, v in enumerate(['c', 's', 'i', 'd']):
                statistic_dic[utt_class][v] = [int(key2item[key][metric_type][i])]
    
    # print
    table = PrettyTable(['class', 'duration', 'c', 's', 'i', 'd', 't', 'e'])
    for class_type in class_types:
        class_c = np.sum(statistic_dic[class_type]['c'])
        class_s = np.sum(statistic_dic[class_type]['s'])
        class_i = np.sum(statistic_dic[class_type]['i'])
        class_d = np.sum(statistic_dic[class_type]['d'])
        class_t = class_c+class_s+class_d
        class_e = class_i+class_s+class_d
        table.add_row([class_type, class2duration[class_type], class_c, class_s, class_i, class_d, class_t, class_e])
        table.add_row([class_type, class2duration[class_type], 100.*class_c/class_t, 100.*class_s/class_t, 100.*class_i/class_t, 100.*class_d/class_t, 100, 100.*class_e/class_t])
    with codecs.open(statistic_table, 'w') as f:
        print(table, file=f)
    print(table)
    return None


def plot_overlap2cer(result_json):
    result_dic = json2dic(result_json)
    keys_list = result_dic['keys']
    key2path = result_dic['key2path']
    # classification
    id2class = {'C07': 'nonoverlap', 'C08': 'nonoverlap', 'C12': 'nonoverlap', 'C01': 'nonoverlap+tv', 'C02': 'nonoverlap+tv', 
                'C09': 'nonoverlap+tv', 'C04': 'overlap', 'C03': 'overlap', 'C10': 'overlap', 'C05': 'overlap+tv', 'C06': 'overlap+tv', 
                'C11': 'overlap+tv'}
    # key2overlap
    duration_factor = 4
    segment2utt = {}
    utt2vad_array = {}
    class2color = {'nonoverlap': 0, 'nonoverlap+tv': 25, 'overlap': 50, 'overlap+tv': 75}
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
    overlap_list = []
    cer_list = []
    tv_on_list = []
    # print([*utt2vad_array.keys()])
    # for key, value in utt2vad_array.items():
    #     print(key, value.shape, np.sum(value==0), np.sum(value==1), np.sum(value==2), np.sum(value==3), np.sum(value==4), np.sum(value==5), np.sum(value==6))
    overlap_step = 0.1
    overlap_class = np.arange(0, 1, overlap_step)+overlap_step
    overlap_class2c = np.zeros(overlap_class.shape)
    overlap_class2s = np.zeros(overlap_class.shape)
    overlap_class2i = np.zeros(overlap_class.shape)
    overlap_class2d = np.zeros(overlap_class.shape)
    for key in keys_list:
        _, _, _, config_id, _, _ = key.split('_')
        class_id = id2class[config_id]
        if class_id in ['nonoverlap', 'overlap']:
            vad_array = utt2vad_array[segment2utt[key][0]][segment2utt[key][1]: segment2utt[key][2]]
            overlap = float(np.sum(vad_array > 1)) / vad_array.shape[0]
            idx = 0
            while overlap > overlap_class[idx]:
                idx+=1
            c, s, i, d = key2path[key]['csid']
            c, s, i, d = int(c), int(s), int(i), int(d)
            overlap_class2c[idx] += c
            overlap_class2s[idx] += s
            overlap_class2i[idx] += i
            overlap_class2d[idx] += d
            # overlap_list.append(overlap)
            # cer_list.append(100.*(i+s+d)/(c+s+d))
            # tv_on_list.append(class2color[class_id])
    overlap_class2t = overlap_class2c + overlap_class2s + overlap_class2d
    overlap_class2e = overlap_class2i + overlap_class2s + overlap_class2d
    overlap_class2ser = overlap_class2s / overlap_class2t * 100
    overlap_class2ier = overlap_class2i / overlap_class2t * 100
    overlap_class2der = overlap_class2d / overlap_class2t * 100
    overlap_class2cer = overlap_class2e / overlap_class2t * 100
    np.savez(os.path.join(os.path.split(result_json)[0], 'overlap2cer_{}.npz'.format(overlap_step)), overlap=overlap_class, cer=overlap_class2cer, ser=overlap_class2ser, 
             ier=overlap_class2ier, der=overlap_class2der)

    print_str = 'overlap: {}\nser: {}\nder: {}\nier: {}\ncer: {}'.format(' '.join([*map(str, overlap_class)]), ' '.join([*map(str, overlap_class2ser)]), ' '.join([*map(str, overlap_class2der)]), 
                                                                         ' '.join([*map(str, overlap_class2ier)]), ' '.join([*map(str, overlap_class2cer)]))
    with codecs.open(os.path.join(os.path.split(result_json)[0], 'overlap2cer_{}.txt'.format(overlap_step)), 'w') as f:
        print(print_str, file=f)
    print(print_str)
    figure = plt.figure()
    plt.plot(overlap_class, overlap_class2cer, label='cer')
    plt.plot(overlap_class, overlap_class2ser, label='s')
    plt.plot(overlap_class, overlap_class2ier, label='i')
    plt.plot(overlap_class, overlap_class2der, label='d')
    plt.xlabel('overlap')
    plt.ylabel('error rate')
    plt.legend(loc='upper right')
    plt.close('all')
    # plt.scatter(np.array(overlap_list), np.array(cer_list), c=np.array(tv_on_list), cmap='viridis')
    # plt.close('all')
    figure.savefig(os.path.join(os.path.split(result_json)[0], 'overlap2cer_{}.png'.format(overlap_step)), dpi=330, bbox_inches='tight')
    return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser('sorce_replume_result')
    parser.add_argument('utt_result', type=str, default='/yrfs1/intern/hangchen2/experiment/EASE',
                        help='result txt')
    parser.add_argument('data_json', type=str, default='/yrfs1/intern/hangchen2/experiment/EASE',
                        help='result txt')
    parser.add_argument('result_json', type=str, default='/yrfs1/intern/hangchen2/experiment/EASE',
                        help='result txt')
    parser.add_argument('--stage', type=int, default=0,
                        help='result txt')
    args = parser.parse_args()
    
    if args.stage <= 0:
        merge_result(utt_result=args.utt_result, data_json=args.data_json, result_json=args.result_json)
    
    if args.stage <= 1:
        statistics_per(result_json=args.result_json)
    
    if args.stage <= 2:
        plot_overlap2cer(result_json=args.result_json)
