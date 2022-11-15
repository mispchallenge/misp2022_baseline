#!/usr/bin/env python
# _*_ coding: UTF-8 _*_
import os
import json
import codecs
import argparse

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

def index_file_misp2022():
    for dataset in ['dev','train']:
        item2dir = {
            'far_wave': 'feature/misp2022_avsr/{}_far_audio_segment'.format(dataset),
            'far_pdf': 'feature/misp2022_avsr/{}_far_tri3_ali'.format(dataset),
            'far_lip': 'feature/misp2022_avsr/{}_far_video_lip_segment'.format(dataset),
        }
        index_dict = {'keys': [], 'duration': [], 'key2path': {}}
        # sum key
        item_list = sorted([*item2dir.keys()])
        summed_key = set(json2dic(os.path.join(item2dir[item_list[0]], 'key2shape.json')).keys())
        for item in item_list[1:]:
            summed_key = summed_key & set(json2dic(os.path.join(item2dir[item], 'key2shape.json')).keys())
        for key in [*summed_key]:
            start, end = key.split('_')[-1].split('-')
            start, end = int(start), int(end)
            duration = round((end - start) / 100., 2)
            index_dict['keys'].append(key)
            index_dict['duration'].append(duration)
            index_dict['key2path'][key] = {k: os.path.join(v, 'pt', '{}.pt'.format(key)) for k,v in item2dir.items()}
        json2dic(jsonpath='feature/misp2022_avsr/{}_far.json'.format(dataset), dic=index_dict)
    return None

def index_file_misp2022_inference():
    for dataset in ['dev']:
        item2dir = {
            'far_wave': 'feature/misp2022_avsr/{}_far_audio_inference_segment'.format(dataset),
            'far_lip': 'feature/misp2022_avsr/{}_far_video_inference_lip_segment'.format(dataset),
        }
        index_dict = {'keys': [], 'duration': [], 'key2path': {}}
        # sum key
        item_list = sorted([*item2dir.keys()])
        summed_key = set(json2dic(os.path.join(item2dir[item_list[0]], 'key2shape.json')).keys())
        for item in item_list[1:]:
            summed_key = summed_key & set(json2dic(os.path.join(item2dir[item], 'key2shape.json')).keys())
        for key in [*summed_key]:
            start, end = key.split('_')[-1].split('-')
            start, end = int(start), int(end)
            duration = round((end - start) / 100., 2)
            index_dict['keys'].append(key)
            index_dict['duration'].append(duration)
            index_dict['key2path'][key] = {k: os.path.join(v, 'pt', '{}.pt'.format(key)) for k,v in item2dir.items()}
        json2dic(jsonpath='feature/misp2022_avsr/{}_far_inference.json'.format(dataset), dic=index_dict)
    return None

if __name__ == '__main__':
    parser = argparse.ArgumentParser('index_file2json')
    parser.add_argument('-s', '--select_mode', type=int, default=0, help='select the mode')
    input_args = parser.parse_args()
    
    if input_args.select_mode == 0:
        index_file_misp2022()
    elif input_args.select_mode == 1:
        index_file_misp2022_inference()
    else:
        raise ValueError('regular_expression {} is error, find no possible mode'.format(input_args.select_mode))