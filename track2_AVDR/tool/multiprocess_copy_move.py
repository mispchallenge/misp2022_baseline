#!/usr/bin/env python
# _*_ coding: UTF-8 _*_
import os
import re
import shutil
import codecs
import json
from tqdm import tqdm
from multiprocessing import Pool


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


def copy_move_file_worker(file_list, keep_source=True, processing_id=None, processing_num=None, **other_params):
    file_num = len(file_list)
    for file_idx in tqdm(range(file_num), leave=True, desc='0' if processing_id is None else str(processing_id)):
        if processing_id is None:
            processing_token = True
        else:
            if file_idx % processing_num == processing_id:
                processing_token = True
            else:
                processing_token = False
        if processing_token:
            source_file, destination_file = file_list[file_idx]
            store_dir = os.path.split(destination_file)[0]
            if not os.path.exists(store_dir):
                os.makedirs(store_dir, exist_ok=True)
            if keep_source:
                shutil.copy(source_file, destination_file)
            else:
                shutil.move(source_file, destination_file)
    return None


def copy_move_file_manager(file_list, keep_source=True, processing_num=1):
    if processing_num > 1:
        pool = Pool(processes=processing_num)
        for i in range(processing_num):
            pool.apply_async(copy_move_file_worker, kwds={
                'file_list': file_list, 'keep_source': keep_source,
                'processing_id': i, 'processing_num': processing_num})
        pool.close()
        pool.join()
    else:
        copy_move_file_worker(file_list=file_list, keep_source=keep_source)
    return None


def generate_file_list(source_dir, destination_root, pattern, ignore=True):
    source_list, destination_list = [], []
    destination_dir = os.path.join(destination_root, os.path.split(source_dir)[-1])
    for root, ds, fs in os.walk(source_dir):
        for f in fs:
            fullname = os.path.join(root, f)
            # match_sign = re.search(pattern, fullname) is not None
            # if (not ignore and match_sign) or (ignore and not match_sign):
            source_list.append([fullname, fullname.replace(source_dir, destination_dir)])
            # destination_list.append(fullname.replace(source_dir, destination_dir))
    return source_list


if __name__ == '__main__':
    # source_list = generate_file_list(
    #     source_dir='/yrfs2/cv1/hangchen2/hszhou2/Project_KWS_2021_MISP_Challenge/final_data_configuration_version5/MISP2021_AVWWS/xiangsi/MISP2021_AVWWS/negative', destination_root='/yrfs2/cv1/hangchen2/data/negative',
    #     pattern=re.compile(r'_Far_\d.wav'), ignore=False)

    # copy_move_file_manager(file_list=source_list, keep_source=True,
    #                        processing_num=15)
    data_dic = json2dic('/yrfs2/cv1/hangchen2/feature/misp2021/selected_4h.json')
    key2path_dic = data_dic['key2path']
    new_key2path_dic = {}
    file_list = []
    target_dir = '/yrfs2/cv1/hangchen2/feature/misp2021_selected_4h'
    for key, value in key2path_dic.items():
        file_list.append([value['mixture_wave'], os.path.join(target_dir, 'mixture_wave', value['mixture_wave'].split('/')[-1])])
        file_list.append([value['lip_frames'], os.path.join(target_dir, 'lip_frames', value['lip_frames'].split('/')[-1])])
        new_key2path_dic[key] = {
            'mixture_wave': os.path.join(target_dir, 'mixture_wave', value['mixture_wave'].split('/')[-1]),
            'lip_frames': os.path.join(target_dir, 'lip_frames', value['lip_frames'].split('/')[-1])
        }
    copy_move_file_manager(file_list=file_list, keep_source=True, processing_num=15)
    json2dic('/yrfs2/cv1/hangchen2/feature/misp2021_selected_4h/index.json', {'keys': data_dic['keys'], 
                                                                                'duration': data_dic['duration'], 
                                                                                'key2path': new_key2path_dic})
        
    
