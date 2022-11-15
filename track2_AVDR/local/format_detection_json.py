#!/usr/bin/env python
# _*_ coding: UTF-8 _*_
import os
import json
import glob
import tqdm
import shutil
import codecs
import argparse


split_c0001_dic = {'R01_S010011012_C0001_I0': 218, 'R01_S021022023_C0001_I0': 204, 'R01_S024025026027_C0001_I1': 220, 
                   'R01_S037038039040_C0001_I2': 0, 'R02_S060061_C0001_I0': 112, 'R02_S062063_C0001_I0': 106, 
                   'R02_S064065_C0001_I0': 119, 'R02_S068069_C0001_I0': 102, 'R03_S070071_C0001_I1': 136, 
                   'R03_S070071_C0001_I2': 0, 'R03_S072073_C0001_I1': 137, 'R03_S072073_C0001_I2': 0, 
                   'R03_S077078_C0001_I1': 171, 'R03_S077078_C0001_I2': 0, 'R03_S079080081_C0001_I0': 267, 
                   'R03_S088089_C0001_I1': 148, 'R03_S088089_C0001_I2': 0, 'R03_S102103104_C0001_I0': 188, 
                   'R03_S105106_C0001_I1': 195, 'R03_S105106_C0001_I2': 0, 'R03_S113114115_C0001_I0': 204, 
                   'R07_S149150_C0001_I1': 163, 'R07_S149150_C0001_I2': 0, 'R07_S151152153_C0001_I0': 206, 
                   'R08_S158159160_C0001_I0': 168, 'R08_S161162163164_C0001_I1': 247, 'R08_S161162163164_C0001_I2': 0, 
                   'R08_S165166_C0001_I1': 149, 'R08_S165166_C0001_I2': 0, 'R08_S167168_C0001_I1': 140, 
                   'R08_S167168_C0001_I2': 0, 'R08_S167168_C0001_I3': 0, 'R08_S169170171172_C0001_I1': 226, 
                   'R08_S169170171172_C0001_I2': 0, 'R11_S193194_C0001_I1': 126, 'R11_S193194_C0001_I2': 0, 
                   'R13_S212213214_C0001_I0': 169, 'R13_S215216217218_C0001_I1': 177, 'R15_S228229_C0001_I1': 150, 
                   'R15_S228229_C0001_I2': 0, 'R16_S242243244245_C0001_I1': 225, 'R52_S272273_C0001_I2': 0, 
                   'R54_S294295296297298299_C0001_I1': 467, 'R54_S294295296297298299_C0001_I2': 0, 
                   'R54_S300301302_C0001_I0': 223, 'R55_S306307_C0001_I1': 71, 'R55_S308309310311_C0001_I1': 828, 
                   'R55_S308309310311_C0001_I2': 0, 'R62_S367368369370_C0001_I2': 0, 'R66_S386387388389390391_C0001_I2': 0, 
                   'R67_S392393394_C0001_I0': 231, 'R67_S395396397398_C0001_I1': 276, 'R67_S395396397398_C0001_I2': 0, 
                   'R70_S409410411_C0001_I0': 220, 'R72_S424425_C0001_I1': 197, 'R72_S424425_C0001_I2': 0, 
                   'R77_S429430_C0001_I2': 0, 'R80_S455456457_C0001_I0': 284, 'R04_S134135_C0009_I1': 110, 
                   'R12_S203204205_C0009_I2': 0, 'R10_S186187188_C0009_I2': 0, 'R12_S203204205_C0009_I1': 164, 
                   'R16_S242243244245_C0001_I2': 0, 'R53_S283284285_C0001_I0': 365, 'R53_S286287288289_C0001_I1': 795, 
                   'R74_S431432433434435436_C0001_I2': 0}


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


def split_detection_json(raw_json, split_json, start_time=0, fps=25):
    detection_dic = json2dic(jsonpath=raw_json)['frames']
    frames_idx = [*detection_dic.keys()]
    start_frame_idx = int(round(start_time * fps))
    if start_frame_idx != 0:
        split_dic = {}
        for key, value in detection_dic.items():
            if int(key) >= start_frame_idx:
                split_dic[str(int(key) - start_frame_idx)] = value
        detection_dic = split_dic
    store_dir = os.path.split(split_json)[0]
    if not os.path.exists(store_dir):
        os.makedirs(store_dir, exist_ok=True)
    json2dic(jsonpath=split_json, dic=detection_dic)
    return None


def input_interface(data_dir, store_dir):
    if not os.path.exists(store_dir):
        os.makedirs(store_dir, exist_ok=True)
    json_paths = glob.glob(os.path.join(data_dir, '*.json'))
    for json_path in tqdm.tqdm(json_paths):
        json_name = os.path.splitext(os.path.split(json_path)[-1])[0]
        json_list = json_name.split('_')
        if '_C00' in json_path:
            start = split_c0001_dic['_'.join(json_list[:4])]
            split_detection_json(raw_json=json_path, split_json=os.path.join(store_dir, '{}.json'.format('_'.join(json_list[:-1]).replace('_C00', '_C'))), 
                                 start_time=start)
        else:
            split_detection_json(raw_json=json_path, split_json=os.path.join(store_dir, '{}.json'.format('_'.join(json_list[:-1]))))
    return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser('split_detection_json')
    parser.add_argument('data_dir', type=str, help='data dir')
    parser.add_argument('store_dir', type=str, help='store dir')
    args = parser.parse_args()
    
    input_interface(data_dir=args.data_dir, store_dir=args.store_dir)
