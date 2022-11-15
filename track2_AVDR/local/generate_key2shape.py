#!/usr/bin/env python
# _*_ coding: UTF-8 _*_
import os
import glob
import json
import codecs
import torch
import argparse
from tqdm import tqdm
import numpy as np
from multiprocessing import Pool

def generate_key2shape_worker(all_file_list, processing_id=None, processing_num=None):
    key2shape_dict = {}
    # 
    # for file_path in tqdm(all_file_list):
    for file_idx in tqdm(range(len(all_file_list)), leave=True, desc='0' if processing_id is None else str(processing_id)):
        if processing_id is None:
            processing_token = True
        else:
            if file_idx % processing_num == processing_id:
                processing_token = True
            else:
                processing_token = False
        if processing_token:
            file_path = all_file_list[file_idx]
            # import pdb; pdb.set_trace();
            key = os.path.splitext(os.path.split(file_path)[-1])[0]
            #start, end = key.split('_')[-1].split('-')
            # duration = round((int(start) - int(end)) / 100., 2)
            data = torch.load(file_path)
            if isinstance(data, dict):
                shape = data['stamp'][-1]
            elif isinstance(data, list):
                data = torch.from_numpy(np.array(data))
                torch.save(data, file_path)
                shape = [*data.shape]
            else:
                shape = [*data.shape]
            key2shape_dict[key] = shape

    return key2shape_dict


def generate_key2shape_manager(file_pattern, processing_num=1):
    all_file_list = sorted(glob.glob(file_pattern))
    if processing_num > 1:
        all_result = []
        pool = Pool(processes=processing_num)
        for i in range(processing_num):
            part_result = pool.apply_async(generate_key2shape_worker, kwds={'all_file_list': all_file_list, 'processing_id': i, 'processing_num': processing_num})
            all_result.append(part_result)
        pool.close()
        pool.join()
        key2shape_dict = {}
        for item in all_result:
            key2shape_dict = {**key2shape_dict, **item.get()}
            
    else:
        key2shape_dict = generate_key2shape_worker(all_file_list=all_file_list)
    
    with codecs.open(os.path.join(os.path.split(os.path.split(file_pattern)[0])[0], 'key2shape.json'), 'w') as handle:
        json.dump(key2shape_dict, handle)    
    return None
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser('generate_key2shape')
    parser.add_argument('file_pattern', type=str, default='./local/tmp/wpe.scp', help='list file of wav, format is scp')
    parser.add_argument('-nj', type=int, default=1, help='list file of wav, format is scp')
    args = parser.parse_args()
    generate_key2shape_manager(file_pattern=args.file_pattern, processing_num=args.nj)

# python local/generate_key2shape.py /raw7/cv1/hangchen2/misp2022/Released/audio/${x}/far_wpe_beamformit_segment/pt/'*.pt'