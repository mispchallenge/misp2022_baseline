#!/usr/bin/env python
# _*_ coding: UTF-8 _*_
import os
import codecs
import json
import glob
import argparse
import numpy as np
from kaldiio import save_ark
from tqdm import tqdm
from multiprocessing import Pool
import torch
import torch.nn.functional as nf


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


def ark_generator(data_dic, ark_path, norm=0., job_id=0):
    for key, path in tqdm(data_dic.items(), desc=str(job_id), leave=True):
        embedding = torch.load(path, map_location=lambda storage, loc: storage)
        log_probability = nf.log_softmax(embedding, dim=-1)
        # dimmin = min(log_probability.shape[1], norm.shape[0])
        # log_probability = log_probability[:, :dimmin]
        # norm = norm[:dimmin]
        # import pdb; pdb.set_trace()
        save_ark(ark=ark_path, array_dict={key: log_probability.numpy() - norm}, append=True)
    return None


def generate_ark(data_dic, predict_folder, count_path='none', num_jobs=1):
    if count_path == 'none':
        ark_folder = os.path.join(predict_folder, 'ark_to_decode_log_posteriors')
        ark_name = 'log_posteriors'
        norm = 0.
    else:
        ark_folder = os.path.join(predict_folder, 'ark_to_decode_log_likelihoods')
        ark_name = 'log_likelihoods'
        with open(count_path) as f:
            row = next(f).strip().strip('[]').strip()
            counts = np.array([np.float32(v) for v in row.split()])
        norm = np.log(counts / np.sum(counts))

    if not os.path.exists(ark_folder):
        os.makedirs(ark_folder, exist_ok=True)

    data_dic_for_job = [{} for _ in range(num_jobs)]
    idx = 0
    for k, v in data_dic.items():
        data_dic_for_job[idx % num_jobs][k] = v
        idx += 1

    print('All {} sample, split to {} ark'.format(idx, num_jobs))

    if num_jobs > 1:
        pool = Pool(processes=num_jobs)
        for i in range(num_jobs):
            pool.apply_async(ark_generator, kwds={
                'data_dic': data_dic_for_job[i], 'norm': norm, 'job_id': i+1,
                'ark_path': os.path.join(ark_folder, '{}.{}.ark'.format(ark_name, i+1))})
        pool.close()
        pool.join()
    else:
        ark_generator(data_dic=data_dic_for_job[0], norm=norm, job_id=1,
                      ark_path=os.path.join(ark_folder, '{}.{}.ark'.format(ark_name, 1)))
    return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser('decode_generate_ark')
    parser.add_argument('-r', '--exp_dir', type=str, default='/yrfs1/intern/hangchen2/experiment/EASE',
                        help='directory of experiment')
    parser.add_argument('-c', '--config', type=str, required=True, help='config id')

    # predict setting
    parser.add_argument('-pd', '--predict_data', type=str, default='test', help='predict data item')
    parser.add_argument('-pi', '--predict_item', type=str, default='predict_posterior', help='predict item')
    parser.add_argument('-um', '--used_model', type=int, default=-1, help='used model during predicting, -1 means best')

    # generate counts
    parser.add_argument('-ac', '--ali_count', type=str, default='none', help='path of count file')
    # number jobs
    parser.add_argument('-nj', '--num_jobs', type=int, default=1, help='number of jobs')

    input_args = parser.parse_args()

    # find exp fold based on config id
    exp_regular_expression = os.path.join(input_args.exp_dir, input_args.config)
    if exp_regular_expression[-1] != '*':
        exp_regular_expression = exp_regular_expression + '*'
    possible_exp_list = glob.glob(exp_regular_expression)
    if len(possible_exp_list) != 1:
        raise ValueError('config id: {} is error, find possible exp folders: {}'.format(input_args.config,
                                                                                         possible_exp_list))
    exp_folder = possible_exp_list[0]

    if input_args.used_model == -1:
        predict_store_dir = os.path.join(exp_folder, 'predict_best_{}'.format(input_args.predict_data))
    else:
        predict_store_dir = os.path.join(exp_folder, 'predict_epoch{}_{}'.format(input_args.used_model,
                                                                                 input_args.predict_data))

    # interface for data_with_predicted.json
    data_with_predicted_dic = json2dic(jsonpath=os.path.join(predict_store_dir, 'data_with_predicted.json'))
    input_data_dic = {}
    for key in data_with_predicted_dic['keys']:
        input_data_dic[key] = data_with_predicted_dic['key2path'][key][input_args.predict_item]

    generate_ark(data_dic=input_data_dic, predict_folder=predict_store_dir, count_path=input_args.ali_count, 
                 num_jobs=input_args.num_jobs)
