#!/usr/bin/env python
# _*_ coding: UTF-8 _*_
import torch
import torch.nn as nn
import torch.distributed as dist

import os
import glob
import logging
from tqdm import tqdm

from pipeline_train import forward_worker
from loader_audio_visual_pdf import get_data_loader
from network.network_index import AVSRWorker
from tool.data_io import safe_store, safe_load

logging.getLogger('matplotlib.font_manager').disabled = True


def write_log(content, logger=None, level=None, **other_params):
    """
    write log
    :param content: content need to write
    :param level: level of content or None
    :param logger: False or logger
    :param other_params: reserved interface
    :return: None
    """
    if not logger:
        pass
    elif logger == 'print':
        print(content)
    elif isinstance(logger, logging.Logger):
        if not level:
            pass
        else:
            assert level in ['debug', 'info', 'warning', 'error', 'critical'], 'unknown level'
            getattr(logger, level)(content)
    else:
        raise NotImplementedError('unknown logger')
    return None


def main_predict(predict_input_dic, predict_store_dir, used_params, hparams, args, logger, fuse_input=True, **other_params):
    predict_dic = {}
    # get data_loaders & data_samplers
    data_loader, data_sampler = get_data_loader(**predict_input_dic, logger=logger,
                                                distributed=args.distributed, key_output=True)

    # init model
    model_worker = AVSRWorker(log_type='info', logger=logger)
    model_hparams = hparams['model']
    # if used_model == -1, use the params of best.tar to init network
    # if args.used_model == -1:
    #     used_params = os.path.join(exp_dir, 'best.tar')
    #     predict_store_dir = os.path.join(exp_dir, 'predict_best_{}'.format(args.predict_data))
    # else:
    #     used_params = os.path.join(exp_dir, 'model', 'epoch{}.tar'.format(args.used_model))
    #     predict_store_dir = os.path.join(exp_dir, 'predict_epoch{}_{}'.format(args.used_model, args.predict_data))
    model_hparams = {**model_hparams, 'pretrained_num': 1, 'pretrained_model': [used_params, 'model_params'],
                     'replace_keys': None, 'unload_keys': None}

    model = model_worker(**model_hparams)
    # copy model to gpu
    model = model.cuda(args.local_rank)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank)

    # start
    write_log(logger=logger, content='{}Predicting with {}{}'.format('-'*12, used_params, '-'*12), level='info')
    # processing bar
    batch_bar = None
    if args.rank == 0:
        batch_bar = tqdm(total=len(data_loader), leave=False, desc='Rank {} Predict Batch'.format(args.rank))
    model.eval()
    output_settings = hparams['data']['model_output']
    with torch.no_grad():
        for batch_keys, *batch_data in data_loader:
            batch_model_output = forward_worker(
                model=model, loader_output=batch_data, model_input_setting=hparams['data']['model_input'], 
                loss_input_setting=None, loss_function=None, args=args, skip_loss=True)
            for key_idx in range(len(batch_keys)):
                key = batch_keys[key_idx]
                predict_item = {}
                for store_item_key, store_item_idx in output_settings['store_items'].items():
                    store_item_data = batch_model_output[store_item_idx][key_idx].cpu()
                    if store_item_key in output_settings['store_item2length']:
                        store_item_length = batch_model_output[output_settings['store_item2length'][store_item_key]][key_idx].cpu().long()
                        store_item_data = store_item_data[:store_item_length]
                    store_item_path = os.path.join(predict_store_dir, store_item_key, '{}.pt'.format(key))
                    predict_item[store_item_key] = store_item_path
                    safe_store(file=store_item_path, data=store_item_data, mode='cover', ftype='torch')
                predict_dic[key] = predict_item
            dist.barrier()
            if args.rank == 0:
                batch_bar.update(1)
    safe_store(file=os.path.join(predict_store_dir, 'predict_world_{}_rank_{}_local_{}.json'.format(
        args.world_size, args.rank, args.local_rank)), data=predict_dic, mode='cover', ftype='json')
    dist.barrier()
    if args.rank == 0:
        batch_bar.close()
        all_predict_dic = {}
        for sub_json in glob.glob(os.path.join(predict_store_dir, 'predict_world_*_rank_*_local_*.json')):
            all_predict_dic.update(safe_load(sub_json))
        safe_store(file=os.path.join(predict_store_dir, 'predict.json'), data=all_predict_dic, mode='cover',
                   ftype='json')
        if fuse_input:
            keys_list = []
            duration_list = []
            key2path_dic = {}
            input_annotates = predict_input_dic['annotate']
            if not isinstance(input_annotates, list):
                input_annotates = [input_annotates]
            for annotate_id in range(len(input_annotates)):
                data_dic = safe_load(file=input_annotates[annotate_id])
                keys_list += data_dic['keys']
                duration_list += data_dic['duration']
                key2path_dic.update(data_dic['key2path'])
            for key in keys_list:
                for sub_key, sub_value in all_predict_dic[key].items():
                    key2path_dic[key][sub_key] = sub_value
            safe_store(file=os.path.join(predict_store_dir, 'data_with_predicted.json'), mode='cover', ftype='json',
                       data={'keys': keys_list, 'duration': duration_list, 'key2path': key2path_dic})
    return None
