#!/usr/bin/env python
# _*_ coding: UTF-8 _*_
import os
import time
import random
import numpy as np
import argparse
import logging

import torch
import torch.distributed as dist

from pipeline_train import main_train
from pipeline_predict import main_predict
from tool.data_io import safe_load, safe_store
from tool.match_search import search

logging.getLogger('matplotlib.font_manager').disabled = True


def init_logger(file_level_modes, **other_params):
    """
    initialize logger
    :param file_level_modes: list of (filepath, log_level, log_mode)
    :param other_params: reserved interface
    :return: logger
    """
    def handler_init(file_level):
        level_map = {
            'debug': logging.DEBUG, 'info': logging.INFO, 'warning': logging.WARNING, 'error': logging.ERROR,
            'critical': logging.CRITICAL}
        sub_file, sub_level, sub_mode = file_level
        if '.log' in sub_file:
            os.makedirs(os.path.split(sub_file)[0], exist_ok=True)
            sub_handler = logging.FileHandler(sub_file, mode=sub_mode)
        elif sub_file == 'command_line':
            sub_handler = logging.StreamHandler()
        else:
            raise ValueError('unknown handle file')
        if sub_level in level_map:
            sub_handler.filter = lambda record: record.levelno == level_map[sub_level]
        else:
            raise ValueError('unknown logger level')
        sub_handler.setFormatter(logging.Formatter(fmt='%(asctime)s|%(message)s', datefmt='%Y%m%d-%H:%M:%S'))
        return sub_handler

    used_logger = logging.getLogger()
    used_logger.setLevel(level=logging.DEBUG)
    for handler_id in range(len(file_level_modes)):
        used_logger.addHandler(handler_init(file_level=file_level_modes[handler_id]))
    return used_logger


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


def init_args(args):
    if args.dist_url == 'env://' and args.world_size == -1:
        args.world_size = int(os.environ['WORLD_SIZE'])

    if args.dist_url == 'env://' and args.rank == -1:
        args.rank = int(os.environ['RANK'])

    if args.dist_url == 'env://' and args.local_rank == -1:
        args.local_rank = int(os.environ['LOCAL_RANK'])
    return None


def init_dist(args, logger):
    torch.cuda.device(args.local_rank)
    args.distributed = args.world_size > 0
    if args.distributed:
        write_log(content='world_size = {}, rank = {}, gpuid = {}'.format(
            args.world_size, args.rank, args.local_rank), level='debug', logger=logger)
        write_log(
            content='MASTER_PORT = {}, MASTER_ADDR = {}'.format(os.environ['MASTER_PORT'], os.environ['MASTER_ADDR']),
            level='debug', logger=logger)
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url)
    return None


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser('train')
    parser.add_argument('-r', '--exp_dir', type=str, default='./exp',
                        help='directory of experiment')
    parser.add_argument('-y', '--yaml_dir', type=str, default='./exp_conf', help='directory of config yaml')
    parser.add_argument('-c', '--config', type=str, required=True, help='config id')
    parser.add_argument('-m', '--mode', type=str, nargs='+', default=['train'], help='select run mode')

    # train setting
    parser.add_argument('-rs', '--random_seed', type=int, default=123456, help='random_seed')
    parser.add_argument('-be', '--begin_epoch', type=int, default=0, help='begin epoch idx')
    parser.add_argument('-es', '--epoch_sum', type=int, default=100, help='epoch sum')
    parser.add_argument('-sp', '--save_point', type=int, default=1, help='model save point')
    parser.add_argument('-pf', '--print_freq', type=int, default=500, help='print frequent during training')
    parser.add_argument('-ss', '--stages', type=str, nargs='+', default=['train', 'test'],
                        help='stages ran during training')
    parser.add_argument('-ms', '--metrics', type=str, nargs='+', default=['mse'], help='used metrics during training')
    parser.add_argument('-si', '--stage_id', type=int, default=1, help='evaluation stage id used during training')
    parser.add_argument('-ci', '--crown_id', type=int, default=0, help='evaluation metric id used during training')
    parser.add_argument('-co', '--comparison', type=str, default='min',
                        help='comparison of evaluation metric used during training')

    # predict setting
    parser.add_argument('-pd', '--predict_data', nargs='+', type=str, default='test', help='predict data item')
    parser.add_argument('-um', '--used_model', type=int, default=-1, help='used model during predicting, -1 means best')

    # for distributed
    parser.add_argument('--world-size', type=int, default=-1, help='number of nodes for distributed training')
    parser.add_argument('--rank', type=int, default=-1, help='node rank for distributed training')
    parser.add_argument('--dist-url', type=str, default='env://', help='url used to set up distributed training')
    parser.add_argument('--dist-backend', type=str, default='nccl', help='distributed backend')
    parser.add_argument('--local_rank', type=int, default=-1, help='gpu local_rank for distributed training')
    input_args = parser.parse_args()

    # find config file based on config id
    config_candidates = search(root_dir=input_args.yaml_dir, regular_expression=input_args.config)
    if len(config_candidates) == 0:
        raise ValueError('regular_expression {} is error, find no possible config'.format(input_args.config))
    elif len(config_candidates) > 1:
        raise ValueError('regular_expression {} is error, find more than 1 possible configs: {}'.format(input_args.config, config_candidates))
    else:
        used_config = config_candidates[0]
    
    # find exp folder based on config id
    exp_folder_candidates = search(root_dir=input_args.exp_dir, regular_expression=input_args.config)
    if len(exp_folder_candidates) == 0:
        exp_folder = os.path.join(input_args.exp_dir, os.path.splitext(os.path.split(used_config)[-1])[0])
    elif len(exp_folder_candidates) > 1:
        raise ValueError('regular_expression {} is error, find more than 1 possible exp folder: {}'.format(input_args.config, exp_folder_candidates))
    else:
        exp_folder = exp_folder_candidates[0]
    
    input_hparams = safe_load(file=used_config)
    init_args(args=input_args)

    for stage_id, stage in enumerate(input_args.mode):
        # logger config
        if stage == 'train':
            safe_store(file=os.path.join(exp_folder, 'train_config.yml'), data=input_hparams, mode='cover')
            log_mode = 'w' if input_args.begin_epoch == 0 else 'a'
            logger_handler = [[os.path.join(exp_folder, 'train_debug_world_{}_rank_{}_local_{}.log'.format(
                input_args.world_size, input_args.rank, input_args.local_rank)), 'debug', log_mode]]
            if input_args.rank == 0:
                logger_handler.extend([[os.path.join(exp_folder, 'model.log'), 'warning', 'w'],
                                       [os.path.join(exp_folder, 'train_info.log'), 'info', log_mode]])
            
             # init logger and dist
            run_logger = init_logger(file_level_modes=logger_handler)
            if stage_id == 0:
                init_dist(args=input_args, logger=run_logger)
            time.sleep(input_args.local_rank)
            
            # set random seed
            write_log(content='Random seed: {}'.format(input_args.random_seed), level='info', logger=run_logger)
            set_random_seed(seed=input_args.random_seed)

            main_train(exp_dir=exp_folder, hparams=input_hparams, args=input_args, logger=run_logger)
        elif stage == 'predict':
            # if input_args.used_model == -1:
            #     predict_store_dir = os.path.join(exp_folder, 'predict_best_{}'.format(input_args.predict_data))
            # else:
            #     predict_store_dir = os.path.join(exp_folder, 'predict_epoch{}_{}'.format(input_args.used_model, input_args.predict_data))
            for predict_idx, predict_data in enumerate(input_args.predict_data):
                if input_args.used_model == -1:
                    used_params = os.path.join(exp_folder, 'best.tar')
                    predict_store_dir = os.path.join(exp_folder, 'predict_best_{}'.format(predict_data))
                else:
                    used_params = os.path.join(exp_folder, 'model', 'epoch{}.tar'.format(input_args.used_model))
                    predict_store_dir = os.path.join(exp_folder, 'predict_epoch{}_{}'.format(input_args.used_model, predict_data))
                
                safe_store(file=os.path.join(predict_store_dir, 'predict_config.yml'), data=input_hparams, mode='cover')
                logger_handler = [[os.path.join(predict_store_dir, 'predict_world_{}_rank_{}_local_{}.log'.format(
                        input_args.world_size, input_args.rank, input_args.local_rank)), 'info', 'a']]
                
                # init logger and dist
                run_logger = init_logger(file_level_modes=logger_handler)
                if stage_id == 0 and predict_idx == 0:
                    init_dist(args=input_args, logger=run_logger)
                time.sleep(input_args.local_rank)
                
                # main_predict(exp_dir=exp_folder, hparams=input_hparams, args=input_args, logger=run_logger)
                main_predict(predict_input_dic=input_hparams['data'][predict_data], predict_store_dir=predict_store_dir, used_params=used_params, 
                             hparams=input_hparams, args=input_args, logger=run_logger)
        else:
            raise NotImplementedError('unknown mode: {}'.format(stage))

        del run_logger
