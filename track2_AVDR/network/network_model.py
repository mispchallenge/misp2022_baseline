#!/usr/bin/env python
# _*_ coding: UTF-8 _*_
import logging

import torch
import torch.nn as nn
from .network_common_module import expend_params

logging.getLogger('matplotlib.font_manager').disabled = True


class BaseModelWorker(object):
    def __init__(self, log_type, logger=None):
        super(BaseModelWorker, self).__init__()
        self.logger = logger
        self.log_type = log_type
        self._build_map()

    def __call__(self, network_name, network_setting, pretrained_num=0, pretrained_model=None, replace_keys=None,
                 unload_keys=None, fixed_type=None, fixed_params=None, fixed_keys=None, **other_params):
        model = self._init_network(network_name=network_name, network_setting=network_setting)
        model = self._load_params(model=model, pretrained_num=pretrained_num, pretrained_model=pretrained_model,
                                  replace_keys=replace_keys, unload_keys=unload_keys)
        model = self._fix_params(model=model, fixed_type=fixed_type, fixed_params=fixed_params, fixed_keys=fixed_keys)
        # model = self._move_device(model=model, used_gpu=used_gpu, distributed=distributed, local_rank=local_rank)
        return model

    def _build_map(self):
        self.name2network = {}
        return None

    def _init_network(self, network_name, network_setting):
        write_log(content='Using network: {}'.format(network_name), level=self.log_type, logger=self.logger)
        model = self.name2network[network_name](**network_setting)
        write_log(content='model struct is {}'.format(model), level=self.log_type, logger=self.logger)
        return model

    def _load_params(self, model, pretrained_num=0, pretrained_model=None, replace_keys=None, unload_keys=None):
        if not pretrained_num:
            write_log(content='No exist model state', level=self.log_type, logger=self.logger)
            write_log(level=self.log_type, logger=self.logger, content='Random params: {}'.format(
                ','.join(sorted([*model.state_dict().keys()]))))
        else:
            model_dict = model.state_dict()
            pretrained_model = expend_params(value=pretrained_model, length=pretrained_num)
            replace_keys = expend_params(value=replace_keys, length=pretrained_num)
            unload_keys = expend_params(value=unload_keys, length=pretrained_num)
            pretrained_params = []
            extra_params = []
            for pretrained_id in range(pretrained_num):
                model_dict, sub_pretrained_params, sub_extra_params = self._replace_params(
                    model_dict=model_dict, pretrained_model=pretrained_model[pretrained_id],
                    replace_keys=replace_keys[pretrained_id], unload_keys=unload_keys[pretrained_id])
                pretrained_params += sub_pretrained_params
                extra_params += sub_extra_params
            model.load_state_dict(model_dict)
            write_log(content='Pretrained params: {}'.format(','.join(sorted(pretrained_params))), level=self.log_type,
                      logger=self.logger)
            write_log(level=self.log_type, logger=self.logger, content='Random params: {}'.format(
                ','.join(sorted([*(set(model_dict.keys()) - set(pretrained_params))]))))
            write_log(
                content='Extra params: {}'.format(','.join(sorted(extra_params))), level=self.log_type,
                logger=self.logger)
        return model

    def _replace_params(self, model_dict, pretrained_model=None, replace_keys=None, unload_keys=None):
        if isinstance(pretrained_model, (dict, str, tuple, list)):
            if isinstance(pretrained_model, str):
                write_log(content='Load pretrained model from {}'.format(pretrained_model), level=self.log_type,
                          logger=self.logger)
                pretrained_dict = torch.load(pretrained_model, map_location=lambda storage, loc: storage)
            elif isinstance(pretrained_model, (tuple, list)):
                write_log(content='Load pretrained model from {}'.format(' of '.join(pretrained_model[::-1])),
                          level=self.log_type, logger=self.logger)
                pretrained_dict = torch.load(pretrained_model[0],
                                             map_location=lambda storage, loc: storage)[pretrained_model[1]]
            else:
                write_log(content='Load model from previous epoch', level=self.log_type, logger=self.logger)
                pretrained_dict = pretrained_model
            pretrained_params = []
            extra_params = []
            for pretrained_param, pretrained_value in pretrained_dict.items():
                if replace_keys:
                    if isinstance(replace_keys, dict):
                        for raw_key in replace_keys:
                            if raw_key in pretrained_param and (
                                    not replace_keys[raw_key] or replace_keys[raw_key] not in pretrained_param):
                                pretrained_param = pretrained_param.replace(raw_key, replace_keys[raw_key])
                                break
                    elif isinstance(replace_keys, str):
                        pretrained_param = replace_keys+pretrained_param
                    else:
                        raise ValueError('unknown replace_keys: {}'.format(replace_keys))
                unload_token = False
                if unload_keys:
                    for unload_key in unload_keys:
                        if unload_key in pretrained_param:
                            unload_token = True
                            break
                if pretrained_param in model_dict and not unload_token:
                    model_dict[pretrained_param] = pretrained_value
                    pretrained_params.append(pretrained_param)
                else:
                    extra_params.append(pretrained_param)
        else:
            raise ValueError('unknown model_state')
        return model_dict, pretrained_params, extra_params

    def _fix_params(self, model, fixed_type=None, fixed_params=None, fixed_keys=None):
        if not fixed_type:
            training_params = []
            fixed_params = []
            for param, value in model.named_parameters():
                if not value.requires_grad:
                    fixed_params.append(param)
                else:
                    training_params.append(param)
        elif fixed_type == 'fixed_params':
            assert isinstance(fixed_params, (str, list, tuple)), 'unknown fixed_params'
            training_params = []
            if isinstance(fixed_params, str):
                fixed_params = [fixed_params]
            if isinstance(fixed_params, tuple):
                fixed_params = [*fixed_params]
            for param, value in model.named_parameters():
                if not value.requires_grad:
                    fixed_params.append(param)
                elif param in fixed_params:
                    value.requires_grad = False
                    fixed_params.append(param)
                else:
                    training_params.append(param)
        elif fixed_type == 'fixed_keys':
            fixed_params = []
            training_params = []
            assert isinstance(fixed_keys, (str, list, tuple)), 'unknown fixed_keys'
            if isinstance(fixed_keys, str):
                fixed_keys = [fixed_keys]
            for param, value in model.named_parameters():
                if not value.requires_grad:
                    fixed_params.append(param)
                else:
                    fixed_token = False
                    for fixed_key in fixed_keys:
                        if fixed_key in param:
                            fixed_token = True
                            break
                    if fixed_token:
                        value.requires_grad = False
                        fixed_params.append(param)
                    else:
                        training_params.append(param)
        else:
            raise ValueError('unknown fix_type')
        write_log(content='Fixed params: {}'.format(','.join(sorted(fixed_params))), level=self.log_type,
                  logger=self.logger)
        write_log(content='Training params: {}'.format(','.join(sorted(training_params))), level=self.log_type,
                  logger=self.logger)
        param_num = 0
        param_requires_grad_num = 0
        for x in model.parameters():
            param_num += x.numel()
            if x.requires_grad:
                param_requires_grad_num += x.numel()
        write_log(content='num of parameters is {}, {} requires_grad'.format(param_num, param_requires_grad_num),
                  level=self.log_type, logger=self.logger)
        return model

    def _move_device(self, model, used_gpu=None, distributed=None, local_rank=0):
        if used_gpu:
            if distributed:
                write_log(content='use multi gpus', level=self.log_type, logger=self.logger)
                # model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
                model = model.cuda(local_rank)
                model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)
            else:
                write_log(content='use single gpu', level=self.log_type, logger=self.logger)
                model = model.cuda()
        else:
            write_log(content='use cpu', level=self.log_type, logger=self.logger)
        return model


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
