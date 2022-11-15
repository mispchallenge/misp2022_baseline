#!/usr/bin/env python
# coding:utf-8
import torch
import logging
import torch.optim as opt
from .optimizer_AdaBound import AdaBound, AdaBoundW

logging.getLogger('matplotlib.font_manager').disabled = True


class OptimizerWorker(object):
    def __init__(self, log_type, logger=None):
        super(OptimizerWorker, self).__init__()
        self.log_type = log_type
        self.logger = logger
        self._build_map()

    def __call__(self, optimizer_type, model_params, group_num=1, optimizer_state=None, force_lr=False,
                 common_setting=None, unusual_setting=None, **other_setting):
        optimizer_setting = {}
        if common_setting is not None:
            optimizer_setting['common_setting'] = common_setting
        if unusual_setting is not None:
            optimizer_setting['unusual_setting'] = unusual_setting
        optimizer = self.init_optimizer(
            optimizer_type=optimizer_type, model_params=model_params, group_num=group_num, **optimizer_setting)
        optimizer = self.load_state(optimizer=optimizer, optimizer_state=optimizer_state)
        optimizer = self.change_lr(optimizer=optimizer, force_lr=force_lr)
        return optimizer

    def _build_map(self):
        self.name2optimizer = {
            'Adadelta': opt.Adadelta, 'Adagrad': opt.Adagrad, 'Adam': opt.Adam, 'AdamW': opt.AdamW,
            'SparseAdam': opt.SparseAdam, 'Adamax': opt.Adamax, 'ASGD': opt.ASGD, 'LBFGS': opt.LBFGS,
            'RMSprop': opt.RMSprop, 'Rprop': opt.Rprop, 'SGD': opt.SGD, 'AdaBound': AdaBound, 'AdaBoundW': AdaBoundW}
        self.name2default_setting = {
            'Adadelta': {'lr': 1.0, 'rho': 0.9, 'eps': 1e-06, 'weight_decay': 0},
            'Adagrad': {'lr': 0.01, 'lr_decay': 0, 'weight_decay': 0, 'initial_accumulator_value': 0},
            'Adam': {'lr': 0.001, 'betas': (0.9, 0.999), 'eps': 1e-08, 'weight_decay': 0, 'amsgrad': False},
            'AdamW': {'lr': 0.001, 'betas': (0.9, 0.999), 'eps': 1e-08, 'weight_decay': 0.01, 'amsgrad': False},
            'SparseAdam': {'lr': 0.001, 'betas': (0.9, 0.999), 'eps': 1e-08},
            'Adamax': {'lr': 0.002, 'betas': (0.9, 0.999), 'eps': 1e-08, 'weight_decay': 0},
            'ASGD': {'lr': 0.01, 'lambd': 0.0001, 'alpha': 0.75, 't0': 1000000.0, 'weight_decay': 0},
            'LBFGS': {'lr': 1, 'max_iter': 20, 'max_eval': None, 'tolerance_grad': 1e-07, 'tolerance_change': 1e-09,
                      'history_size': 100, 'line_search_fn': None},
            'RMSprop': {'lr': 0.01, 'alpha': 0.99, 'eps': 1e-08, 'weight_decay': 0, 'momentum': 0, 'centered': False},
            'Rprop': {'lr': 0.01, 'etas': (0.5, 1.2), 'step_sizes': (1e-06, 50)},
            'SGD': {'lr': 10, 'momentum': 0, 'dampening': 0, 'weight_decay': 0, 'nesterov': False},
            'AdaBound': {'lr': 1e-3, 'betas': (0.9, 0.999), 'final_lr': 0.1, 'gamma': 1e-3, 'eps': 1e-8,
                         'weight_decay': 0, 'amsbound': False},
            'AdaBoundW': {'lr': 1e-3, 'betas': (0.9, 0.999), 'final_lr': 0.1, 'gamma': 1e-3, 'eps': 1e-8,
                          'weight_decay': 0, 'amsbound': False}}

    def init_optimizer(self, optimizer_type, model_params, group_num=1, **optimizer_setting):
        write_log(content='Using optimizer: {}'.format(optimizer_type), level=self.log_type, logger=self.logger)
        model_params = model_params if isinstance(model_params, list) else [model_params]
        common_setting = optimizer_setting.get('common_setting', {})
        unusual_setting = optimizer_setting.get('unusual_setting', [{}])
        assert group_num == len(unusual_setting) and group_num == len(model_params)
        default_params = self.name2default_setting[optimizer_type]
        optimizer_inputs = []
        for group_idx in range(group_num):
            optimizer_input = {'params': model_params[group_idx], **default_params}
            optimizer_input.update(common_setting)
            optimizer_input.update(unusual_setting[group_idx])
            optimizer_inputs.append(optimizer_input)
        optimizer = self.name2optimizer[optimizer_type](optimizer_inputs)
        return optimizer

    def load_state(self, optimizer, optimizer_state=False):
        if not optimizer_state:
            write_log(content='No exist optimizer state', level=self.log_type, logger=self.logger)
        elif isinstance(optimizer_state, (dict, str, tuple, list)):
            if isinstance(optimizer_state, str):
                already_optimizer_state = torch.load(optimizer_state, map_location=lambda storage, loc: storage)
            elif isinstance(optimizer_state, (tuple, list)):
                already_optimizer_state = torch.load(
                    optimizer_state[0], map_location=lambda storage, loc: storage)[optimizer_state[1]]
            else:
                already_optimizer_state = optimizer_state
            optimizer.load_state_dict(already_optimizer_state)
            write_log(content='Loading exist optimizer state', level=self.log_type, logger=self.logger)
        else:
            raise ValueError('unknown optimizer_state')
        return optimizer

    def change_lr(self, optimizer, force_lr=False):
        group_num = len(optimizer.param_groups)
        if not force_lr:
            write_log(content='Invariable learning rate', level=self.log_type, logger=self.logger)
        elif isinstance(force_lr, (list, float)):
            force_lr = [force_lr for _ in range(group_num)] if isinstance(force_lr, float) else force_lr
            for i in range(group_num):
                optimizer.param_groups[i]['lr'] = force_lr[i]
            write_log(content='Force learning rate to: {}'.format(force_lr), level=self.log_type, logger=self.logger)
        else:
            raise ValueError('unknown force_lr')
        return optimizer


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
