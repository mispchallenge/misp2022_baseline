#!/usr/bin/env python
# coding:utf-8
import math
import logging
import torch
from torch.optim import Optimizer
from torch.optim import lr_scheduler as lrs


class LRSchedulerWorker(object):
    def __init__(self, log_type, logger=None):
        super(LRSchedulerWorker, self).__init__()
        self.log_level = log_type
        self.logger = logger
        self._build_map()

    def __call__(self, optimizer, scheduler_type, scheduler_setting, scheduler_state=None, **other_params):
        scheduler = self.init_scheduler(
            scheduler_type, optimizer, group_num=1, **scheduler_setting)
        scheduler = self.load_state(scheduler=scheduler, scheduler_state=scheduler_state)
        return scheduler

    def _build_map(self):
        self.name2scheduler = {
            'constant': keep_constant,
            'power': reduce_with_power,
            'plateau': reduce_on_plateau,
            'improve': ReduceByImprove,
            'cosine': reduce_cosine_annealing,
            'cosine_restart': reduce_cosine_annealing_warm_restart
            }
        self.name2default_setting = {
            'constant': {},
            'power': {'sleep_epochs': 25, 'reduce_factor': 0.5},
            'plateau': {'mode': 'min', 'patience': 3, 'reduce_factor': 0.5, 'cooldown': 3, 'min_lr': 1e-5},
            'improve': {'factor': 0.5, 'patience': 3},
            'cosine': {'t_max': 4, 'min_lr': 1e-5, 'last_epoch': -1},
            'cosine_restart': {'t0': 20, 't_mult': 2, 'min_lr': 1e-5}
        }

    def init_scheduler(self, scheduler_type, optimizer, group_num=1, **scheduler_setting):
        write_log(content='Using scheduler: {}'.format(scheduler_type), level=self.log_level, logger=self.logger)
        default_setting = self.name2default_setting[scheduler_type]
        if group_num == 1:
            default_setting.update(scheduler_setting)
            scheduler = self.name2scheduler[scheduler_type](
                **{**{'optimizer': optimizer}, **default_setting})
        else:
            raise NotImplementedError('cannot support more than 1 params group')
        return scheduler

    def load_state(self, scheduler, scheduler_state=None):
        if not scheduler_state:
            write_log(content='No exist scheduler state', level=self.log_level, logger=self.logger)
        elif isinstance(scheduler_state, (dict, str, tuple, list)):
            if isinstance(scheduler_state, str):
                already_optimizer_state = torch.load(scheduler_state, map_location=lambda storage, loc: storage)
                write_log(content='Loading exist scheduler state from {}'.format(scheduler_state),
                          level=self.log_level, logger=self.logger)
            elif isinstance(scheduler_state, (tuple, list)):
                already_optimizer_state = torch.load(
                    scheduler_state[0], map_location=lambda storage, loc: storage)[scheduler_state[1]]
                write_log(content='Loading exist scheduler state from {}'.format('.'.join(scheduler_state)),
                          level=self.log_level, logger=self.logger)
            else:
                already_optimizer_state = scheduler_state
                write_log(content='Loading exist scheduler state', level=self.log_level, logger=self.logger)
            scheduler.load_state_dict(already_optimizer_state)
        else:
            raise ValueError('unknown scheduler state')
        return scheduler


def keep_constant(optimizer, **params):
    def lr_constant(epoch):
        return 1.

    scheduler = lrs.LambdaLR(
        optimizer,
        lr_lambda=lr_constant,
        last_epoch=-1)
    return scheduler


def reduce_with_power(optimizer, sleep_epochs, reduce_factor, **params):
    def lr_power_epoch(epoch):
        if epoch >= sleep_epochs:
            factor = math.pow(0.5, (epoch - sleep_epochs + 1) * reduce_factor)
        else:
            factor = 1.
        return factor

    scheduler = lrs.LambdaLR(
        optimizer,
        lr_lambda=lr_power_epoch,
        last_epoch=-1)
    return scheduler


def reduce_on_plateau(optimizer, mode, patience, reduce_factor, cooldown, min_lr, **params):
    scheduler = lrs.ReduceLROnPlateau(
        optimizer, mode=mode, factor=reduce_factor, patience=patience, verbose=False, threshold=0.0001,
        threshold_mode='rel', cooldown=cooldown, min_lr=min_lr, eps=1e-08)
    return scheduler


def reduce_cosine_annealing(optimizer, t_max, min_lr, last_epoch, **other_params):
    scheduler = lrs.CosineAnnealingLR(optimizer=optimizer, T_max=t_max, eta_min=min_lr, last_epoch=last_epoch)
    return scheduler


def reduce_cosine_annealing_warm_restart(optimizer, t0, t_mult, min_lr, **other_params):
    scheduler = lrs.CosineAnnealingWarmRestarts(optimizer, T_0=t0, T_mult=t_mult, eta_min=min_lr, last_epoch=-1)
    return scheduler


class ReduceByImprove(object):
    def __init__(self, optimizer, factor=0.1, patience=10, **params):
        super(ReduceByImprove, self).__init__()
        assert factor < 1.0, 'Factor should be < 1.0.'
        self.factor = factor
        # Attach optimizer
        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(type(optimizer).__name__))
        self.optimizer = optimizer
        self.patience = patience

    def step(self, no_improve):
        if no_improve >= self.patience:
            self._reduce_lr()

    def _reduce_lr(self):
        for i, param_group in enumerate(self.optimizer.param_groups):
            old_lr = float(param_group['lr'])
            new_lr = old_lr * self.factor
            param_group['lr'] = new_lr

    def state_dict(self):
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)


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
