#!/usr/bin/env python
# _*_ coding: UTF-8 _*_
import time
import torch
import torch.nn as nn
import torch.distributed as dist

import os
import numpy as np
from tqdm import tqdm
import logging
import matplotlib.pyplot as plt

from loader_audio_visual_pdf import get_data_loader
from network.network_index import AVSRWorker
from optimizer.optimizer_optimizer import OptimizerWorker
from optimizer.lr_scheduler import LRSchedulerWorker
from loss.loss_function import LossFunctionWorker
from tool.data_io import safe_load, safe_store

plt.switch_backend('agg')
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


def load_previous_work(exp_dir, logger, args, **other_params):
    if args.begin_epoch == 0:
        write_log(logger=logger, content='No continue', level='info')
        model_params = None
        optimizer_state = None
        scheduler_state = None
        metrics_tensor = None
        lr_tensor = None
        best_crown = None
        crown_no_improve = None
    else:
        write_log(logger=logger, content='Continue from {}'.format(args.begin_epoch - 1), level='info')
        continue_path = os.path.join(exp_dir, 'model', 'epoch{}.tar'.format(args.begin_epoch - 1))
        model_package = safe_load(file=continue_path, ftype='torch')
        model_params = model_package['model_params']
        optimizer_state = model_package['optimizer_state']
        scheduler_state = model_package['scheduler_state']
        metrics_package = safe_load(file=os.path.join(exp_dir, 'metrics.pt'))
        metrics_tensor = metrics_package['metrics'][:, :args.begin_epoch, :]
        lr_tensor = metrics_package['lr'][:args.begin_epoch, :]
        metrics_num = metrics_tensor.shape[-1]
        stage_num = metrics_tensor.shape[0]
        group_num = lr_tensor.shape[-1]
        metrics_tensor = torch.cat([metrics_tensor,
                                   torch.zeros(stage_num, args.epoch_sum - args.begin_epoch, metrics_num)], dim=1)
        lr_tensor = torch.cat([lr_tensor, torch.zeros(args.epoch_sum - args.begin_epoch, group_num)], dim=0)
        best_crown = torch.cat([metrics_package['best_crown'][:args.begin_epoch],
                                torch.zeros(args.epoch_sum - args.begin_epoch)], dim=0)
        crown_no_improve = torch.cat([metrics_package['crown_no_improve'][:args.begin_epoch],
                                      torch.zeros(args.epoch_sum - args.begin_epoch)], dim=0)
    return model_params, optimizer_state, scheduler_state, metrics_tensor, lr_tensor, best_crown, crown_no_improve


# def forward_worker(model, loader_output, input_settings, output_settings, loss_function, args, skip_loss=False,
#                    **other_params):
#     data_input = []
#     label_input = []
#     for data_idx in input_settings['data_idxes']:
#         if data_idx not in input_settings['length_idxes']:
#             data_input.append(loader_output[data_idx].cuda(args.local_rank, non_blocking=True))
#         else:
#             data_input.append(loader_output[data_idx])
#     model_output = model(*data_input)
#     if not skip_loss:
#         for label_idx in input_settings['label_idxes']:
#             if label_idx not in input_settings['length_idxes']:
#                 label_input.append(loader_output[label_idx].cuda(args.local_rank, non_blocking=True))
#             else:
#                 label_input.append(loader_output[label_idx])
#         loss, metrics = loss_function(*[model_output[i] for i in output_settings['data_idxes']], *label_input)
#         return loss, metrics
#     return model_output


def forward_worker(model, loader_output, model_input_setting, loss_input_setting, loss_function, args, skip_loss=False,
                   **other_params):    
    model_input = {}
    loss_input = {}
    for k, v in model_input_setting['items'].items():
        if k in model_input_setting['gpu_items']:
            model_input[k] = loader_output[v].cuda(args.local_rank, non_blocking=True)
        else:
            model_input[k] = loader_output[v]
    model_output = model(**model_input)
    
    if not skip_loss:
        for k, v in loss_input_setting['items_from_loader'].items():
            if k in loss_input_setting['gpu_items']:
                loss_input[k] = loader_output[v].cuda(args.local_rank, non_blocking=True)
            else:
                loss_input[k] = loader_output[v]
        
        for k, v in loss_input_setting['items_from_model'].items():
            if k in loss_input_setting['gpu_items']:
                loss_input[k] = model_output[v].cuda(args.local_rank, non_blocking=True)
            else:
                loss_input[k] = model_output[v]
        loss, metrics = loss_function(**loss_input)
        return loss, metrics
    return model_output
        

def train_one_epoch(epoch_idx, model, optimizer, data_loader, data_sampler, loss_function, model_input_setting, 
                    loss_input_setting, gradient, logger, args, **other_params):

    data_sampler.set_epoch(epoch_idx)

    model.train()

    # time
    epoch_begin_time = time.time()
    batch_time = time.time()
    io_time = time.time()
    batch_time_interval = 0.0
    io_time_interval = 0.0

    # metrics
    epoch_total_metrics = torch.zeros(len(args.metrics))

    # batch_count
    batch_num = len(data_loader)
    valid_batch_num = 0
    batch_bar = None
    if args.rank == 0:
        batch_bar = tqdm(total=batch_num, leave=False, desc='Rank {} Train Batch'.format(args.rank))
    optimizer.zero_grad()
    write_log(content='{}Training: {} epoch{}'.format('-' * 12, epoch_idx, '-' * 12), logger=logger, level='debug')
    for batch_idx, batch_data in enumerate(data_loader):
        io_time_interval += (time.time() - io_time)
        try:
            batch_loss, batch_metrics = forward_worker(
                model=model, loader_output=batch_data, model_input_setting=model_input_setting, loss_input_setting=loss_input_setting,
                loss_function=loss_function, args=args, skip_loss=False)
        except RuntimeError as error:
            if 'out of memory' in str(error):
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
                write_log(content='NO.{} batch pass during training, ran out of memory in froward'.format(batch_idx),
                          logger=logger, level='debug')
            else:
                raise error
        else:
            try:
                batch_loss.backward()
            except RuntimeError as error:
                if 'out of memory' in str(error):
                    if hasattr(torch.cuda, 'empty_cache'):
                        torch.cuda.empty_cache()
                    optimizer.zero_grad()
                    write_log(content='NO.{} batch pass during training, ran out of memory in backward'.format(
                        batch_idx), logger=logger, level='debug')
                else:
                    raise error
            else:
                valid_batch_num = valid_batch_num + 1
                if not gradient:
                    pass
                else:
                    for grad_process in gradient:
                        if grad_process == 'grad_clip':
                            if isinstance(gradient['grad_clip'], float):
                                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient['grad_clip'])
                            elif isinstance(gradient['grad_clip'], list) and gradient['grad_clip'][0] == 'norm':
                                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient['grad_clip'][1])
                            elif isinstance(gradient['grad_clip'], list) and gradient['grad_clip'][0] == 'value':
                                torch.nn.utils.clip_grad_value_(model.parameters(), gradient['grad_clip'][1])
                            else:
                                raise ValueError('unknown grad_clip')
                        else:
                            raise NotImplementedError('gradient processing only support grad_clip, '
                                                      'but got {}'.format(grad_process))
                optimizer.step()
                optimizer.zero_grad()
                epoch_total_metrics += batch_metrics
                batch_time_interval += (time.time() - batch_time)
                batch_time = time.time()
                if batch_idx % args.print_freq == 0 or (batch_idx + 1) == batch_num:
                    average_metrics_str = '/'.join([*map(str, (epoch_total_metrics / valid_batch_num).tolist())])
                    current_metrics_str = '/'.join([*map(str, batch_metrics.tolist())])
                    io_speed = 1000 * io_time_interval / (batch_idx + 1)
                    batch_speed = 1000 * batch_time_interval / valid_batch_num
                    cur_time = (time.time() - epoch_begin_time) / 60.
                    total_time = cur_time * batch_num * 1.0 / valid_batch_num
                    log_content = 'Epoch {}|Batch {}|Metrics {}|Average {}|Current {}|Speed IO/Batch {:.2f}/{:.2f} ms/'\
                                  'batch|Time Consume/Residue {:.2f}/{:.2f} min'.format(
                        epoch_idx, batch_idx, '/'.join(args.metrics), average_metrics_str, current_metrics_str,
                        io_speed, batch_speed, cur_time, total_time - cur_time)
                    write_log(content=log_content, logger=logger, level='debug')
        if args.rank == 0:
            batch_bar.update(1)
        io_time = time.time()
    if args.rank == 0:
        batch_bar.close()
    epoch_duration = time.time() - epoch_begin_time
    epoch_metrics = epoch_total_metrics / valid_batch_num
    return epoch_duration, epoch_metrics


def eval_one_epoch(epoch_idx, model, data_loader, loss_function, model_input_setting, loss_input_setting, phase, logger,
                   args, **other_params):
    eval_start = time.time()
    valid_batch_num = 0
    epoch_total_metrics = torch.zeros(len(args.metrics))
    batch_bar = None
    if args.rank == 0:
        batch_bar = tqdm(total=len(data_loader), leave=False, desc='Rank {} {} Batch'.format(args.rank, phase.title()))
    model.eval()
    write_log(content='{}{}ing: {} epoch{}'.format('-' * 12, phase.title(), epoch_idx, '-' * 12), logger=logger,
              level='debug')
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(data_loader):
            try:
                batch_loss, batch_metrics = forward_worker(model=model, loader_output=batch_data,
                                                           model_input_setting=model_input_setting, 
                                                           loss_input_setting=loss_input_setting,
                                                           loss_function=loss_function, args=args, skip_loss=False)
            except RuntimeError as error:
                if 'out of memory' in str(error):
                    if hasattr(torch.cuda, 'empty_cache'):
                        torch.cuda.empty_cache()
                    write_log(content='NO.{} batch pass during {}ing, run out of memory in froward'.format(
                        batch_idx, phase.lower()), logger=logger, level='debug')
                else:
                    raise error
            else:
                epoch_total_metrics += batch_metrics
                valid_batch_num = valid_batch_num + 1
            dist.barrier()
            if args.rank == 0:
                batch_bar.update(1)
    if args.rank == 0:
        batch_bar.close()
    epoch_duration = time.time() - eval_start
    epoch_metrics = epoch_total_metrics / valid_batch_num
    return epoch_duration, epoch_metrics


def reduce_tensor(tensor: torch.Tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt


def summary_one_epoch(exp_dir, epoch_idx, epoch_metrics, epoch_time, model, optimizer, scheduler, logger, args,
                      metrics_tensor=None, lr=None, crown_no_improve=None, best_crown=None, **other_params):

    # record
    metric_num = len(args.metrics)
    group_num = len(optimizer.param_groups)
    stage_num = len(args.stages)

    # first epoch, init
    if epoch_idx == 0:
        metrics_tensor = torch.zeros(stage_num, args.epoch_sum, metric_num)
        lr = torch.zeros(args.epoch_sum, group_num)
        crown_no_improve = torch.zeros(args.epoch_sum)
        best_crown = torch.zeros(args.epoch_sum)

    metrics_tensor[:, epoch_idx, :] = epoch_metrics
    lr[epoch_idx] = torch.tensor([optimizer.param_groups[i]['lr'] for i in range(group_num)])

    # comparison
    if epoch_idx == 0:
        best_token = True
        current_crown_no_improve = False
    elif epoch_idx > 0 and args.comparison == 'min':
        best_token = metrics_tensor[args.stage_id, epoch_idx, args.crown_id] < best_crown[epoch_idx - 1]
        current_crown_no_improve = metrics_tensor[args.stage_id, epoch_idx, args.crown_id] >= \
                                   metrics_tensor[args.stage_id, epoch_idx - 1, args.crown_id]
    elif epoch_idx > 0 and args.comparison == 'max':
        best_token = bool(metrics_tensor[args.stage_id, epoch_idx, args.crown_id] > best_crown[epoch_idx - 1])
        current_crown_no_improve = metrics_tensor[args.stage_id, epoch_idx, args.crown_id] <= \
                                   metrics_tensor[args.stage_id, epoch_idx - 1, args.crown_id]
    else:
        raise ValueError('unknown comparison')
    if epoch_idx > 0:
        crown_no_improve[epoch_idx] = float(current_crown_no_improve) * (crown_no_improve[epoch_idx - 1] + 1)
    if best_token:
        best_crown[epoch_idx] = metrics_tensor[args.stage_id, epoch_idx, args.crown_id]
    else:
        best_crown[epoch_idx] = best_crown[epoch_idx - 1]

    # save metrics
    if args.rank == 0:
        # checkpoint
        if (epoch_idx + 1) % args.save_point == 0:
            write_log(logger=logger,
                      content='Epoch {}/{}|Saving {} result'.format(epoch_idx, args.epoch_sum, epoch_idx),
                      level='info')
            safe_store(file=os.path.join(exp_dir, 'model', 'epoch{}.tar'.format(epoch_idx)),
                       data={'model_params': model.module.state_dict(), 'optimizer_state': optimizer.state_dict(),
                             'scheduler_state': scheduler.state_dict()}, mode='cover', ftype='torch')

        # store best
        if best_token:
            safe_store(file=os.path.join(exp_dir, 'best.tar'),
                       data={'model_params': model.module.state_dict(), 'optimizer_state': optimizer.state_dict(),
                             'scheduler_state': scheduler.state_dict()}, mode='cover', ftype='torch')
            write_log(content='Epoch {}/{}|According {} {}, Saving {} result to best'.format(
                epoch_idx, args.epoch_sum, args.stages[args.stage_id], args.metrics[args.crown_id], epoch_idx),
                logger=logger, level='info')

        safe_store(
            file=os.path.join(exp_dir, 'metrics.pt'),
            data={'metrics': metrics_tensor[:, :epoch_idx + 1, :], 'lr': lr[:epoch_idx + 1, :],
                  'best_crown': best_crown[:epoch_idx + 1], 'crown_no_improve': crown_no_improve[:epoch_idx + 1]},
            mode='cover', ftype='torch')

        # plot
        for metric_idx in range(metric_num):
            plot_data = {args.stages[stage_idx].lower(): metrics_tensor[stage_idx, :epoch_idx + 1, metric_idx]
                         for stage_idx in range(stage_num)}
            plot(figure_path=os.path.join(exp_dir, '{}.png'.format(args.metrics[metric_idx])),
                 y_label=args.metrics[metric_idx], x_label='epoch', **plot_data)
        plot(figure_path=os.path.join(exp_dir, 'lr.png'), y_label='lr', x_label='epoch',
             **{'group_{}'.format(group_idx): lr[:epoch_idx + 1, group_idx] for group_idx in range(group_num)})

        # summary current epoch
        log_content = 'Epoch {}/{}|Speed {:.2f} min/epoch|Residue Time {:.2f} min|Lr {}|Metrics {}|{}'.format(
            epoch_idx, args.epoch_sum, epoch_time/60., epoch_time*(args.epoch_sum-epoch_idx-1.)/60.,
            '/'.join([*map(str, lr[epoch_idx].tolist())]), '/'.join(args.metrics),
            '|'.join(['{} {}'.format(args.stages[stage_idx].title(),
                                     '/'.join([*map(str, metrics_tensor[stage_idx, epoch_idx, :].tolist())]))
                      for stage_idx in range(stage_num)]))
        write_log(content=log_content, logger=logger, level='info')

        # start next epoch
        if epoch_idx == args.epoch_sum - 1:
            write_log(content='{}End{}'.format('-' * 12, '-' * 12), logger=logger, level='info')
            log_content = 'End|previous_{}_{}: {}|best_{}_{}: {}|{}_{}_no_improve: {}'.format(
                args.stages[args.stage_id], args.metrics[args.crown_id],
                metrics_tensor[args.stage_id][epoch_idx][args.crown_id], args.stages[args.stage_id],
                args.metrics[args.crown_id], best_crown[epoch_idx], args.stages[args.stage_id],
                args.metrics[args.crown_id], crown_no_improve[epoch_idx])
        else:
            write_log(content='{}Epoch: {}{}'.format('-' * 12, epoch_idx+1, '-' * 12), logger=logger, level='info')
            log_content = 'Epoch {}/{}|previous_{}_{}: {}|best_{}_{}: {}|{}_{}_no_improve: {}'.format(
                epoch_idx + 1, args.epoch_sum, args.stages[args.stage_id], args.metrics[args.crown_id],
                metrics_tensor[args.stage_id][epoch_idx][args.crown_id], args.stages[args.stage_id],
                args.metrics[args.crown_id], best_crown[epoch_idx],args.stages[args.stage_id],
                args.metrics[args.crown_id], crown_no_improve[epoch_idx])
        write_log(content=log_content, logger=logger, level='info')

    return lr, best_crown, crown_no_improve, metrics_tensor


def plot(figure_path, y_label, x_label, **data):
    figure = plt.figure()
    os.makedirs(os.path.split(figure_path)[0], exist_ok=True)
    for k, x in data.items():
        if isinstance(x, (list, np.ndarray, torch.Tensor)):
            plt.plot(list(range(len(x))), x, label=k)
        elif isinstance(x, tuple):
            plt.plot(x[0], x[1], label=k)
        else:
            raise ValueError('unknown data value')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend(loc='upper right')
    plt.close('all')
    figure.savefig(figure_path, dpi=330, bbox_inches='tight')
    return None


def main_train(exp_dir, hparams, args, logger, **other_params):

    # get data_loaders & data_samplers
    data_loaders = []
    data_samplers = []
    for stage_idx in range(len(args.stages)):
        data_loader, data_sampler = get_data_loader(
            **hparams['data'][args.stages[stage_idx]], seed=args.random_seed, epoch=args.begin_epoch,
            logger=logger, distributed=args.distributed)
        data_loaders.append(data_loader)
        data_samplers.append(data_sampler)

    # init record
    (model_params, optimizer_state, scheduler_state, metrics_array, lr_array, best_crown,
     crown_no_improve) = load_previous_work(exp_dir=exp_dir, logger=logger, args=args)

    # init model
    model_worker = AVSRWorker(log_type='warning', logger=logger)
    model_hparams = hparams['model']
    # if begin_epoch != 0, use the params of begin_epoch - 1 to init network
    if model_params:
        model_hparams = {**model_hparams, 'pretrained_num': 1, 'pretrained_model': model_params, 'replace_keys': None,
                         'unload_keys': None}
    model = model_worker(**model_hparams)
    # copy model to gpu
    model = model.cuda(args.local_rank)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank)

    # init optimizer
    optimizer_worker = OptimizerWorker(log_type='info', logger=logger)
    optimizer = optimizer_worker(model_params=filter(lambda p: p.requires_grad, model.parameters()),
                                 optimizer_state=optimizer_state, **hparams['optimizer'])

    # init lr_scheduler
    scheduler_worker = LRSchedulerWorker(log_type='info', logger=logger)
    scheduler = scheduler_worker(optimizer=optimizer, scheduler_state=scheduler_state, **hparams['scheduler'])

    # init loss
    loss_function_worker = LossFunctionWorker(log_type='info', logger=logger)
    loss_function = loss_function_worker(**hparams['loss']).cuda(args.local_rank)

    # start
    write_log(logger=logger, content='{}Epoch: {}{}'.format('-'*12, args.begin_epoch, '-'*12), level='info')
    if args.begin_epoch >= 1:
        log_content = 'Epoch {}/{}|previous_{}_{}: {}|best_{}_{}: {}|{}_{}_no_improve: {}'.format(
            args.begin_epoch, args.epoch_sum, args.stages[args.stage_id], args.metrics[args.crown_id],
            metrics_array[args.stage_id][args.begin_epoch - 1][args.crown_id], args.stages[args.stage_id],
            args.metrics[args.crown_id], best_crown[args.begin_epoch - 1], args.stages[args.stage_id],
            args.metrics[args.crown_id], crown_no_improve[args.begin_epoch - 1])
    else:
        log_content = 'Epoch {}/{}|previous_{}_{}: {}|best_{}_{}: {}|{}_{}_no_improve: {}'.format(
            args.begin_epoch, args.epoch_sum, args.stages[args.stage_id], args.metrics[args.crown_id],
            None, args.stages[args.stage_id], args.metrics[args.crown_id], None, args.stages[args.stage_id],
            args.metrics[args.crown_id], None)
    write_log(logger=logger, content=log_content, level='info')
    # processing bar
    epoch_bar = None
    if args.rank == 0:
        epoch_bar = tqdm(total=args.epoch_sum - args.begin_epoch, leave=True, desc='Rank {} Epoch'.format(args.rank))
    for epoch_idx in range(args.begin_epoch, args.epoch_sum):
        epoch_metrics = []
        epoch_times = 0.
        for stage_idx in range(len(args.stages)):
            if args.stages[stage_idx] == 'train':
                fixed_model_keys = []
                if 'fixed_keys' in model_hparams:
                    fixed_model_keys = model_hparams['fixed_keys']
                epoch_time, epoch_metric = train_one_epoch(
                    epoch_idx=epoch_idx, model=model, optimizer=optimizer, data_loader=data_loaders[stage_idx],
                    data_sampler=data_samplers[stage_idx], loss_function=loss_function, logger=logger, args=args,
                    model_input_setting=hparams['data']['model_input'], loss_input_setting=hparams['data']['loss_input'],
                    gradient=hparams['gradient'], fixed_model_keys=fixed_model_keys)
            elif args.stages[stage_idx] in ['dev', 'eval', 'addition', 'eval_addition']:
                epoch_time, epoch_metric = eval_one_epoch(
                    epoch_idx=epoch_idx, phase=args.stages[stage_idx], model=model, data_loader=data_loaders[stage_idx],
                    loss_function=loss_function, logger=logger, args=args, model_input_setting=hparams['data']['model_input'],
                    loss_input_setting=hparams['data']['loss_input'])
            else:
                raise NotImplementedError('unknown stage')
            # print('11111')
            epoch_metric = reduce_tensor(epoch_metric.cuda(args.local_rank)).cpu()
            # print('22222')
            epoch_metrics.append(epoch_metric)
            epoch_times += epoch_time

        # Use a barrier() to make sure that process loads the model after process 0 saves it.
        dist.barrier()

        # summary
        lr_array, best_crown, crown_no_improve, metrics_array = summary_one_epoch(
            exp_dir=exp_dir, epoch_idx=epoch_idx, epoch_metrics=torch.stack(epoch_metrics, dim=0),
            epoch_time=epoch_times, model=model, optimizer=optimizer, scheduler=scheduler, logger=logger, args=args,
            metrics_tensor=metrics_array, lr=lr_array, crown_no_improve=crown_no_improve, best_crown=best_crown)

        # lr_scheduler
        if epoch_idx != args.epoch_sum - 1:
            if hparams['scheduler']['scheduler_type'] == 'plateau':
                scheduler.step(metrics_array[args.stage_id][epoch_idx][args.crown_id])
            elif hparams['scheduler']['scheduler_type'] in ['constant', 'power', 'cosine', 'cosine_restart']:
                scheduler.step()
            elif hparams['scheduler']['scheduler_type'] == 'improve':
                scheduler.step(crown_no_improve[epoch_idx])
            else:
                raise ValueError('unknown scheduler_type')
        if args.rank == 0:
            epoch_bar.update(1)
    if args.rank == 0:
        epoch_bar.close()
    return None
        