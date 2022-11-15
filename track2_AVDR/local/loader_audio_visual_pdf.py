#!/usr/bin/env python
# _*_ coding: UTF-8 _*_
import torch
from torch.utils.data import DataLoader
import numpy as np

from tool.data_io import safe_load
from loader.sampler_dynamic_distributed import DynamicBatchSampler, DistributedSamplerWrapper
from loader.loader_truncation_dynamic_distributed import BaseTruncationDataset, PaddedBatch

# from prefetch_generator import BackgroundGenerator
#
#
# class DataLoaderX(DataLoader):
#     def __iter__(self):
#         return BackgroundGenerator(super().__iter__())

class AudioVisualPdfTruncationDataset(BaseTruncationDataset):
    def __init__(self, annotate, repeat_num, max_duration, hop_duration, items, duration_factor=None, deleted_keys=None,
                 key_output=False, logger=None):
        super(AudioVisualPdfTruncationDataset, self).__init__(
            annotate=annotate, repeat_num=repeat_num, max_duration=max_duration, hop_duration=hop_duration, items=items,
            duration_factor=duration_factor, deleted_keys=deleted_keys, key_output=key_output, logger=logger)

    def _get_value(self, key, item, begin, duration, item2file):
        if item in ['far_head', 'far_lip', 'middle_head', 'middle_lip', 'far_wave', 'middle_wave', 'near_wave', 'far_gss_wave']:
            item2sample_rate = {'fear_head': 25, 'far_lip': 25, 'middle_head': 25, 'middle_lip': 25, 
                                'far_gss_wave': 16000, 'far_wave': 16000, 'middle_wave': 16000, 'near_wave': 16000}
            begin_point = int(np.around(begin * item2sample_rate[item]))
            end_point = int(np.around(begin_point + duration * item2sample_rate[item]))
            item_data = safe_load(file=item2file[item])[begin_point:end_point]
        elif item in ['key']:
            item_data = key
        elif item.split('_')[-3] in ['classification', 'posteriori']:
            *distance, item_type, grain, interval = item.split('_')
            distance = '_'.join(distance)
            grain = int(grain)
            interval = float(interval)
            pdf_information = safe_load(file=item2file['{}_pdf'.format(distance)])
            characters = pdf_information['pdf']
            end_timestamp = np.array(pdf_information['stamp']) - begin
            if item_type == 'classification':
                item_data = (torch.ones(int(np.around(duration/interval))) * grain).long()
                begin_timestamp = 0.
                for i in range(len(characters)):
                    if end_timestamp[i] > 0:
                        begin_idx = int(np.around(begin_timestamp / interval))
                        if end_timestamp[i] <= duration:
                            end_idx = int(np.around(end_timestamp[i] / interval))
                            item_data[begin_idx:end_idx] = characters[i]
                            begin_timestamp = end_timestamp[i]
                        else:
                            end_idx = int(np.around(duration / interval))
                            item_data[begin_idx:end_idx] = characters[i]
                            break
            else:
                item_data = torch.zeros(int(np.around(duration/interval)), grain, dtype=torch.float)
                begin_timestamp = 0.
                for i in range(len(characters)):
                    if end_timestamp[i] > 0:
                        begin_idx = int(np.around(begin_timestamp / interval))
                        if end_timestamp[i] <= duration:
                            end_idx = int(np.around(end_timestamp[i] / interval))
                            if characters[i] < grain:
                                item_data[begin_idx:end_idx, characters[i]] = 1.
                            begin_timestamp = end_timestamp[i]
                        else:
                            end_idx = int(np.around(duration / interval))
                            if characters[i] < grain:
                                item_data[begin_idx:end_idx, characters[i]] = 1.
                            break
        else:
            raise NotImplementedError('unknown output')
        return item_data


def get_data_loader(
        annotate, items, batch_size, max_batch_size, target_shape, pad_value, repeat=1, max_duration=100,
        hop_duration=10, duration_factor=None, deleted_keys=None, key_output=False, dynamic=True,
        bucket_length_multiplier=1.1, shuffle=False, drop_last=False, num_workers=4, pin_memory=False, seed=123456,
        epoch=0, logger=None, distributed=False, **other_params):
    # import pdb; pdb.set_trace()
    dataset = AudioVisualPdfTruncationDataset(annotate=annotate, repeat_num=repeat, max_duration=max_duration,
                                           hop_duration=hop_duration, items=items, duration_factor=duration_factor,
                                           deleted_keys=deleted_keys, key_output=key_output, logger=logger)
                                           
    data_sampler = DynamicBatchSampler(lengths_list=dataset.duration, batch_size=batch_size, dynamic=dynamic,
                                       max_batch_size=max_batch_size, epoch=epoch, drop_last=drop_last, logger=logger,
                                       bucket_length_multiplier=bucket_length_multiplier, shuffle=shuffle, seed=seed)

    if distributed:
        data_sampler = DistributedSamplerWrapper(sampler=data_sampler, seed=seed, shuffle=shuffle, drop_last=drop_last)

    collate_fn = PaddedBatch(items=dataset.items, target_shape=target_shape, pad_value=pad_value)

    data_loader = DataLoader(dataset=dataset, collate_fn=collate_fn, num_workers=num_workers, pin_memory=pin_memory,
                             batch_sampler=data_sampler)

    return data_loader, data_sampler
