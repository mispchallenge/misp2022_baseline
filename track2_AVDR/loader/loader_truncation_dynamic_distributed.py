#!/usr/bin/env python
# _*_ coding: UTF-8 _*_
import torch
import torch.nn.functional as nf
from torch.utils.data import Dataset

import json
import codecs
import logging
import numpy as np


class BaseTruncationDataset(Dataset):
    def __init__(self, annotate, repeat_num, max_duration, hop_duration, items, duration_factor=None, deleted_keys=None,
                 key_output=False, logger=None, **other_params):
        super(BaseTruncationDataset, self).__init__()
        deleted_keys = [] if deleted_keys is None else deleted_keys
        if isinstance(annotate, str):
            annotate = [annotate]
        elif isinstance(annotate, list) and all(isinstance(a, str) for a in annotate):
            annotate = annotate
        else:
            raise ValueError('unknown annotate type: {}.'.format(annotate))
        annotate_num = len(annotate)
        repeat_num = self.expend_params(value=repeat_num, length=annotate_num)
        max_duration = self.expend_params(value=max_duration, length=annotate_num)
        hop_duration = self.expend_params(value=hop_duration, length=annotate_num)
        duration_factor = self.expend_params(value=duration_factor, length=annotate_num)

        self.items = ['key', *items] if items[0] != 'key' and key_output else items
        self.keys = []
        self.duration = []
        self.begin = []
        self.key2path = {}
        for annotate_id in range(annotate_num):
            write_log(content='Load {} from {}, max_duration is {} s, hop_duration is {} s, repeat {} times.'.format(
                ','.join(items), annotate[annotate_id], max_duration[annotate_id], hop_duration[annotate_id],
                repeat_num[annotate_id]), level='info', logger=logger)
            with codecs.open(annotate[annotate_id], 'r') as handle:
                data_dic = json.load(handle)
            data_keys = data_dic['keys']
            data_duration = data_dic['duration']
            data_key2path = data_dic['key2path']
            # del sample
            for key_idx in range(len(data_keys)-1, -1, -1):
                if data_keys[key_idx] in deleted_keys:
                    data_key2path.pop(data_keys[key_idx])
                    data_keys.pop(key_idx)
                    data_duration.pop(key_idx)
            self.key2path.update(data_key2path)
            split_keys, split_duration, split_begin = self.cut_off(
                keys=data_keys, duration=data_duration, max_duration=max_duration[annotate_id],
                hop_duration=hop_duration[annotate_id], duration_factor=duration_factor[annotate_id])
            for _ in range(repeat_num[annotate_id]):
                self.keys = self.keys + split_keys
                self.duration = self.duration + split_duration
                self.begin = self.begin + split_begin
            del data_dic
        write_log(content='Delete samples: {}'.format(deleted_keys), level='info', logger=logger)
        write_log(content='All duration is {} h'.format(np.sum(self.duration) / 3600.), level='info', logger=logger)

    def __getitem__(self, index):
        main_key = self.keys[index]
        item2paths = self.key2path[main_key]
        value_lst = []
        for item in self.items:
            value = self._get_value(key=main_key, item=item, begin=self.begin[index], duration=self.duration[index],
                                    item2file=item2paths)

            if isinstance(value, list):
                value_lst.extend(value)
            else:
                value_lst.append(value)
            del value
        return value_lst

    def _get_value(self, key, item, begin, duration, item2file):
        return key

    def __len__(self):
        return len(self.keys)

    @staticmethod
    def cut_off(keys, duration, max_duration, hop_duration, duration_factor=None):
        split_keys = []
        split_duration = []
        split_begin = []
        for idx in range(len(keys)):
            idx_key = keys[idx]
            idx_duration = duration[idx]
            idx_begin = 0.
            while idx_duration > max_duration:
                split_keys.append(idx_key)
                split_duration.append(max_duration)
                split_begin.append(idx_begin)
                idx_duration = idx_duration - hop_duration
                idx_begin = idx_begin + hop_duration
            split_keys.append(idx_key)
            if duration_factor:
                final_duration = idx_duration - idx_duration % duration_factor
            else:
                final_duration = idx_duration
            split_duration.append(final_duration)
            split_begin.append(idx_begin)
        return split_keys, split_duration, split_begin

    @staticmethod
    def expend_params(value, length):
        if isinstance(value, list):
            if len(value) == length:
                return value
            else:
                raise ValueError('list have unmatched length: {}'.format(value))
        else:
            return [value for _ in range(length)]


class PaddedBatch(object):
    def __init__(self, items, target_shape, pad_value):
        self.items = items
        self.target_shape = target_shape
        self.pad_value = pad_value

    def __call__(self, dataset_outputs):
        pad_idx = 0
        batched_value = []
        for item, *batch_values in zip(self.items, *dataset_outputs):
            if item in ['key']:
                batched_value.append(batch_values)
            else:
                batched_value.extend(
                    self._batch_pad_right(target_shape=self.target_shape[pad_idx], pad_value=self.pad_value[pad_idx],
                                          tensors=batch_values))
                pad_idx = pad_idx + 1
        return batched_value

    def _batch_pad_right(self, target_shape, pad_value, tensors):
        if not len(tensors):
            raise IndexError("Tensors list must not be empty")

        if len(tensors) > 1 and not(any([tensors[i].ndim == tensors[0].ndim for i in range(1, len(tensors))])):
            raise IndexError("All tensors must have same number of dimensions")

        shape_items = torch.zeros(len(tensors)+1, tensors[0].ndim, dtype=torch.long)
        shape_items[-1, :] = torch.tensor(target_shape)
        for x_idx in range(len(tensors)):
            shape_items[x_idx] = torch.tensor(tensors[x_idx].shape)
        target_shape = shape_items.max(dim=0).values.tolist()
        length = shape_items[:-1, 0]

        batched = []
        for t in tensors:
            batched.append(self.pad_right_to(tensor=t, target_shape=target_shape, value=pad_value))
        batched = torch.stack(batched)
        return batched, length
    # def _batch_pad_right(self, target_shape, pad_value, tensors):
    #     if not len(tensors):
    #         raise IndexError("Tensors list must not be empty")

    #     if len(tensors) == 1:
    #         # if there is only one tensor in the batch we simply unsqueeze it.
    #         return tensors[0].unsqueeze(0), torch.tensor([tensors[0].shape[0]])

    #     if not(any([tensors[i].ndim == tensors[0].ndim for i in range(1, len(tensors))])):
    #         raise IndexError("All tensors must have same number of dimensions")

    #     length = []
    #     for x in tensors:
    #         length.append(x.shape[0])
    #     max_length = max(length)
    #     length = torch.tensor(length)

    #     batched = []
    #     target_shape = [max_length, *target_shape]
    #     for t in tensors:
    #         batched.append(self.pad_right_to(tensor=t, target_shape=target_shape, value=pad_value))
    #     batched = torch.stack(batched)
    #     return batched, length

    @staticmethod
    def pad_right_to(tensor, target_shape, mode="constant", value=0):
        """
        This function takes a torch tensor of arbitrary shape and pads it to target
        shape by appending values on the right.
        Parameters
        ----------
        tensor : input torch tensor
            Input tensor whose dimension we need to pad.
        target_shape:
            Target shape we want for the target tensor its len must be equal to tensor.ndim
        mode : str
            Pad mode, please refer to torch.nn.functional.pad documentation.
        value : float
            Pad value, please refer to torch.nn.functional.pad documentation.
        Returns
        -------
        tensor : torch.Tensor
            Padded tensor.
        valid_vals : list
            List containing proportion for each dimension of original, non-padded values.
        """
        assert len(target_shape) == tensor.ndim, 'target_shape is {}, but tensor shape is {}'.format(target_shape,
                                                                                                     tensor.shape)
        pads = []  # this contains the abs length of the padding for each dimension.
        i = len(target_shape) - 1  # iterating over target_shape ndims
        while i >= 0:
            assert (target_shape[i] >= tensor.shape[i]), 'Target shape must be >= original shape for every dim'
            pads.extend([0, target_shape[i] - tensor.shape[i]])
            i -= 1
        tensor = nf.pad(tensor, pads, mode=mode, value=value)
        return tensor


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


# class AudioVisualTruncationDataset(BaseTruncationDataset):
#     def __init__(self, annotate, repeat_num, max_duration, hop_duration, items, duration_factor=None, deleted_keys=None,
#                  key_output=False, logger=None):
#         super(AudioVisualTruncationDataset, self).__init__(
#             annotate=annotate, repeat_num=repeat_num, max_duration=max_duration, hop_duration=hop_duration, items=items,
#             duration_factor=duration_factor, deleted_keys=deleted_keys, key_output=key_output, logger=logger)
#
#     def _get_value(self, key, item, begin, duration, item2file):
#         if item in ['lip_frames', 'clean_wave', 'mixture_wave', 'irm']:
#             item2sample_rate = {'lip_frames': 25, 'clean_wave': 16000, 'mixture_wave': 16000, 'irm': 100}
#             begin_point = int(np.around(begin * item2sample_rate[item]))
#             end_point = int(np.around(begin_point + duration * item2sample_rate[item]))
#             if item in ['irm']:
#                 item_data = safe_load(file=item2file[item])[begin_point: end_point].float()
#             else:
#                 item_data = safe_load(file=item2file[item])[begin_point:end_point]
#         elif item in ['key']:
#             item_data = key
#         elif item.split('_')[0] in ['classification', 'posterior']:
#             item_type, language, grain, interval = item.split('_')
#             grain = int(grain)
#             interval = float(interval)
#             fa_information = safe_load(file=item2file['fa'])
#             if language == 'mandarin':
#                 characters = fa_information['CD-phone']
#                 end_timestamp = np.array(fa_information['end_frame_40ms'])*0.04 - begin
#             elif language == 'english':
#                 if grain in [39, 13, 10]:
#                     characters = fa_information['phone_39']
#                     end_timestamp = np.array(fa_information['phone_39_end_time']) - begin
#                 elif grain in [2704, 2008, 1256, 448]:
#                     characters = fa_information['senone_{}'.format(grain)]
#                     end_timestamp = np.array(fa_information['senone_{}_end_time'.format(grain)]) - begin
#                 else:
#                     raise ValueError('unknown grain {} for english'.format(grain))
#             else:
#                 raise NotImplementedError('unknown language: {}'.format(language))
#             if item_type == 'classification':
#                 item_data = (torch.ones(int(np.around(duration/interval))) * grain).long()
#                 begin_timestamp = 0.
#                 for i in range(len(characters)):
#                     if end_timestamp[i] > 0:
#                         begin_idx = int(np.around(begin_timestamp / interval))
#                         if end_timestamp[i] <= duration:
#                             end_idx = int(np.around(end_timestamp[i] / interval))
#                             item_data[begin_idx:end_idx] = self.map_character_token(characters[i], grain, language)
#                             begin_timestamp = end_timestamp[i]
#                         else:
#                             end_idx = int(np.around(duration / interval))
#                             item_data[begin_idx:end_idx] = self.map_character_token(characters[i], grain, language)
#                             break
#             else:
#                 item_data = torch.zeros(int(np.around(duration/interval)), grain, dtype=torch.float)
#                 begin_timestamp = 0.
#                 for i in range(len(characters)):
#                     if end_timestamp[i] > 0:
#                         begin_idx = int(np.around(begin_timestamp / interval))
#                         token = self.map_character_token(characters[i], grain, language)
#                         if end_timestamp[i] <= duration:
#                             end_idx = int(np.around(end_timestamp[i] / interval))
#                             if token < grain:
#                                 item_data[begin_idx:end_idx, token] = 1.
#                             begin_timestamp = end_timestamp[i]
#                         else:
#                             end_idx = int(np.around(duration / interval))
#                             if token < grain:
#                                 item_data[begin_idx:end_idx, token] = 1.
#                             break
#         else:
#             raise NotImplementedError('unknown output')
#         return item_data
#
#     @staticmethod
#     def map_character_token(character, grain, language):
#         if language == 'mandarin':
#             if grain in [218095, 9004, 8004, 7004, 6004, 5004, 4004, 3004, 2004, 1004]:
#                 if character in map_mandarin_cdphone_senone:
#                     token = map_mandarin_cdphone_senone[character][grain]
#                 else:
#                     token = grain
#             elif grain == 179:
#                 phone = character.split('-')[-1].split('+')[0]
#                 token = map_mandarin_phone179_token[phone]
#             elif grain == 61:
#                 phone = character.split('-')[-1].split('+')[0]
#                 token = map_mandarin_phone61_token[map_mandarin_phone179_phone61[phone]]
#             elif grain == 32:
#                 phone = character.split('-')[-1].split('+')[0]
#                 token = map_mandarin_phone32_token[map_mandarin_phone61_phone32[map_mandarin_phone179_phone61[phone]]]
#             elif grain == 8:
#                 phone = character.split('-')[-1].split('+')[0]
#                 token = map_mandarin_place8_token[map_mandarin_phone32_place8[map_mandarin_phone61_phone32[
#                     map_mandarin_phone179_phone61[phone]]]]
#             else:
#                 raise NotImplementedError('unknown grain')
#         elif language == 'english':
#             if grain == 39:
#                 phone = character
#                 token = map_english_phone39_token[phone]
#             elif grain == 13:
#                 phone = character
#                 token = map_english_viseme13_token[map_english_phone39_viseme13[phone]]
#             elif grain == 10:
#                 phone = character
#                 token = map_english_place10_token[map_english_phone39_place10[phone]]
#             elif grain in [2704, 2008, 1256, 448]:
#                 token = character
#             else:
#                 raise NotImplementedError('unknown grain')
#         else:
#             raise NotImplementedError('unknown language')
#         return token


# def get_data_loader(
#         annotate, items, batch_size, max_batch_size, target_shape, pad_value, repeat=1, max_duration=100,
#         hop_duration=10, duration_factor=None, deleted_keys=None, key_output=False, dynamic=True,
#         bucket_length_multiplier=1.1, shuffle=False, drop_last=False, num_workers=4, pin_memory=False, seed=123456,
#         epoch=0, logger=None, distributed=False, **other_params):
#     dataset = AudioVisualTruncationDataset(annotate=annotate, repeat_num=repeat, max_duration=max_duration,
#                                            hop_duration=hop_duration, items=items, duration_factor=duration_factor,
#                                            deleted_keys=deleted_keys, key_output=key_output, logger=logger)
#
#     data_sampler = DynamicBatchSampler(lengths_list=dataset.duration, batch_size=batch_size, dynamic=dynamic,
#                                        max_batch_size=max_batch_size, epoch=epoch, drop_last=drop_last, logger=logger,
#                                        bucket_length_multiplier=bucket_length_multiplier, shuffle=shuffle, seed=seed)
#
#     if distributed:
#         data_sampler = DistributedSamplerWrapper(sampler=data_sampler, seed=seed, shuffle=shuffle, drop_last=drop_last)
#
#     collate_fn = PaddedBatch(items=dataset.items, target_shape=target_shape, pad_value=pad_value)
#
#     data_loader = DataLoader(dataset=dataset, collate_fn=collate_fn, num_workers=num_workers, pin_memory=pin_memory,
#                              batch_sampler=data_sampler)
#
#     return data_loader, data_sampler


# class AudioVisualTruncationDataset(Dataset):
#     def __init__(self, annotate, repeat_num, max_duration, hop_duration, items, key_output=False, logger=None,
#                  **other_params):
#         super(AudioVisualTruncationDataset, self).__init__()
#         if isinstance(annotate, str):
#             annotate = [annotate]
#         elif isinstance(annotate, list) and all(isinstance(a, str) for a in annotate):
#             annotate = annotate
#         else:
#             raise ValueError('unknown annotate type: {}.'.format(annotate))
#         annotate_num = len(annotate)
#         repeat_num = self.expend_params(value=repeat_num, length=annotate_num)
#         max_duration = self.expend_params(value=max_duration, length=annotate_num)
#         hop_duration = self.expend_params(value=hop_duration, length=annotate_num)
#
#         self.items = ['key', *items] if items[0] != 'key' and key_output else items
#         self.keys = []
#         self.duration = []
#         self.begin = []
#         self.key2path = {}
#         for annotate_id in range(annotate_num):
#             write_log(content='Load {} from {}, max_duration is {} s, hop_duration is {} s, repeat {} times.'.format(
#                 ','.join(items), annotate[annotate_id], max_duration[annotate_id], hop_duration[annotate_id],
#                 repeat_num[annotate_id]), level='info', logger=logger)
#             data_dic = safe_load(file=annotate[annotate_id])
#             self.key2path.update(data_dic['key2path'])
#             split_keys, split_duration, split_begin = self.cut_off(
#                 keys=data_dic['keys'], duration=data_dic['duration'], max_duration=max_duration[annotate_id],
#                 hop_duration=hop_duration[annotate_id])
#             for _ in range(repeat_num[annotate_id]):
#                 self.keys = self.keys + split_keys
#                 self.duration = self.duration + split_duration
#                 self.begin = self.begin + split_begin
#             del data_dic
#         write_log(content='All duration is {} h'.format(np.sum(self.duration) / 3600.), level='info', logger=logger)
#
#     def __getitem__(self, index):
#         main_key = self.keys[index]
#         item2paths = self.key2path[main_key]
#         value_lst = []
#         for item in self.items:
#             value = self._get_value(key=main_key, item=item, begin=self.begin[index], duration=self.duration[index],
#                                     item2file=item2paths)
#             value_lst.append(value)
#             del value
#         return value_lst
#
#     def _get_value(self, key, item, begin, duration, item2file):
#         if item in ['lip_frames', 'clean_wave', 'mixture_wave', 'irm']:
#             item2sample_rate = {'lip_frames': 25, 'clean_wave': 16000, 'mixture_wave': 16000, 'irm': 100}
#             begin_point = int(np.around(begin * item2sample_rate[item]))
#             end_point = int(np.around(begin_point + duration * item2sample_rate[item]))
#             if item in ['irm']:
#                 item_data = safe_load(file=item2file[item])[begin_point: end_point].float()
#             else:
#                 item_data = safe_load(file=item2file[item])[begin_point:end_point]
#         elif item in ['key']:
#             item_data = key
#         elif item.split('_')[0] in ['classification', 'posterior']:
#             item_type, language, grain, interval = item.split('_')
#             grain = int(grain)
#             interval = float(interval)
#             fa_information = safe_load(file=item2file['fa'])
#             if language == 'mandarin':
#                 characters = fa_information['CD-phone']
#                 end_timestamp = np.array(fa_information['end_frame_40ms'])*0.04 - begin
#             elif language == 'english':
#                 if grain in [39, 13, 10]:
#                     characters = fa_information['phone_39']
#                     end_timestamp = np.array(fa_information['phone_39_end_time']) - begin
#                 elif grain in [2704, 2008, 1256, 448]:
#                     characters = fa_information['senone_{}'.format(grain)]
#                     end_timestamp = np.array(fa_information['senone_{}_end_time'.format(grain)]) - begin
#                 else:
#                     raise ValueError('unknown grain {} for english'.format(grain))
#             else:
#                 raise NotImplementedError('unknown language: {}'.format(language))
#             if item_type == 'classification':
#                 item_data = (torch.ones(int(np.around(duration/interval))) * grain).long()
#                 begin_timestamp = 0.
#                 for i in range(len(characters)):
#                     if end_timestamp[i] > 0:
#                         begin_idx = int(np.around(begin_timestamp / interval))
#                         if end_timestamp[i] <= duration:
#                             end_idx = int(np.around(end_timestamp[i] / interval))
#                             item_data[begin_idx:end_idx] = self.map_character_token(characters[i], grain, language)
#                             begin_timestamp = end_timestamp[i]
#                         else:
#                             end_idx = int(np.around(duration / interval))
#                             item_data[begin_idx:end_idx] = self.map_character_token(characters[i], grain, language)
#                             break
#             else:
#                 item_data = torch.zeros(int(np.around(duration/interval)), grain, dtype=torch.float)
#                 begin_timestamp = 0.
#                 for i in range(len(characters)):
#                     if end_timestamp[i] > 0:
#                         begin_idx = int(np.around(begin_timestamp / interval))
#                         token = self.map_character_token(characters[i], grain, language)
#                         if end_timestamp[i] <= duration:
#                             end_idx = int(np.around(end_timestamp[i] / interval))
#                             if token < grain:
#                                 item_data[begin_idx:end_idx, token] = 1.
#                             begin_timestamp = end_timestamp[i]
#                         else:
#                             end_idx = int(np.around(duration / interval))
#                             if token < grain:
#                                 item_data[begin_idx:end_idx, token] = 1.
#                             break
#         else:
#             raise NotImplementedError('unknown output')
#         return item_data
#
#     def __len__(self):
#         return len(self.keys)
#
#     @staticmethod
#     def cut_off(keys, duration, max_duration, hop_duration):
#         split_keys = []
#         split_duration = []
#         split_begin = []
#         for idx in range(len(keys)):
#             idx_key = keys[idx]
#             idx_duration = duration[idx]
#             idx_begin = 0.
#             while idx_duration > max_duration:
#                 split_keys.append(idx_key)
#                 split_duration.append(max_duration)
#                 split_begin.append(idx_begin)
#                 idx_duration = idx_duration - hop_duration
#                 idx_begin = idx_begin + hop_duration
#             split_keys.append(idx_key)
#             split_duration.append(idx_duration)
#             split_begin.append(idx_begin)
#         return split_keys, split_duration, split_begin
#
#     @staticmethod
#     def map_character_token(character, grain, language):
#         if language == 'mandarin':
#             if grain in [218095, 9004, 8004, 7004, 6004, 5004, 4004, 3004, 2004, 1004]:
#                 if character in map_mandarin_cdphone_senone:
#                     token = map_mandarin_cdphone_senone[character][grain]
#                 else:
#                     token = grain
#             elif grain == 179:
#                 phone = character.split('-')[-1].split('+')[0]
#                 token = map_mandarin_phone179_token[phone]
#             elif grain == 61:
#                 phone = character.split('-')[-1].split('+')[0]
#                 token = map_mandarin_phone61_token[map_mandarin_phone179_phone61[phone]]
#             elif grain == 32:
#                 phone = character.split('-')[-1].split('+')[0]
#                 token = map_mandarin_phone32_token[map_mandarin_phone61_phone32[map_mandarin_phone179_phone61[phone]]]
#             elif grain == 8:
#                 phone = character.split('-')[-1].split('+')[0]
#                 token = map_mandarin_place8_token[map_mandarin_phone32_place8[map_mandarin_phone61_phone32[
#                     map_mandarin_phone179_phone61[phone]]]]
#             else:
#                 raise NotImplementedError('unknown grain')
#         elif language == 'english':
#             if grain == 39:
#                 phone = character
#                 token = map_english_phone39_token[phone]
#             elif grain == 13:
#                 phone = character
#                 token = map_english_viseme13_token[map_english_phone39_viseme13[phone]]
#             elif grain == 10:
#                 phone = character
#                 token = map_english_place10_token[map_english_phone39_place10[phone]]
#             elif grain in [2704, 2008, 1256, 448]:
#                 token = character
#             else:
#                 raise NotImplementedError('unknown grain')
#         else:
#             raise NotImplementedError('unknown language')
#         return token
#
#     @staticmethod
#     def expend_params(value, length):
#         if isinstance(value, list):
#             if len(value) == length:
#                 return value
#             else:
#                 raise ValueError('list have unmatched length: {}'.format(value))
#         else:
#             return [value for _ in range(length)]
