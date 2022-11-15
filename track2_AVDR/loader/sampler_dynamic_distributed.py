#!/usr/bin/env python
# _*_ coding: UTF-8 _*_
import torch
import logging
import numpy as np
from operator import itemgetter
from torch.utils.data import DistributedSampler


class DynamicBatchSampler(object):
    """
    This BatchSampler batches examples together by grouping them by their length.
    Every example in the batch have approximate the same length and thus padding is minimized.
    This enables faster training on datasets where length of examples can vary significantly.
    Inspired by: https://www.tensorflow.org/api_docs/python/tf/data/experimental/bucket_by_sequence_length
    Dynamic batching is performed by specifying a max_batch_length which is the
    upper limit for the sum of the length of examples in a batch:
    e.g., if ex1 has length 4, ex2 length 5 and if max_batch_length is set to 6
    ex1 and ex2 will be placed, alone, in two distinct batches.
    Length for each example can be obtained in two manners.
    If the input dataset is a DynamicItemDataset it can be obtained by specifying a
    length_func. Default assumes a "duration" entry is in the annotation.
    Length for each example can also be passed to this class upon instantiation
    by specifying a list containing the length for each example and passing it to
    lengths_list.
    Examples are grouped together by defining a set of possible discrete intervals
    (buckets) multiple of a left_bucket_length.
    A bucket_length_multiplier is used to specify the number of possible buckets.
    E.g., if max_batch_length = 32 and left_bucket_length = 10, bucket_length_multiplier = 2
    there will be 3 buckets: [0, 10), [10, 20), [20, 40).
    A common choice would be setting left_bucket_length to approximatively the length
    of your shortest example in the dataset.
    Decreasing bucket_length_multiplier creates more buckets in the whole interval
    of [left_bucket_length, max_batch_size]: e.g. if max_batch_length = 32 and left_bucket_length = 10,
    bucket_length_multiplier = 1.5 the number of buckets increases to 8.
    With right boundaries: [10 12 14 17 21 25 30 36].
    Thus examples with length less than 10 are all grouped together but more buckets
    are created for longer examples.
    Note that the bucket boundary grows exponentially using the multiplier.
    The buckets can also be specified by passing a list to the bucket_boundaries
    argument instead of specifying a left_bucket_length and a bucket_length_multiplier.
    Arguments
    ---------
    batch_size : int
        Upper limit for the sum of the length of examples in a batch.
        Should be chosen based on your GPU memory.
    max_batch_size : int
        Minimum length of a bucket. Specifies resolution of buckets and thus this sampler random.
        A common choice is to set this to length of your shortest example.
    bucket_length_multiplier : float
        Multiplier for bucket length, specifies number of buckets from left_bucket_length to max_batch_length.
    shuffle : bool
        Whether or not shuffle examples between each epoch.
    bucket_boundaries : None
        Overrides bucket_length_multiplier and left_bucket_length by specifying manually
        the buckets right boundaries.
    lengths_list: list
        Overrides length_func by passing a list containing the length of each example
        in the dataset. This argument must be set when the dataset is a plain
        Pytorch Dataset object and not a DynamicItemDataset object as length_func
        cannot be used on Pytorch Datasets.
    epoch : int
        The epoch to start at.
    drop_last : bool
         If ``True``, the sampler will drop the last examples which
         have not been grouped.
    """
    def __init__(self, lengths_list, batch_size, max_batch_size=None, dynamic=False, bucket_length_multiplier=1.1,
                 shuffle=True, bucket_boundaries=None, seed=42, epoch=0, drop_last=False, logger=None):
        left_bucket_length = min(lengths_list)
        max_length = max(lengths_list)

        # take length of examples from this argument and bypass length_key
        self._ex_lengths = {}
        for index in range(len(lengths_list)):
            self._ex_lengths[str(index)] = lengths_list[index]

        if bucket_boundaries is not None:
            if not all([x >= 1 for x in bucket_boundaries]):
                raise ValueError('All elements in bucket boundaries should be >= 1.')
            if not len(set(bucket_boundaries)) == len(bucket_boundaries):
                raise ValueError('Bucket_boundaries should not contain duplicates.')

        self._bucket_boundaries = np.array(
            self.get_data_boundaries(
                max_length=max_length, bucket_boundaries=bucket_boundaries,
                left_bucket_length=left_bucket_length, bucket_length_multiplier=bucket_length_multiplier))

        self._shuffle = shuffle
        self._seed = seed
        self.logger = logger
        self._drop_last = drop_last
        self._bucket_lens = []
        # Calculate bucket lengths
        for i in range(1, len(self._bucket_boundaries)):
            if dynamic:
                if max_batch_size is not None:
                    self._bucket_lens.append(
                        np.clip(int(batch_size*max_length / self._bucket_boundaries[i]), 1, max_batch_size))
                else:
                    self._bucket_lens.append(max(int(batch_size * max_length / self._bucket_boundaries[i]), 1))
            else:
                self._bucket_lens.append(batch_size)
        self._epoch = epoch
        self.first_init = True
        self._generate_batches()

    def _generate_batches(self):

        if self._shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self._seed + self._epoch)
            sampler = torch.randperm(len(self._ex_lengths), generator=g).tolist()  # type: ignore
        else:
            sampler = range(len(self._ex_lengths))  # type: ignore

        self._batches = []
        bucket_batches = [[] for _ in self._bucket_lens]
        bucket_stats = [0 for _ in self._bucket_lens]
        for idx in sampler:
            item_len = self._ex_lengths[str(idx)]
            bucket_id = max(np.searchsorted(self._bucket_boundaries, item_len) - 1, 0)
            bucket_batches[bucket_id].append(idx)
            bucket_stats[bucket_id] += 1
            if len(bucket_batches[bucket_id]) >= self._bucket_lens[bucket_id]:
                self._batches.append(bucket_batches[bucket_id])
                bucket_batches[bucket_id] = []
        # Dump remaining batches - we might even want to shuffle those
        if not self._drop_last:
            for batch in bucket_batches:
                if batch:
                    self._batches.append(batch)
        if self.first_init:  # only log at first epoch
            write_log(
                content='DynamicBatchSampler: Created {} batches, {} buckets used.'.format(
                    len(self._batches), sum(np.array(bucket_stats) != 0)), logger=self.logger, level='info')
            bucket_left_boundary = self._bucket_boundaries[0]
            bucket_id = 0
            for i in range(1, len(self._bucket_boundaries)):
                if bucket_stats[i-1] != 0:
                    write_log(
                        content='DynamicBatchSampler: Bucket {} with boundary {}-{} and batch_size {} has {} '
                                'examples.'.format(bucket_id, np.around(bucket_left_boundary, 2),
                                                   np.around(self._bucket_boundaries[i], 2),
                                                   self._bucket_lens[i-1], bucket_stats[i-1]),
                        logger=self.logger, level='info')
                    bucket_left_boundary = self._bucket_boundaries[i]
                    bucket_id = bucket_id + 1
            self.first_init = False

    def __iter__(self):
        for batch in self._batches:
            yield batch
        if self._shuffle:  # re-generate batches only if shuffling
            self._generate_batches()

    def set_epoch(self, epoch):
        """
        You can also just access self.epoch, but we maintain this interface
        to mirror torch.utils.data.distributed.DistributedSampler
        """
        self._epoch = epoch
        self._generate_batches()

    def __len__(self):
        return len(self._batches)

    @staticmethod
    def get_data_boundaries(max_length, bucket_boundaries, left_bucket_length, bucket_length_multiplier):
        if not bucket_boundaries:
            if left_bucket_length <= 0:
                raise ValueError('left_bucket_length must be >0 if no bucket_boundaries set')
            if bucket_length_multiplier < 1.0:
                raise ValueError('bucket_length_multiplier must be >1.0 if no bucket_boundaries set')
            bucket_boundaries = {left_bucket_length}
            bucket_boundary = float(left_bucket_length)
            while True:
                bucket_boundary *= bucket_length_multiplier
                if bucket_boundary >= max_length:
                    break
                bucket_boundaries.add(bucket_boundary)
            bucket_boundaries.add(max_length)
        return list(sorted(bucket_boundaries))


# Heavily inspired by Catalyst, which is under Apache 2.0 licence.
# https://github.com/catalyst-team/catalyst/blob/51428d7756e62b9b8ee5379f38e9fd576eeb36e5/catalyst/data/sampler.py#L522
class DistributedSamplerWrapper(DistributedSampler):
    """This wrapper allows using any sampler with Distributed Data Parallel (DDP) correctly.
    Passing blindly the sampler to each DDP process will cause to have access
    within each process to all the data in the dataset instead of only a subset
    of it which is unique to each process.  This wrapper prevents this and
    allows to use only a subset of the original data for each process.
    NOTE
    ----
    This is is automatically applied to any sampler in the Brain class when DDP
    training is used.
    """

    def __init__(self, sampler, num_replicas=None, rank=None, shuffle=True, seed=0, drop_last=False):
        # DistributedSampler only calls len() on dataset
        # so a sampler is fine to pass there, as well.
        super().__init__(dataset=sampler, num_replicas=num_replicas, rank=rank, shuffle=shuffle, seed=seed,
                         drop_last=drop_last)
        self.sampler = sampler

    def __iter__(self):
        # It is easiest to use a random access interface to the wrapped
        # sampler's indices, so we just fetch all indices from the wrapped
        # sampler
        sampler_indices = list(self.sampler.__iter__())
        indices_of_indices = super().__iter__()
        # Itemgetter fetches the wrapped sampler indices from the positions
        # pointed to by DistributedSampler
        return iter(itemgetter(*indices_of_indices)(sampler_indices))

    def set_epoch(self, epoch):
        """Pass set_epoch() through to DistributedSampler and the wrapper one"""
        super().set_epoch(epoch)
        if hasattr(self.sampler, "set_epoch"):
            self.sampler.set_epoch(epoch)


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


if __name__ == '__main__':
    # from loader_truncation_dynamic_distributed import AudioVisualTruncationDataset
    #
    # checkout_dataset = AudioVisualTruncationDataset(
    #     annotate='/yrfs2/cv1/hangchen2/feature/iflytek_av_chinese/av_data/train_noisy.json', repeat_num=1,
    #     max_duration=100, hop_duration=6, items=['mixture_wave', 'lip_frames', 'irm'], logger='print')
    # checkout_batch_sampler = DynamicBatchSampler(
    #     lengths_list=checkout_dataset.duration, batch_size=32, max_batch_size=256, shuffle=True, bucket_boundaries=None,
    #     seed=42, bucket_length_multiplier=1.1, epoch=0, drop_last=False, logger='print')
    # checkout_distributed_batch_sampler = DistributedSamplerWrapper(
    #     sampler=checkout_batch_sampler, num_replicas=4, rank=0, shuffle=True, seed=123456, drop_last=False)
    #
    # for checkout_batch in checkout_distributed_batch_sampler:
    #     print(checkout_batch)
    #     break
    pass
