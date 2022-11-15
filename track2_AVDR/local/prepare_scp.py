#!/usr/bin/env python
# -- coding: UTF-8 
import os
import glob
import argparse
from tqdm import tqdm
from multiprocessing import Pool


from tool.data_io import safe_load, safe_store


# wav.scp <recording-id> <extended-filename>
def prepare_wav_scp(file_pattern, store_dir):
    all_file_lines = []
    file_path_list = glob.glob(file_pattern)
    for file_path in file_path_list:
        record_id = os.path.splitext(os.path.split(file_path)[-1])[0]
        all_file_lines.append('{} {}'.format(record_id, file_path))
    safe_store(file=os.path.join(store_dir, 'wav.scp'), data=sorted(all_file_lines), mode='cover', ftype='txt')
    return None

if __name__ == '__main__':
    parser = argparse.ArgumentParser('')
    parser.add_argument('file_pattern', type=str, default='', help='directory of wav')
    parser.add_argument('store_dir', type=str, default='data/train_far', help='set types')
    parser.add_argument('-nj', type=int, default=15, help='number of process')
    args = parser.parse_args()

    print('Preparing wav.scp with {}'.format(args.file_pattern))
    prepare_wav_scp(file_pattern=args.file_pattern, store_dir=args.store_dir)