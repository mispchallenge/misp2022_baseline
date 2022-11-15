#!/usr/bin/env python
# _*_ coding: UTF-8 _*_
import os
import argparse
import torch
import kaldiio
from tqdm import tqdm
from multiprocessing import Pool

def pdf_array2pt(pdf_array, frame_dur, frame_shift, pt_path):
    pdf_id = [pdf_array[0]]
    stamp = [frame_dur]
    for pdf in pdf_array[1:]:
        if pdf == pdf_id[-1]:
            stamp[-1] += frame_shift
        else:
            pdf_id.append(pdf)
            stamp.append(stamp[-1] + frame_shift)
    if not os.path.exists(os.path.split(pt_path)[0]):
        os.makedirs(os.path.split(pt_path)[0], exist_ok=True)
    torch.save({'pdf': pdf_id, 'stamp': stamp}, pt_path)
    return None

def pdf_ark2pt_worker(pdf_dic, frame_dur, frame_shift, output_dir, processing_id=None, processing_num=None):
    pdf_keys = sorted([*pdf_dic.keys()])
    for pdf_idx in tqdm(range(len(pdf_keys)), leave=True, desc='0' if processing_id is None else str(processing_id)):
    # for pdf_idx in range(len(pdf_keys)):
        if processing_id is None:
            processing_token = True
        else:
            if pdf_idx % processing_num == processing_id:
                processing_token = True
            else:
                processing_token = False
        if processing_token and not os.path.exists(os.path.join(output_dir, '{}.pt'.format(pdf_keys[pdf_idx]))):
            pdf_array2pt(pdf_array=pdf_dic[pdf_keys[pdf_idx]], frame_dur=frame_dur, frame_shift=frame_shift, pt_path=os.path.join(output_dir, '{}.pt'.format(pdf_keys[pdf_idx])))
    return None


def pdf_ark2pt_manager(pdf_scp, frame_dur, frame_shift, output_dir, processing_num=1):
    pdf_dic = kaldiio.load_scp(pdf_scp)
    if processing_num > 1:
        pool = Pool(processes=processing_num)
        for i in range(processing_num):
            pool.apply_async(pdf_ark2pt_worker, kwds={'pdf_dic': pdf_dic, 'frame_dur': frame_dur, 'frame_shift': frame_shift, 'output_dir': output_dir, 
                                                      'processing_id': i, 'processing_num': processing_num})
        pool.close()
        pool.join()
    else:
        pdf_ark2pt_worker(pdf_dic=pdf_dic, frame_dur=frame_shift, frame_shift=frame_shift, output_dir=output_dir)
    return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser('pdf_ark2pt')
    parser.add_argument('pdf_scp', type=str, default='./local/tmp/wpe.scp', help='list file of wav, format is scp')
    parser.add_argument('frame_dur', type=float, default=0.02, help='list file of wav, format is scp')
    parser.add_argument('frame_shift', type=float, default=0.01, help='list file of wav, format is scp')
    parser.add_argument('output_dir', type=str, default='wpe', help='output wpe data root')
    parser.add_argument('-nj', type=int, default='1', help='number of process')
    args = parser.parse_args()
    pdf_ark2pt_manager(pdf_scp=args.pdf_scp, frame_dur=args.frame_dur, frame_shift=args.frame_shift, output_dir=args.output_dir, 
                       processing_num=args.nj)
