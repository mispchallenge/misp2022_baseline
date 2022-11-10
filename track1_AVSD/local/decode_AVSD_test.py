# -*- coding: utf-8 -*-

import os
import numpy as np
import torch
import tqdm
import argparse
import multiprocessing
import prefetch_generator
import json

import config
from reader.reader_avsd import Audio_AEmbedding_Video_Worse_Data_Decode_Reader, decoder_collate_fn, Label_Generate_From_RTTM
from model.avsd_net import AIVECTOR_ConformerVEmbedding_SD_JOINT


softmax = torch.nn.Softmax(dim=2)

def preds_to_rttm(utt, num_spk, preds, output_path):
    for i, p in enumerate(preds):
        np.save(os.path.join(output_path, f"{utt[i]}.npy"), p[:num_spk[i], ...])

def multi_process_write(utt, num_spk, preds, output_path, num_thread=32):
    utts = [ l.tolist() for l in np.array_split(utt, num_thread) ]
    num_spks = [ l.tolist() for l in np.array_split(num_spk, num_thread) ]
    preds = [ l for l in np.array_split(preds, num_thread) ]
    processes = []
    for id in range(num_thread):
            p = multiprocessing.Process(target=preds_to_rttm, args=(utts[id], num_spks[id], preds[id], output_path, ))
            p.start()
            processes.append(p)
    for p in processes:
        p.join()

def main(args):
    if not os.path.exists(args.prob_dir+"/v_sd"):
        os.makedirs(args.prob_dir+"/v_sd")
    if not os.path.exists(args.prob_dir+"/av_sd"):
            os.makedirs(args.prob_dir+"/av_sd")
    label_train = Label_Generate_From_RTTM(args.rttm_train)
    dataset_dev = Audio_AEmbedding_Video_Worse_Data_Decode_Reader(args.fbank_dir, args.ivector_decode, args.lip_decode_scp, args.ivector_train, args.lip_train_scp, label_train, min_speaker=2, max_speaker=6, max_utt_durance=800, frame_shift=600, discard_video=args.discard_rate/100.)

    with open(os.path.join(args.prob_dir, "session2speaker"), "w") as IN:
        data = json.dumps(dataset_dev.speakers, indent=4)
        IN.write(data)

def make_argparse():
    # Set up an argument parser.
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('--vsd_model_path', metavar='PATH', required=True,
                        help='model_path.')
    parser.add_argument('--avsd_model_path', metavar='PATH', required=True,
                        help='model_path.')
    parser.add_argument('--fbank_dir', metavar='PATH', required=True,
                        type=str, help='fbank_dir.')
    parser.add_argument('--rttm_train', metavar='PATH', required=True,
                        type=str, help='rttm_train.')
    parser.add_argument('--lip_train_scp', metavar='PATH', required=True,
                        type=str, help='lip_train_scp.')
    parser.add_argument('--lip_decode_scp', metavar='PATH', required=True,
                        type=str, help='lip_decode_scp.')
    parser.add_argument('--ivector_train', metavar='PATH', required=True, 
                        type=str, help='ivector_train.')
    parser.add_argument('--ivector_decode', metavar='PATH', required=True, 
                        type=str, help='ivector_decode.')
    parser.add_argument('--discard_rate', metavar='PATH', default=0, type=int,
                        help='discard_rate.')
    parser.add_argument('--num_workers', metavar='PATH', default=8, type=int,
                        help='num_workers.')
    parser.add_argument('--minibatchsize', metavar='PATH', default=6, type=int,
                        help='minibatchsize.')
    parser.add_argument('--prob_dir', metavar='PATH', required=True,
                        help='prob_dir.')

    return parser


if __name__ == '__main__':
    parser = make_argparse()
    args = parser.parse_args()
    main(args)
