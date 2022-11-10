# -*- coding: utf-8 -*-

import os
from nbformat import NBFormatError
import numpy as np
import torch
import tqdm
import argparse
import time
import HTK
from model.vsd_net import Visual_VAD_Conformer_Net
from reader.reader_vsd import myDataLoader, myDataset
import multiprocessing

def preds_to_rttm(utt, preds, output_path):
    for i, p in enumerate(preds):
        np.save(os.path.join(output_path, f"{utt[i]}.npy"), p)

def multi_process_write(utt, preds, output_path, num_thread=32):
    utts = [ l.tolist() for l in np.array_split(utt, num_thread) ]
    preds = [ l for l in np.array_split(preds, num_thread) ]
    processes = []
    for id in range(num_thread):
            p = multiprocessing.Process(target=preds_to_rttm, args=(utts[id], preds[id], output_path, ))
            p.start()
            processes.append(p)
    for p in processes:
        p.join()
    return 0

def write_htks(utt_name, data, nframe, output_dir):
    for id, name in enumerate(utt_name):
        HTK.writeHtk(os.path.join(output_dir, f"{name}.htk"), data[id, :nframe[id], :])

def multi_process_embedding_write(utt_name, data, nframe, embedding_output_dir, num_thread):
    utt_names = [ l.tolist() for l in np.array_split(utt_name, num_thread) ]
    datas = [ l for l in np.array_split(data, num_thread) ]
    nframes = [ l.tolist() for l in np.array_split(nframe, num_thread) ]
    processes = []
    for id in range(num_thread):
            p = multiprocessing.Process(target=write_htks, args=(utt_names[id], datas[id], nframes[id], embedding_output_dir, ))
            p.start()
            processes.append(p)
    for p in processes:
        p.join()
    return 0

def main(args):
    if not os.path.exists(args.prob_dir):
        os.makedirs(args.prob_dir)
    if not os.path.exists(args.embedding_output_dir):
        os.makedirs(args.embedding_output_dir)
    #torch.cuda.set_device(0)
    dataset_train = myDataset(args.lip_train_scp, args.rttm_train, dev_scp=args.lip_decode_scp)
    dataloader_train = myDataLoader(dataset=dataset_train,
                            batch_size=32,
                            shuffle=False,
                            num_workers=8)
    softmax = torch.nn.Softmax(dim=1)
    with torch.no_grad():
        nnet = Visual_VAD_Conformer_Net()
        checkpoint = torch.load(args.model_path)
        state_dict = {}
        for l in checkpoint['model'].keys():
            state_dict[l.replace("module.", "")] = checkpoint['model'][l]
        nnet.load_state_dict(state_dict)
        nnet = torch.nn.DataParallel(nnet, device_ids = [0, 1]).cuda()
        nnet.eval()
        for video_feature, utt_name, current_frame in tqdm.tqdm(dataloader_train, total=len(dataloader_train)):
            ypreds, embedding = nnet(video_feature.cuda(), torch.from_numpy(np.array(current_frame, dtype=np.int32)), return_embedding=True) # (B, T, 256)
            ypreds = softmax(ypreds).detach().cpu().numpy()
            cur_frame = 0
            preds = []
            for n in current_frame:
                preds.append(ypreds[cur_frame:(cur_frame+n), :])
                cur_frame += n
            multi_process_write(utt_name, preds, args.prob_dir, 8)
            #multi_process_embedding_write(utt_name, embedding.detach().cpu().numpy(), current_frame, args.embedding_output_dir, 16)


def make_argparse():
    # Set up an argument parser.
    parser = argparse.ArgumentParser(description='Prepare ivector extractor weights for ivector extraction.')

    parser.add_argument('--model_path', metavar='PATH', required=True,
                        help='model_path.')
    parser.add_argument('--prob_dir', metavar='PATH', required=True,
                        help='prob_dir.')
    parser.add_argument('--embedding_output_dir', metavar='PATH', required=True,
                        help='embedding_output_dir.')
    parser.add_argument('--lip_train_scp', metavar='PATH', required=True,
                        help='lip_train_scp.')
    parser.add_argument('--rttm_train', metavar='PATH', required=True,
                        help='rttm_train.')
    parser.add_argument('--lip_decode_scp', metavar='PATH', required=True,
                        help='lip_decode_scp.')
    return parser


if __name__ == '__main__':
    parser = make_argparse()
    args = parser.parse_args()
    main(args)
