# -*- coding: utf-8 -*-

import os
from nbformat import NBFormatError
import numpy as np
import torch
import tqdm
import argparse
import time
import HTK
from model.vsd_net import Visual_VAD_Conformer_Embedding
from reader.reader_vsd import myDataLoader, myDataset
import multiprocessing

def write_htks(utt_name, data, nframe):
    for id, name in enumerate(utt_name):
        HTK.writeHtk(os.path.join(args.output_dir, f"{name}.htk"), data[id, :nframe[id], :])

def multi_process_write(utt_name, data, nframe, num_thread):
    utt_names = [ l.tolist() for l in np.array_split(utt_name, num_thread) ]
    datas = [ l for l in np.array_split(data, num_thread) ]
    nframes = [ l.tolist() for l in np.array_split(nframe, num_thread) ]
    processes = []
    for id in range(num_thread):
            p = multiprocessing.Process(target=write_htks, args=(utt_names[id], datas[id], nframes[id], ))
            p.start()
            processes.append(p)
    for p in processes:
        p.join()

def main(args):
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    #torch.cuda.set_device(0)
    file_train_path = args.file_train_path
    rttm_train_path = args.rttm_train_path
    #dataset_train = myDataset(file_train_path, rttm_train_path, dev_scp="scp_dir/dev.scp") #, lip_train_mean, lip_train_var)
    dataset_train = myDataset(file_train_path, rttm_train_path) #, lip_train_mean, lip_train_var)
    dataloader_train = myDataLoader(dataset=dataset_train,
                            batch_size=32,
                            shuffle=False,
                            num_workers=8)
    with torch.no_grad():
        nnet = Visual_VAD_Conformer_Embedding()
        checkpoint = torch.load(args.model_path)
        state_dict = {}
        for l in checkpoint['model'].keys():
            state_dict[l.replace("module.", "")] = checkpoint['model'][l]
        nnet.load_state_dict(state_dict)
        nnet = torch.nn.DataParallel(nnet, device_ids = [0, 1]).cuda()
        nnet.eval()
        #point = time.time()
        for video_feature, utt_name, current_frame in tqdm.tqdm(dataloader_train, total=len(dataloader_train)):
            outputs = nnet(video_feature.cuda(), torch.from_numpy(np.array(current_frame, dtype=np.int32))) # (B, T, 256)
            multi_process_write(utt_name, outputs.detach().cpu().numpy(), current_frame, 16)

def make_argparse():
    # Set up an argument parser.
    parser = argparse.ArgumentParser(description='Prepare ivector extractor weights for ivector extraction.')

    #parser.add_argument('--feature_list', metavar='PATH', required=True,
    #                    help='feature_list')
    parser.add_argument('--model_path', metavar='PATH', required=True,
                        help='model_path.')
    parser.add_argument('--output_dir', metavar='PATH', required=True,
                        help='output_dir.')
    parser.add_argument('--file_train_path', metavar='PATH', default="scp_dir/MISP_Far/train.face.scp",
                        help='file_train_path.')
    parser.add_argument('--rttm_train_path', metavar='PATH', default="scp_dir/MISP_Far/train.rttm",
                        help='rttm_train_path.')

    return parser


if __name__ == '__main__':
    parser = make_argparse()
    args = parser.parse_args()
    main(args)
