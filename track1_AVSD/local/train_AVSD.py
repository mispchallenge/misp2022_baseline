# -*- coding: utf-8 -*-
import argparse
import os
import time
import torch
import numpy as np
import torch.optim as optim
import torch.nn as nn
import sys
import copy
import random
import tqdm
sys.path.append("..")
import utils
from model.avsd_net import AIVECTOR_ConformerVEmbedding_SD
from reader.reader_avsd import Label_Generate_From_RTTM, collate_fn, Audio_AEmbedding_VEmbedding_Worse_Data_Reader
import config
from loss_function import SoftCrossEntropy_SingleTargets


def main(args):
    # Compile and configure all the model parameters.
    model_path = os.path.join("model", args.project)
    if not os.path.isdir(model_path):
        os.makedirs(model_path)
    log_dir = args.logdir
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)
    logger = utils.get_logger(log_dir + '/' + args.project)
    torch.backends.cudnn.enabled = False
    #seed_torch(args.seed)

    # define the model
    nnet = AIVECTOR_ConformerVEmbedding_SD(config.configs_SC_Multiple_6Speakers_AEmbedding_VEmbedding_2Classes)
    #nnet = nnet.cuda()

    # training setups
    optimizer = optim.Adam(nnet.parameters(), lr=args.lr)

    if args.start_iter > 0:
        checkpoint = torch.load(os.path.join(model_path, "{}_{}.model".format(args.model_name, args.start_iter-1)))
        state_dict = {}
        for l in checkpoint['model'].keys():
            state_dict[l.replace("module.", "")] = checkpoint['model'][l]
        nnet.load_state_dict(state_dict)
    
    nnet = torch.nn.DataParallel(nnet, device_ids = [0, 1, 2, 3]).cuda()
    #softmax = torch.nn.Softmax(dim=2)
    label_2classes_train = Label_Generate_From_RTTM(args.rttm_train_path)
    for iter_ in range(args.start_iter, args.end_iter):
        start_time = time.time()
        running_loss = 0.0
        nnet.train()
        dataset_train = Audio_AEmbedding_VEmbedding_Worse_Data_Reader(args.train_audio_fea_dir, args.train_speaker_embedding, args.train_video_fea_scp, label_2classes_train, set_video_silence=0.15, set_video_speak=0.20, min_speaker=2, max_speaker=6, max_utt_durance=800, frame_shift=600, audio_type=args.audio_type.split("+"))
        dataloader_train = torch.utils.data.DataLoader(dataset_train, 
                                batch_size=args.minibatchsize_train, 
                                collate_fn=collate_fn, 
                                shuffle=True, drop_last=True, 
                                num_workers=args.train_num_workers)
        all_file = len(dataloader_train)
        checkpoint_save_model = int(all_file // 6)
        batch_id = 0
        for audio_fea, audio_embedding, video_embedding, mask_label, nframe in tqdm.tqdm(dataloader_train):
            audio_fea = audio_fea.cuda()
            audio_embedding = audio_embedding.cuda()
            video_embedding = video_embedding.cuda()
            mask_label = mask_label.cuda()
            outputs = nnet(audio_fea, audio_embedding, video_embedding, torch.from_numpy(np.array(nframe, dtype=np.int32)))
            loss = SoftCrossEntropy_SingleTargets(outputs, mask_label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            batch_id += 1
            if (batch_id % checkpoint_save_model == 0):
                torch.save({'model': nnet.state_dict(), \
                    'optimizer': optimizer.state_dict()}, \
                    os.path.join(model_path, "{}_{}-chunk{}.model".format(args.model_name, iter_, batch_id)))
                
        logger.info("Iteration:{0}, loss = {1:.6f} ".format(iter_, running_loss / all_file))
        torch.save({'model': nnet.state_dict(), \
                    'optimizer': optimizer.state_dict()}, \
                    os.path.join(model_path, "{}_{}.model".format(args.model_name, iter_)))
        end_time = time.time()
        logger.info("Time used for each epoch training: {} seconds.".format(end_time - start_time))
        logger.info("*" * 50)

def seed_torch(seed=42):
    seed = int(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True

if __name__=="__main__":
    # Arguement Parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", default=1e-4, type=float, help="learning rate")
    parser.add_argument("--minibatchsize_train", default=48, type=int)
    parser.add_argument("--seed", default=617, type=int)
    parser.add_argument("--project", default='video_model', type=str)
    parser.add_argument("--logdir", default='./log/', type=str)
    parser.add_argument("--model_name", default='av_diarization', type=str)
    parser.add_argument("--start_iter", default=0, type=int)
    parser.add_argument("--end_iter", default=20, type=int)
    parser.add_argument("--train_num_workers", default=16, type=int, help="number of training workers")
    parser.add_argument("--train_audio_fea_dir", default='train_audio_fea_dir', type=str)
    parser.add_argument("--train_speaker_embedding", default='train_speaker_embedding', type=str)
    parser.add_argument("--train_video_fea_scp", default='train_video_fea_scp', type=str)
    parser.add_argument("--rttm_train_path", default='rttm_train_path', type=str)
    parser.add_argument("--audio_type", default='audio_type', type=str)
    args = parser.parse_args()

    # torch.backends.cudnn.enabled=False
    # run main
    main(args)
