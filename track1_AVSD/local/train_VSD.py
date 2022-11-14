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
from model.vsd_net import Visual_VAD_Conformer_Net
from reader.reader_train_vsd import myDataLoader, myDataset
import prefetch_generator

def SoftCrossEntropy(inputs, target, reduction='mean'):
    '''
    inputs: Time * Num_class 
    target: Time * Num_class
    '''
    #print(inputs.shape)
    #print(target.shape)
    log_likelihood = -torch.nn.functional.log_softmax(inputs, dim=-1)
    batch = inputs.shape[0]
    if reduction == 'mean':
        loss = torch.sum(torch.mul(log_likelihood, target)) / batch
    else:
        loss = torch.sum(torch.mul(log_likelihood, target))
    return loss

def main(args):
    # Compile and configure all the model parameters.
    model_path = os.path.join("model", args.project)
    if not os.path.isdir(model_path):
        os.makedirs(model_path)
    log_dir = args.logdir
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)
    logger = utils.get_logger(log_dir + '/' + args.project)
    #seed_torch(args.seed)

    # load mean and var
    #lip_train_mean_var = np.load('scp_dir/train_mean_var_lip.npz')
    #lip_train_mean = lip_train_mean_var['_mean']
    #lip_train_var = lip_train_mean_var['_var']

    # file path
    file_train_path = args.file_train_path
    rttm_train_path = args.rttm_train_path
    file_dev_path = args.file_dev_path
    rttm_dev_path = args.rttm_dev_path
    
    # define the dataloader
    print("loading the dataset ...")

    dataset_train = myDataset(file_train_path, rttm_train_path) #, lip_train_mean, lip_train_var)
    dataset_dev = myDataset(file_dev_path, rttm_dev_path, min_dur=25 * 8, max_dur=25 * 8, frame_shift=25 * 8) #, lip_train_mean, lip_train_var)
    dataloader_train = myDataLoader(dataset=dataset_train,
                            batch_size=args.minibatchsize_train,
                            shuffle=True,
                            num_workers=args.train_num_workers)
    dataloader_dev = myDataLoader(dataset=dataset_dev,
                            batch_size=args.minibatchsize_dev,
                            shuffle=False,
                            num_workers=args.dev_num_workers)
    print("- done.")
    all_file = len(dataloader_train)
    all_file_dev = len(dataloader_dev)
    print("- {} training samples, {} dev samples".format(len(dataset_train), len(dataset_dev)))
    print("- {} training batch, {} dev batch".format(len(dataloader_train), len(dataloader_dev)))

    # define the model
    nnet = Visual_VAD_Conformer_Net()
    #nnet = nnet.cuda()

    # training setups
    optimizer = optim.Adam(nnet.parameters(), lr=args.lr_lipencoder)

    if args.start_iter == 0:
        pretrained_dict = torch.load("./model/pretrained/lipreading_LRW.pt", map_location=torch.device("cpu"))
        model_dict = nnet.lip_encoder.state_dict()
        new_model_dict = dict()
        for k, v in model_dict.items():
            if ('video_frontend.' + k) in pretrained_dict.keys() and v.size() == pretrained_dict['video_frontend.' + k].size():
                new_model_dict[k] = pretrained_dict['video_frontend.' + k]
        new_model_dict = {k: v for k, v in new_model_dict.items()
                        if k in model_dict.keys()
                        and v.size() == model_dict[k].size()}
        nnet.lip_encoder.load_state_dict(new_model_dict)
    else:
        checkpoint = torch.load(os.path.join(model_path, "{}_{}.model".format(args.model_name, args.start_iter-1)))
        state_dict = {}
        for l in checkpoint['model'].keys():
            state_dict[l.replace("module.", "")] = checkpoint['model'][l]
        nnet.load_state_dict(state_dict)
    #CE_loss = SoftCrossEntropy 
    
    #nnet = torch.nn.DataParallel(nnet, device_ids = [int(l) for l in args.gpu.split(',')]).cuda()
    nnet = torch.nn.DataParallel(nnet, device_ids = [0, 1, 2, 3]).cuda()
    softmax = torch.nn.Softmax(dim=1)
    for iter_ in range(args.start_iter, args.end_iter):
        start_time = time.time()
        running_loss = 0.0
        nnet.train()
        for video_feature, data_label_torch, current_frame in tqdm.tqdm(prefetch_generator.BackgroundGenerator(dataloader_train), total=len(dataloader_train)):
            video_feature = video_feature.cuda()
            data_label_torch = data_label_torch.cuda()
            outputs = nnet(video_feature, torch.from_numpy(np.array(current_frame, dtype=np.int32)))
            loss = SoftCrossEntropy(outputs, data_label_torch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        
        logger.info("Iteration:{0}, loss = {1:.6f} ".format(iter_, running_loss / all_file))
        # eval
        # torch.save(nnet.state_dict(), os.path.join(model_path, "{}_{}.model".format(args.model_name, iter_)))
        torch.save({'model': nnet.state_dict(), \
                    'optimizer': optimizer.state_dict()}, \
                    os.path.join(model_path, "{}_{}.model".format(args.model_name, iter_)))

        nnet.eval()
        pre_list, label_list = [], []
        with torch.no_grad():
            running_loss_dev = 0.0
            for video_feature_dev, data_label_torch_dev, current_frame_dev in tqdm.tqdm(prefetch_generator.BackgroundGenerator(dataloader_dev), total=len(dataloader_dev)):
                video_feature_dev = video_feature_dev.cuda()
                data_label_torch_dev = data_label_torch_dev.cuda()
                outputs_dev = nnet(video_feature_dev, torch.from_numpy(np.array(current_frame_dev, dtype=np.int32)))
                loss_dev = SoftCrossEntropy(outputs_dev, data_label_torch_dev)
                
                running_loss_dev += loss_dev.item()
                #outputs_dev_np = (torch.ceil(softmax(outputs_dev)-0.5)).data.cpu().numpy()
                #print(outputs_dev_np.shape)
                #print(data_label_torch_dev.data.cpu().numpy().shape)
                outputs_dev_np = softmax(outputs_dev).data.cpu().numpy()
                pre_list.extend(outputs_dev_np[:,1])
                label_list.extend(list(data_label_torch_dev.data.cpu().numpy()[:, 1]))
            pre_label = np.zeros(len(pre_list))
            pre_label[ np.array(pre_list) > 0.5 ] = 1
            label_list = np.array(label_list)
            total_frame = len(label_list)
            accurate_frame = np.sum(pre_label==label_list)
            speech_frame = np.sum(label_list==1)
            silence_frame = np.sum(label_list==0)
            FA_frame = np.sum(pre_label[label_list==0]==1)
            MISS_frame = np.sum(pre_label[label_list==1]==0)
            logger.info(f"total_frame:{total_frame} accurate_frame:{accurate_frame} speech_frame:{speech_frame} silence_frame:{silence_frame} FA_frame:{FA_frame} MISS_frame:{MISS_frame}")
            logger.info("Far video Epoch:%d, Train loss=%.4f, Valid loss=%.4f, ACC=%.4f, FA:%.4f, MISS:%.4f" % (iter_,
                        running_loss / all_file, running_loss_dev / all_file_dev, accurate_frame / total_frame, FA_frame / total_frame, MISS_frame / total_frame))

        
        nnet.train()

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
    parser.add_argument("--lr_lipencoder", default=1e-4, type=float, help="learning rate")
    parser.add_argument("--lr_lipdecoder", default=1e-4, type=float, help="learning rate")
    parser.add_argument("--minibatchsize_train", default=64, type=int)
    parser.add_argument("--minibatchsize_dev", default=64, type=int)
    parser.add_argument("--seed", default=617, type=int)
    parser.add_argument("--project", default='conformer_v_sd_manual_label', type=str)
    parser.add_argument("--logdir", default='./log/', type=str)
    parser.add_argument("--model_name", default='conformer_v_sd', type=str)
    parser.add_argument("--start_iter", default=0, type=int)
    parser.add_argument("--end_iter", default=5, type=int)
    parser.add_argument("--train_num_workers", default=16, type=int, help="number of training workers")
    parser.add_argument("--dev_num_workers", default=16, type=int, help="number of validation workers")
    parser.add_argument("--file_train_path", default='scp_dir/MISP_Far/train.lip.scp', type=str)
    parser.add_argument("--rttm_train_path", default='scp_dir/MISP_Far/train.rttm', type=str)
    parser.add_argument("--file_dev_path", default='scp_dir/MISP_Far/dev.lip.scp', type=str)
    parser.add_argument("--rttm_dev_path", default='scp_dir/MISP_Far/dev.rttm', type=str)
    args = parser.parse_args()
    # torch.backends.cudnn.enabled=False
    # run main
    main(args)
