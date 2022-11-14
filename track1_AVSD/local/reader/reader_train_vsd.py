import numpy as np
import torch
import sys
import os
sys.path.append(".")
sys.path.append("..")
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset,DataLoader
import HTK
import random
from .data_augment import RandAugment
from PIL import Image


class LibriSpeech_Manuel_Label_Generate():
    def __init__(self, oracle_rttm, differ_silence_inference_speech=False):
        self.differ_silence_inference_speech = differ_silence_inference_speech
        self.frame_label = self.get_label(oracle_rttm)

    def get_label(self, oracle_rttm):
        '''
        SPEAKER session0_CH0_0L 1  116.38    3.02 <NA> <NA> 5683 <NA>
        '''
        files = open(oracle_rttm)
        MAX_len = {}
        rttm = {}
        self.all_speaker_list = []
        for line in files:
            line = line.split(" ")
            session = line[1]
            if not session in MAX_len.keys():
                MAX_len[session] = 0
            start = np.int(np.float(line[3]) * 100)
            end = np.int(np.float(line[4]) * 100) + start
            if end > MAX_len[session]:
                MAX_len[session] = end
        files.close()
        files = open(oracle_rttm)
        for line in files:
            line = line.split(" ")
            session = line[1]
            spk = line[-3]
            self.all_speaker_list.append(spk)
            if not session in rttm.keys():
                rttm[session] = {}
            if not spk in rttm[session].keys():
                if self.differ_silence_inference_speech:
                    rttm[session][spk] = np.zeros([MAX_len[session], 3], dtype=np.int8)
                else:
                    rttm[session][spk] = np.zeros([MAX_len[session], 2], dtype=np.int8)
            #print(line[3])
            start = np.int(np.float(line[3]) * 25)
            end = np.int(np.float(line[4]) * 25) + start
            rttm[session][spk][start: end, 1] = 1
        for session in rttm.keys():
            for spk in rttm[session].keys():
                rttm[session][spk][:, 0] = 1 - rttm[session][spk][:, 1]
        if self.differ_silence_inference_speech:
            for session in rttm.keys():
                num_speaker = 0
                temp_label = {}
                for spk in rttm[session].keys():
                    num_speaker += rttm[session][spk][:, 1] # sum the second dim
                for spk in rttm[session]:
                    num_inference_speaker = num_speaker - rttm[session][spk][:, 1] # get the number of no-target_speaker
                    temp_label[spk] = copy.deepcopy(rttm[session][spk])
                    without_target_speaker_mask = rttm[session][spk][:, 1] == 0 # get true when this is no-target_speaker
                    # 3 class: silence(0), target speech(1), inference speech(2)
                    temp_label[spk][without_target_speaker_mask & (num_inference_speaker>0), 0] = 0 # there is no-target_speaker
                    temp_label[spk][without_target_speaker_mask & (num_inference_speaker>0), 2] = 1
                rttm[session] = temp_label
        self.all_speaker_list = list(set(self.all_speaker_list))
        files.close()
        # pdb.set_trace()
        # print(len(rttm.keys()))
        return rttm
            
    def mixture_num_speaker(self, session):
        return len(self.frame_label[session])

    def get_session_length(self, session):
        for spk in self.frame_label[session].keys():
            return len(self.frame_label[session][spk])

    def get_label_single_speaker(self, session, speaker, start=0, end=None):
        try:
            return self.frame_label[session][speaker][start: end, :]
        except:
            print("{} {} not in labels! return []".format(session, speaker))
            return []



class myDataset(Dataset):
    def __init__(self, feature_scp, rttm_path, lip_train_mean=None, lip_train_var=None, data_augment=False, min_dur=25, max_dur=25 * 8, frame_shift=25 * 6):
        self.lip_train_mean = lip_train_mean
        self.lip_train_var = lip_train_var
        if lip_train_mean==None:
            self.norm = False
        else:
            self.norm = True
        self.data_augment = data_augment
        if self.data_augment:
            self.image_augment = RandAugment(2, 9)
        self.min_dur = min_dur
        self.max_dur = max_dur
        self.frame_shift = frame_shift
        self.label = LibriSpeech_Manuel_Label_Generate(rttm_path)
        self.feature_list = self.get_feature_info(feature_scp)

    def get_feature_info(self, feature_scp):
        feature_list = []
        with open(feature_scp) as SCP_IO:
            for l in SCP_IO:
                '''
                /disk2/mkhe/data/misp2021/detection_lip/train/middle/R03_S102103104_C08_I0_Middle_104-25364-37719.htk
                '''
                session = "_".join(os.path.basename(l).split("_")[:-2])
                speaker, start, end = os.path.basename(l).split("_")[-1].split(".")[0].split("-")
                start = int(start)
                end = int(end) + 1
                if(end - start < self.min_dur):
                    continue
                cur_frame = start
                try:
                    session_length = self.label.get_session_length(session)
                except:
                    print(f"{session} not in label")
                    #break
                    continue
                while(cur_frame < end and (cur_frame+self.max_dur) < session_length):
                    if cur_frame + self.max_dur <= end:
                        feature_list.append((l.rstrip(), session, speaker, cur_frame, cur_frame+self.max_dur, cur_frame-start))
                        cur_frame += self.frame_shift
                    else:
                        cur_frame = max(start, end-self.max_dur)
                        feature_list.append((l.rstrip(), session, speaker, cur_frame, end, cur_frame-start))
                        break
        return feature_list

    def __getitem__(self, idx):
        path, session, speaker, start, end, offset = self.feature_list[idx]
        try:
            lip_feature = HTK.readHtk_start_end3D(path, offset, end-start+offset)
        except:
            print(f"load htk failed at {self.feature_list[idx]}")
            return None

        T = lip_feature.shape[0]
        lip_feature = np.array(lip_feature, dtype=np.uint8)
        if self.data_augment:
            for t in range(T):
                lip_feature[t, ...] = np.array(self.image_augment.forward(Image.fromarray(lip_feature[t, ...])))
        if self.norm:
            lip_feature = (lip_feature[:T,:,:] - np.tile(self.lip_train_mean, (T, 1, 1)) ) / \
                            np.sqrt(np.tile(self.lip_train_var, (T, 1, 1)) + 1e-6)

        mask_label = self.label.get_label_single_speaker(session, speaker, start=start, end=end)
        if mask_label == []:
            print(f"mask_label=[] at {self.feature_list[idx]}")
            return None
        else:
            return lip_feature, mask_label

    def __len__(self):
        return len(self.feature_list)

def myCollateFn(sample_batch):
    sample_batch = [ x for x in sample_batch if x != None ]
    sample_batch = sorted(sample_batch, key=lambda x: x[0].shape[0], reverse=True)
    data_feature = [torch.from_numpy(x[0].astype(np.float32)) for x in sample_batch]
    data_label = [torch.from_numpy(x[1].astype(np.float32)) for x in sample_batch]
    data_length = [x.shape[0]//1 for x in data_feature]
    data_feature = pad_sequence(data_feature, batch_first=True, padding_value=0.0)
    data_label = torch.cat(data_label, dim=0) # Time_Batch1 + Time_Batch2 + ... + Time_BatchN
    return data_feature, data_label, data_length

def collate_fn2(sample_batch):
    mask_label = [torch.from_numpy(sample_batch[0][1][i, ...].astype(np.float32)) for i in range(len(sample_batch[0][1]))]
    mask_label = torch.cat(mask_label, dim=0)
    #print(sample_batch[0][0].shape)
    return torch.from_numpy(sample_batch[0][0].astype(np.float32)), mask_label, [200 for i in range(8)]

class myDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super(myDataLoader, self).__init__(*args, **kwargs)
        self.collate_fn = myCollateFn
