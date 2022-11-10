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

def is_overlap(seg1, seg2):
    if seg1[1] <= seg2[0] or seg1[0] >= seg2[1]:
        return False
    else:
        return True

class myDataset(Dataset):
    def __init__(self, feature_scp, rttm_path, dev_scp=None, lip_train_mean=None, lip_train_var=None, min_dur=25, max_dur=25 * 8, frame_shift=25 * 6):
        self.lip_train_mean = lip_train_mean
        self.lip_train_var = lip_train_var
        if lip_train_mean==None:
            self.norm = False
        else:
            self.norm = True
        self.min_dur = min_dur
        self.max_dur = max_dur
        self.frame_shift = frame_shift
        self.label = LibriSpeech_Manuel_Label_Generate(rttm_path)
        self.silent_video_segments = self.get_silent_video_segments(feature_scp)
        if dev_scp == None:
            self.feature_list = self.get_feature_info(feature_scp)
        else:
            self.feature_list = self.get_feature_info(dev_scp)

    def get_feature_info(self, video_scp, field="Far"):
        feature_list = []
        MAX_DUR = {}
        with open(video_scp) as SCP_IO:
            for l in SCP_IO:
                '''
                /disk2/mkhe/data/misp2021/detection_lip/train/middle/R03_S102103104_C08_I0_Middle_104-25364-37719.htk
                '''
                session = "_".join(os.path.basename(l).split("_")[:-2])
                speaker, start, end = os.path.basename(l).split("_")[-1].split(".")[0].split("-")
                start = int(start)
                end = int(end) + 1
                if session not in MAX_DUR.keys():
                    MAX_DUR[session] = 0
                if MAX_DUR[session] < end:
                    MAX_DUR[session] = end

        video_path = {}
        with open(video_scp) as SCP_IO:
            for l in SCP_IO:
                session = "_".join(os.path.basename(l).split("_")[:-2])
                speaker, start, end = os.path.basename(l).split("_")[-1].split(".")[0].split("-")
                start = int(start)
                end = int(end) + 1
                if session not in video_path.keys():
                    video_path[session] = {}
                if speaker not in video_path[session].keys():
                    video_path[session][speaker] = []
                video_path[session][speaker].append([start, end, l.rstrip()])
        for session in video_path.keys():
            for speaker in video_path[session].keys():
                video_path[session][speaker].sort()
        for session in video_path.keys():
            cur_frame = 0
            while cur_frame + self.max_dur <= MAX_DUR[session]:
                for spk in video_path[session].keys():
                    cur_video_path = []
                    i = 0
                    while i < len(video_path[session][spk]) and cur_frame + self.max_dur > video_path[session][spk][i][0]:
                        if is_overlap([cur_frame, cur_frame + self.max_dur], video_path[session][spk][i][:2]):
                            cur_video_path.append(video_path[session][spk][i][2])
                        i += 1
                    while video_path[session][spk] != [] and cur_frame + self.frame_shift > video_path[session][spk][0][1]:
                        video_path[session][spk] = video_path[session][spk][1:]
                    feature_list.append([cur_video_path, f"{session}_{field}_{spk}", cur_frame, cur_frame+self.max_dur])
                cur_frame += self.frame_shift
        return feature_list

    def get_silent_video_segments(self, video_scp):
        silent_video_segments = []
        with open(video_scp) as SCP_IO:
            for l in SCP_IO:
                '''
                /disk2/mkhe/data/misp2021/detection_lip/train/middle/R03_S102103104_C08_I0_Middle_104-25364-37719.htk
                '''
                session = "_".join(os.path.basename(l).split("_")[:-2])
                speaker, start, end = os.path.basename(l).split("_")[-1].split(".")[0].split("-")

                try:
                    session_length = self.label.get_session_length(session)
                except:
                    #print(f"get_silent_video_segments: {session} not in label")
                    #break
                    continue

                start = int(start)
                end = int(end) + 1
                if(end - start < self.max_dur):
                    continue
                cur_frame = start
                while cur_frame + self.max_dur <= end and cur_frame + self.max_dur <= session_length//4:
                    if np.sum(self.label.frame_label[session][speaker][cur_frame*4:(cur_frame+self.max_dur)*4][:, 1]) < 0.01 * self.max_dur:
                        silent_video_segments.append((l.rstrip(), cur_frame-start, cur_frame+self.max_dur-start))
                    cur_frame += self.frame_shift
        print(f"silent video segments: {len(silent_video_segments)}")
        return silent_video_segments

    def get_slience_video(self, durance):
        current_durance = 0
        silence_video_fea = []
        while current_durance < durance:
            video_path, start, end = self.silent_video_segments[np.random.choice(range(len(self.silent_video_segments)))]
            current_durance += end - start
            silence_video_fea.append(HTK.readHtk_start_end3D(video_path, start, end))
        return np.vstack(silence_video_fea)[:durance, ...]

    def __getitem__(self, idx):
        cur_video_path, session_speaker, start, end = self.feature_list[idx]
        video_fea = np.zeros([end - start, 96, 96])
        utt_name = f"{session_speaker}-{start}-{end}"
        #try:
        if cur_video_path == []:
            video_fea[:end - start, ...] = self.get_slience_video(end - start)
        else:
            for i, s in enumerate(cur_video_path):
                _, c_start, c_end = os.path.basename(s).split("_")[-1].split(".")[0].split("-")
                c_start, c_end = int(c_start), int(c_end)+1
                if i == 0:
                    silence_start = start
                if c_start > silence_start:
                    silence = self.get_slience_video(c_start - silence_start)
                    try:
                        video_fea[silence_start-start:c_start-start, ...] = silence
                    except:
                        print(f"{session_speaker} start={start} silence_start={silence_start} c_start={c_start}")
                        print(video_fea[silence_start-start:c_start-start, ...].shape)
                        print(silence.shape)
                silence_start = c_end
                if c_start < start:
                    if c_end < end:
                        video_fea[:c_end-start, ...] = HTK.readHtk_start_end3D(s, start-c_start, c_end-c_start)
                    else:
                        video_fea[:end-start, ...] = HTK.readHtk_start_end3D(s, start-c_start, end-c_start)
                else:
                    if c_end < end:
                        video_fea[c_start-start:c_end-start, ...] = HTK.readHtk_start_end3D(s, 0, c_end-c_start)
                    else:
                        if c_start < end:
                            video_fea[c_start-start:end-start, ...] = HTK.readHtk_start_end3D(s, 0, end-c_start)
            if end > silence_start:
                video_fea[silence_start-start:end-start, ...] = self.get_slience_video(end - silence_start)
        #except:
        #    print(utt_name)
        #    return None
        return video_fea, utt_name

    def __len__(self):
        return len(self.feature_list)

def myCollateFn(sample_batch):
    sample_batch = [ x for x in sample_batch if x != None ]
    sample_batch = sorted(sample_batch, key=lambda x: x[0].shape[0], reverse=True)
    data_feature = [torch.from_numpy(x[0].astype(np.float32)) for x in sample_batch]
    utt_name = [x[1] for x in sample_batch]
    data_length = [x.shape[0]//1 for x in data_feature]
    data_feature = pad_sequence(data_feature, batch_first=True, padding_value=0.0)
    return data_feature, utt_name, data_length

class myDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super(myDataLoader, self).__init__(*args, **kwargs)
        self.collate_fn = myCollateFn
