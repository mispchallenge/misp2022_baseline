# -*- coding: utf-8 -*-

from email.mime import audio
import torch
import numpy as np
import os
import scipy.io as sio
import copy
import HTK
from torch.nn.utils.rnn import pad_sequence
import tqdm


def is_overlap(seg1, seg2):
    if seg1[1] <= seg2[0] or seg1[0] >= seg2[1]:
        return False
    else:
        return True


def collate_fn(sample_batch):
    '''
        returns:
        audio_fea: [T, F]
        audio_embedding: [num_speaker, video_embedding]
        video_fea: [num_speaker, T, 96, 96]
        video_label: [num_speaker, T, C]
        mask_label: [num_speaker, T, C]
        return audio_fea, audio_embedding, video_fea, video_label, mask_label
    '''
    # audio_fea, audio_embedding, video_embedding, mask_label
    batchsize = len(sample_batch)
    num_speaker, T, C = sample_batch[0][4].shape
    speaker_index = np.array(range(num_speaker))
    np.random.shuffle(speaker_index)
    sample_batch = sorted(sample_batch, key=lambda x: x[0].shape[0], reverse=True)
    audio_fea = [torch.from_numpy(x[0].astype(np.float32)) for x in sample_batch]
    audio_embedding = [torch.from_numpy(x[1][speaker_index].astype(np.float32)) for x in sample_batch]
    video_fea = [torch.from_numpy(x[2][speaker_index].astype(np.float32)) for x in sample_batch]
    video_label = [torch.from_numpy(x[3][speaker_index].astype(np.float32)) for x in sample_batch]
    mask_label = [torch.from_numpy(x[4][speaker_index].astype(np.float32)) for x in sample_batch]
    nframe = [x.shape[0]//1 for x in audio_fea]
    audio_fea = pad_sequence(audio_fea, batch_first=True, padding_value=0.0).transpose(1, 2)
    audio_embedding = torch.stack(audio_embedding)
    video_fea = pad_sequence(video_fea, batch_first=True, padding_value=0.0)
    video_label = torch.stack(video_label).reshape(batchsize*num_speaker*T//4, C)
    mask_label = torch.cat(mask_label, dim=1) # Time_Batch1 + Time_Batch2 + ... + Time_BatchN
    return audio_fea, audio_embedding, video_fea, video_label, mask_label, nframe


def decoder_collate_fn(sample_batch):
    #audio_fea, audio_embedding, video_fea, utt, num_speaker
    '''
        returns:
        audio_fea: [T, F]
        audio_embedding: [num_speaker, video_embedding]
        video_fea: [num_speaker, T, 96, 96]
        video_label: [num_speaker, T, C]
        mask_label: [num_speaker, T, C]
        return audio_fea, audio_embedding, video_fea, video_label, mask_label
    '''
    # audio_fea, audio_embedding, video_embedding, mask_label
    # print(len(sample_batch))
    audio_fea = [torch.from_numpy(x[0].astype(np.float32)) for x in sample_batch]
    audio_embedding = [torch.from_numpy(x[1].astype(np.float32)) for x in sample_batch]
    video_fea = [torch.from_numpy(x[2].astype(np.float32)) for x in sample_batch]
    utt = [x[3] for x in sample_batch]
    num_speaker = [x[4] for x in sample_batch]
    nframe = [x.shape[0]//1 for x in audio_fea]
    audio_fea = pad_sequence(audio_fea, batch_first=True, padding_value=0.0).transpose(1, 2)
    audio_embedding = torch.stack(audio_embedding)
    video_fea = pad_sequence(video_fea, batch_first=True, padding_value=0.0)
    return audio_fea, audio_embedding, video_fea, nframe, utt, num_speaker


class Audio_AEmbedding_Video_Worse_Data_Decode_Reader():
    def __init__(self, audio_dir, aembedding_scp, video_scp, train_aembedding_scp, train_video_scp, label, differ_speaker=True, min_speaker=0, max_speaker=6, max_utt_durance=800, frame_shift=None, discard_video=0.0):
        self.max_utt_durance = max_utt_durance
        if frame_shift == None:
            self.frame_shift = self.max_utt_durance // 2
        else:
            self.frame_shift = frame_shift
        self.differ_speaker = differ_speaker
        self.discard_video = discard_video
        self.label = label
        self.min_speaker = min_speaker
        self.max_speaker = max_speaker
        self.hasaembedding = False
        if aembedding_scp != "None":
            self.hasaembedding = True
            self.aembedding = LoadIVector(aembedding_scp)
        self.train_aembedding = LoadIVector(train_aembedding_scp)
        self.silent_video_segments, self.speech_video_segments = self.get_silent_speech_video_segments(train_video_scp)
        self.normal, self.worse_silence, self.worse_speech = 0, 0, 0
        self.feature_list, self.speakers = self.get_feature_info(audio_dir, video_scp)
        self.silent_video_segments_id = 0

    def get_silent_speech_video_segments(self, video_scp):
        silent_video_segments = []
        speech_video_segments = []
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
                    continue
                start = int(start)
                end = int(end) + 1
               # if(end - start < self.max_utt_durance):
                #    continue
                cur_frame = start
                #while cur_frame + self.max_utt_durance <= end and cur_frame + self.max_utt_durance <= session_length//4:
                    #len_speech = np.sum(self.label.frame_label[session][speaker][cur_frame*4:(cur_frame+self.max_utt_durance)*4, 1])
                    #if len_speech < 0.01 * self.max_utt_durance * 4:
                silent_video_segments.append((l.rstrip(), cur_frame-start, cur_frame+self.max_utt_durance-start))
                    #elif len_speech > 0.99 * self.max_utt_durance * 4:
                    #    speech_video_segments.append((l.rstrip(), cur_frame-start, cur_frame+self.max_utt_durance-start))
                #print(silent_video_segments)
                cur_frame += self.frame_shift
        print(f"silent video segments: {len(silent_video_segments)}")
        print(f"speech video segments: {len(speech_video_segments)}")
        np.random.shuffle(silent_video_segments)
        np.random.shuffle(speech_video_segments)
        return silent_video_segments, speech_video_segments

    def get_slience_video(self, durance):
        video_path, start, end = self.silent_video_segments[self.silent_video_segments_id]
        #print(video_path, start, end, durance)
        self.silent_video_segments_id += 1
        #print(self.silent_video_segments_id, len(self.silent_video_segments))
        if self.silent_video_segments_id == len(self.silent_video_segments):
            self.silent_video_segments_id = 0
            np.random.shuffle(self.silent_video_segments)
        if end-start-durance == 0:
            r_start = 0
        else:
            r_start = np.random.randint(0, end-start-durance)
        #print(r_start)
        silence_video_fea = HTK.readHtk_start_end3D(video_path, 0, 198)
        session = "_".join(os.path.basename(video_path).split("_")[:-2])
        speaker, _start, _ = os.path.basename(video_path).split("_")[-1].split(".")[0].split("-")
        _start = int(_start)
        silence_video_label = self.label.get_video_label_single_speaker(session, speaker, 0, 198)
        #print(f"{self.silent_video_segments[self.silent_video_segments_id]} {session} {speaker} {(_start+start+r_start)/25} {(_start+start+r_start+durance)/25} {np.sum(silence_video_label[:, 0])}")
        return silence_video_fea, silence_video_label

    def get_feature_info(self, audio_dir, video_scp):
        feature_list = []
        same_session_feature_list = {}
        MAX_DUR = {}
        speakers = {}
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
                    speakers[session] = []
                if speaker not in speakers[session]:
                    speakers[session].append(speaker)
                if speaker not in video_path[session].keys():
                    video_path[session][speaker] = []
                video_path[session][speaker].append([start, end, l.rstrip()])
        for session in video_path.keys():
            for speaker in video_path[session].keys():
                video_path[session][speaker].sort()
        if self.hasaembedding:
            for spk in self.aembedding.session2spk[session]:
                if spk not in speakers[session]:
                    speakers[session].append(spk)

        np.random.seed(42)
        sort_session = list(video_path.keys())
        sort_session.sort()
        for session in sort_session:
            try:
                MIN_LEN = HTK.readHtk_info(os.path.join(audio_dir, f"{session}.fea"))[0] // 4
            except:
                print(os.path.join(audio_dir, f"{session}.fea"))
                print(f"DataReader discarded {session}")
                continue
            same_session_feature_list[session] = []
            cur_frame = 0
            while cur_frame + self.max_utt_durance <= MIN_LEN:
                cur_video_path = {}
                sort_speaker = list(video_path[session].keys())
                sort_speaker.sort()
                for spk in sort_speaker:
                    cur_video_path[spk] = []
                    i = 0
                    while i < len(video_path[session][spk]) and cur_frame + self.max_utt_durance > video_path[session][spk][i][0]:
                        if is_overlap([cur_frame, cur_frame + self.max_utt_durance], video_path[session][spk][i][:2]):
                            cur_video_path[spk].append(video_path[session][spk][i][2])
                        i += 1
                    while video_path[session][spk] != [] and cur_frame + self.frame_shift > video_path[session][spk][0][1]:
                        video_path[session][spk] = video_path[session][spk][1:]
                    if np.random.uniform() < self.discard_video:
                        cur_video_path[spk] = []
                utt = f"{cur_frame}-{cur_frame+self.max_utt_durance}"
                same_session_feature_list[session].append([session, utt, cur_video_path, os.path.join(audio_dir, f"{session}.fea")])
                cur_frame += self.frame_shift
            feature_list.extend(same_session_feature_list[session])
        #total_item = self.normal + self.worse_speech + self.worse_silence
        #print(f"Normal:{self.normal / total_item} Set_silence:{self.worse_silence / total_item} Set_speak:{self.worse_speech / total_item}")
        return feature_list, speakers

    def load_fea(self, path, start, end):
        try:
            _, _, sampSize, _, data = HTK.readHtk_start_end(path, start, end)
        except:
            print("{} {} {}".format(path, start, end))
        htkdata= np.array(data).reshape(end - start, int(sampSize / 4))
        return end - start, htkdata

    def load_vembedding(self, path):
        nSamples, _, sampSize, _, data = HTK.readHtk(path)
        return np.array(data).reshape(nSamples, int(sampSize / 4))

    def __len__(self):
        return len(self.feature_list)
    
    def get_video_feature(self, session, speakers, start, end, sample_video_path):
        video_fea = np.zeros([self.max_speaker, end - start, 96, 96])
        #video_label = np.zeros([self.max_speaker, end - start, 2], dtype=np.int8)
        for id, spk in enumerate(speakers):
            if spk not in sample_video_path.keys():
                video_fea[id, :end - start, ...], _ = self.get_slience_video(end - start)
                continue
            cur_video_path = sample_video_path[spk]
            if cur_video_path == []:  #total silence
                video_fea[id, :end - start, ...],_ = self.get_slience_video(end - start)
            elif type(cur_video_path[0]) is not str:   #worse data
                #print(f"{session} {cur_video_path}")
                tmp_session = "_".join(os.path.basename(cur_video_path[0][0]).split("_")[:-2])
                tmp_speaker, tmp_start, _ = os.path.basename(cur_video_path[0][0]).split("_")[-1].split(".")[0].split("-")
                tmp_start = int(tmp_start)
                video_fea[id, :end - start, ...] = HTK.readHtk_start_end3D(cur_video_path[0][0], cur_video_path[0][1], cur_video_path[0][2])
                #video_label[id, :end - start, ...] = self.label.get_video_label_single_speaker(tmp_session, tmp_speaker, tmp_start+cur_video_path[0][1], tmp_start+cur_video_path[0][2])
            else:
                for i, s in enumerate(cur_video_path):
                    _, c_start, c_end = os.path.basename(s).split("_")[-1].split(".")[0].split("-")
                    c_start, c_end = int(c_start), int(c_end)+1
                    if i == 0:
                        silence_start = start
                    if c_start > silence_start:
                        silence, label = self.get_slience_video(c_start - silence_start)
                        video_fea[id, silence_start-start:c_start-start, ...] = silence
                        #video_label[id, silence_start-start:c_start-start, ...] = label
                    silence_start = c_end
                    if c_start < start:
                        if c_end < end:
                            video_fea[id, :c_end-start, ...] = HTK.readHtk_start_end3D(s, start-c_start, c_end-c_start)
                            #video_label[id, :c_end-start, ...] = self.label.get_video_label_single_speaker(session, spk, start, c_end)
                        else:
                            video_fea[id, :end-start, ...] = HTK.readHtk_start_end3D(s, start-c_start, end-c_start)
                            #video_label[id, :end-start, ...] = self.label.get_video_label_single_speaker(session, spk, start, end)
                    else:
                        if c_end < end:
                            video_fea[id, c_start-start:c_end-start, ...] = HTK.readHtk_start_end3D(s, 0, c_end-c_start)
                            #video_label[id, c_start-start:c_end-start, ...] = self.label.get_video_label_single_speaker(session, spk, c_start, c_end)
                        elif c_start < end:
                            video_fea[id, c_start-start:end-start, ...] = HTK.readHtk_start_end3D(s, 0, end-c_start)
                            #video_label[id, c_start-start:end-start, ...] = self.label.get_video_label_single_speaker(session, spk, c_start, end)
                if end > silence_start:
                    video_fea[id, silence_start-start:end-start, ...],_ = self.get_slience_video(end - silence_start)
        num_speaker = len(speakers)
        for l in range(num_speaker, self.max_speaker):
            video_fea[l, :end - start, ...], _ = self.get_slience_video(end - start)
        return video_fea

    def __getitem__(self, idx):
        #[session, utt, cur_video_path, os.path.join(audio_dir, "misp2021_train_middle_raw_cmn_slide", f"{session}_Middle_0_RAW.fea")]
        session, utt, sample_video_path, audio_path = self.feature_list[idx]
        real_session = os.path.basename(audio_path).split(".")[0]
        start, end = utt.split('-')
        start, end = int(start), int(end)
        _, audio_fea = self.load_fea(audio_path, start*4, end*4)
        utt = f"{session}-{start*4}-{end*4}"
        #print(utt)
        #mask_label, speakers = self.label.get_mixture_utterance_label(session, start=start*4, end=end*4)
        #num_speaker, T, C = mask_label.shape
        num_speaker = len(self.speakers[session]) + 0
        video_fea = self.get_video_feature(session, self.speakers[session], start, end, sample_video_path)
        audio_embedding = []
        for spk in self.speakers[session]:
            try:
                audio_embedding.append(self.aembedding.get_speaker_embedding("{}-{}".format(real_session, spk)))
            except:
                data_set_speaker = list(self.train_aembedding.speaker_embedding.keys())
                audio_embedding.append(self.train_aembedding.get_speaker_embedding(np.random.choice(data_set_speaker)))

        if num_speaker < self.max_speaker:
            data_set_speaker = list(self.train_aembedding.speaker_embedding.keys())
            for speaker in np.random.choice(data_set_speaker, self.max_speaker - num_speaker, replace=True):
                audio_embedding.append(self.train_aembedding.get_speaker_embedding(speaker))
        audio_embedding = np.stack(audio_embedding)
        #audio_fea, audio_embedding, video_embedding, utt, num_speaker
        #print(f"{len(audio_fea)} {len(video_fea)}")
        '''
        returns:
        audio_fea: [T, F]
        audio_embedding: [num_speaker, video_embedding]
        video_fea: [num_speaker, T, 96, 96]
        video_label: [num_speaker, T, C]
        mask_label: [num_speaker, T, C]
        '''
        return audio_fea, audio_embedding, video_fea, utt, num_speaker

class Audio_AEmbedding_Video_Worse_Data_Reader():
    def __init__(self, audio_dir, aembedding_scp, video_scp, label, set_video_silence=0, set_video_speak=0, differ_speaker=True, min_speaker=0, max_speaker=6, max_utt_durance=800, frame_shift=None, mixup_rate=0, alpha=0.5):
        self.max_utt_durance = max_utt_durance
        if frame_shift == None:
            self.frame_shift = self.max_utt_durance // 2
        else:
            self.frame_shift = frame_shift
        self.differ_speaker = differ_speaker
        self.label = label
        self.set_video_silence = set_video_silence
        self.set_video_speak = set_video_speak
        self.min_speaker = min_speaker
        self.max_speaker = max_speaker
        self.mixup_rate = mixup_rate
        self.alpha = alpha
        self.speaker_embedding = LoadIVector(aembedding_scp)
        self.silent_video_segments, self.speech_video_segments = self.get_silent_speech_video_segments(video_scp)
        self.normal, self.worse_silence, self.worse_speech = 0, 0, 0
        self.same_session_feature_list, self.feature_list = self.get_feature_info(audio_dir, video_scp)
        self.silent_video_segments_id = 0

    def get_silent_speech_video_segments(self, video_scp):
        silent_video_segments = []
        speech_video_segments = []
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
                    continue
                start = int(start)
                end = int(end) + 1
                if(end - start < self.max_utt_durance):
                    continue
                cur_frame = start
                while cur_frame + self.max_utt_durance <= end and cur_frame + self.max_utt_durance <= session_length//4:
                    len_speech = np.sum(self.label.frame_label[session][speaker][cur_frame*4:(cur_frame+self.max_utt_durance)*4, 1])
                    if len_speech < 0.01 * self.max_utt_durance * 4:
                        silent_video_segments.append((l.rstrip(), cur_frame-start, cur_frame+self.max_utt_durance-start))
                    elif len_speech > 0.90 * self.max_utt_durance * 4:
                        speech_video_segments.append((l.rstrip(), cur_frame-start, cur_frame+self.max_utt_durance-start))
                    cur_frame += self.frame_shift
        print(f"silent video segments: {len(silent_video_segments)}")
        print(f"speech video segments: {len(speech_video_segments)}")
        np.random.shuffle(silent_video_segments)
        np.random.shuffle(speech_video_segments)
        return silent_video_segments, speech_video_segments

    def get_slience_video(self, durance):
        video_path, start, end = self.silent_video_segments[self.silent_video_segments_id]
        self.silent_video_segments_id += 1
        if self.silent_video_segments_id == len(self.silent_video_segments):
            self.silent_video_segments_id = 0
            np.random.shuffle(self.silent_video_segments)
        if end-start-durance == 0:
            r_start = 0
        else:
            r_start = np.random.randint(0, end-start-durance)
        silence_video_fea = HTK.readHtk_start_end3D(video_path, r_start+start, r_start+start+durance)
        session = "_".join(os.path.basename(video_path).split("_")[:-2])
        speaker, _start, _ = os.path.basename(video_path).split("_")[-1].split(".")[0].split("-")
        _start = int(_start)
        silence_video_label = self.label.get_video_label_single_speaker(session, speaker, _start+start+r_start, _start+start+r_start+durance)
        #print(f"{self.silent_video_segments[self.silent_video_segments_id]} {session} {speaker} {(_start+start+r_start)/25} {(_start+start+r_start+durance)/25} {np.sum(silence_video_label[:, 0])}")
        return silence_video_fea, silence_video_label

    def worse_data(self, item):
        #item = [session, utt, cur_video_path, os.path.join(audio_dir, "misp2021_train_middle_raw_cmn_slide", f"{session}_Middle_0_RAW.fea")]
        session, utt, video_fea, audio_path = item
        start, end = utt.split('-')
        start, end = int(start), int(end)
        choose = np.random.uniform()
        #print(choose)
        if choose < self.set_video_silence:
            # worse video
            max_speech = 0
            label, speakers = self.label.get_mixture_utterance_label(session, start*4, end*4)
            for i, spk in enumerate(speakers):
                if max_speech < np.sum(label[i][:, 1]):
                    max_speech = np.sum(label[i][:, 1])
                    speaker = spk
            #print(f"{session}-{speaker}-{max_speech}")
            if max_speech >= 0.8 * (end - start):
                video_fea[speaker] = [self.silent_video_segments[self.worse_silence % int(len(self.silent_video_segments))]]
                self.worse_silence += 1
                return [session, utt, video_fea, audio_path]

        elif choose < self.set_video_silence + self.set_video_speak:
            min_speech = np.inf
            label, speakers = self.label.get_mixture_utterance_label(session, start*4, end*4)
            for i, spk in enumerate(speakers):
                if min_speech > np.sum(label[i][:, 1]):
                    min_speech = np.sum(label[i][:, 1])
                    speaker = spk
            if min_speech <= 0.2 * (end - start):
                video_fea[speaker] = [self.speech_video_segments[self.worse_speech % int(len(self.speech_video_segments))]]
                self.worse_speech += 1
                return [session, utt, video_fea, audio_path]
        self.normal += 1
        return item

    def get_feature_info(self, audio_dir, video_scp):
        feature_list = []
        same_session_feature_list = {}
        MAX_DUR = {}
        speakers = {}
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
                    speakers[session] = []
                if speaker not in speakers[session]:
                    speakers[session].append(speaker)
                if speaker not in video_path[session].keys():
                    video_path[session][speaker] = []
                video_path[session][speaker].append([start, end, l.rstrip()])
        for session in video_path.keys():
            for speaker in video_path[session].keys():
                video_path[session][speaker].sort()

        for session in video_path.keys():
            try:
                total_frame0 = HTK.readHtk_info(os.path.join(audio_dir, "misp2021_train_middle_raw_cmn_slide", f"{session}_Middle_0_RAW.fea"))[0] // 4 
                total_frame1 = HTK.readHtk_info(os.path.join(audio_dir, "misp2021_train_middle_raw_cmn_slide", f"{session}_Middle_1_RAW.fea"))[0] // 4
                total_frame2 = HTK.readHtk_info(os.path.join(audio_dir, "misp2021_train_middle_wpe_cmn_slide", f"{session}_Middle_0_WPE.fea"))[0] // 4
                total_frame3 = HTK.readHtk_info(os.path.join(audio_dir, "misp2021_train_middle_wpe_cmn_slide", f"{session}_Middle_1_WPE.fea"))[0] // 4
                session_length = self.label.get_session_length(session) // 4
            except:
                print(f"DataReader discarded {session}")
                continue
            num_speaker = int(len(speakers[session]))
            if num_speaker != self.label.mixture_num_speaker(session):
                print(f"{session} video speaker number ({num_speaker}) not equal oracle speaker num ({self.label.mixture_num_speaker(session)}) speakers[session]({speakers[session]})")
                continue
            MIN_LEN = min(total_frame0, total_frame1, total_frame2, total_frame3, session_length, MAX_DUR[session])
            same_session_feature_list[session] = []
            cur_frame = 0
            while cur_frame + self.max_utt_durance <= MIN_LEN:
                cur_video_path = {}
                for spk in video_path[session].keys():
                    cur_video_path[spk] = []
                    i = 0
                    while i < len(video_path[session][spk]) and cur_frame + self.max_utt_durance > video_path[session][spk][i][0]:
                        if is_overlap([cur_frame, cur_frame + self.max_utt_durance], video_path[session][spk][i][:2]):
                            cur_video_path[spk].append(video_path[session][spk][i][2])
                        i += 1
                    while video_path[session][spk] != [] and cur_frame + self.frame_shift > video_path[session][spk][0][1]:
                        video_path[session][spk] = video_path[session][spk][1:]
                utt = f"{cur_frame}-{cur_frame+self.max_utt_durance}"
                same_session_feature_list[session].append(self.worse_data([session, utt, cur_video_path, os.path.join(audio_dir, "misp2021_train_middle_raw_cmn_slide", f"{session}_Middle_0_RAW.fea")]))
                same_session_feature_list[session].append(self.worse_data([session, utt, cur_video_path, os.path.join(audio_dir, "misp2021_train_middle_raw_cmn_slide", f"{session}_Middle_1_RAW.fea")]))
                same_session_feature_list[session].append(self.worse_data([session, utt, cur_video_path, os.path.join(audio_dir, "misp2021_train_middle_wpe_cmn_slide", f"{session}_Middle_0_WPE.fea")]))
                same_session_feature_list[session].append(self.worse_data([session, utt, cur_video_path, os.path.join(audio_dir, "misp2021_train_middle_wpe_cmn_slide", f"{session}_Middle_1_WPE.fea")]))
                cur_frame += self.frame_shift
            feature_list.extend(same_session_feature_list[session])
        total_item = self.normal + self.worse_speech + self.worse_silence
        print(f"Normal:{self.normal / total_item} Set_silence:{self.worse_silence / total_item} Set_speak:{self.worse_speech / total_item}")
        return same_session_feature_list, feature_list

    def load_fea(self, path, start, end):
        try:
            _, _, sampSize, _, data = HTK.readHtk_start_end(path, start, end)
        except:
            print("{} {} {}".format(path, start, end))
        htkdata= np.array(data).reshape(end - start, int(sampSize / 4))
        return end - start, htkdata

    def load_vembedding(self, path):
        nSamples, _, sampSize, _, data = HTK.readHtk(path)
        return np.array(data).reshape(nSamples, int(sampSize / 4))

    def __len__(self):
        return len(self.feature_list)
    
    def get_video_feature(self, session, speakers, start, end, sample_video_path):
        video_fea = np.zeros([self.max_speaker, end - start, 96, 96])
        video_label = np.zeros([self.max_speaker, end - start, 2], dtype=np.int8)
        for id, spk in enumerate(speakers):
            cur_video_path = sample_video_path[spk]
            if cur_video_path == []:  #total silence
                video_fea[id, :end - start, ...], video_label[id, :end - start, ...] = self.get_slience_video(end - start)
            elif type(cur_video_path[0]) is not str:   #worse data
                #print(f"{session} {cur_video_path}")
                tmp_session = "_".join(os.path.basename(cur_video_path[0][0]).split("_")[:-2])
                tmp_speaker, tmp_start, _ = os.path.basename(cur_video_path[0][0]).split("_")[-1].split(".")[0].split("-")
                tmp_start = int(tmp_start)
                video_fea[id, :end - start, ...] = HTK.readHtk_start_end3D(cur_video_path[0][0], cur_video_path[0][1], cur_video_path[0][2])
                video_label[id, :end - start, ...] = self.label.get_video_label_single_speaker(tmp_session, tmp_speaker, tmp_start+cur_video_path[0][1], tmp_start+cur_video_path[0][2])
            else:
                for i, s in enumerate(cur_video_path):
                    _, c_start, c_end = os.path.basename(s).split("_")[-1].split(".")[0].split("-")
                    c_start, c_end = int(c_start), int(c_end)+1
                    if i == 0:
                        silence_start = start
                    if c_start > silence_start:
                        silence, label = self.get_slience_video(c_start - silence_start)
                        video_fea[id, silence_start-start:c_start-start, ...] = silence
                        video_label[id, silence_start-start:c_start-start, ...] = label
                    silence_start = c_end
                    if c_start < start:
                        if c_end < end:
                            video_fea[id, :c_end-start, ...] = HTK.readHtk_start_end3D(s, start-c_start, c_end-c_start)
                            video_label[id, :c_end-start, ...] = self.label.get_video_label_single_speaker(session, spk, start, c_end)
                        else:
                            video_fea[id, :end-start, ...] = HTK.readHtk_start_end3D(s, start-c_start, end-c_start)
                            video_label[id, :end-start, ...] = self.label.get_video_label_single_speaker(session, spk, start, end)
                    else:
                        if c_end < end:
                            video_fea[id, c_start-start:c_end-start, ...] = HTK.readHtk_start_end3D(s, 0, c_end-c_start)
                            video_label[id, c_start-start:c_end-start, ...] = self.label.get_video_label_single_speaker(session, spk, c_start, c_end)
                        elif c_start < end:
                            video_fea[id, c_start-start:end-start, ...] = HTK.readHtk_start_end3D(s, 0, end-c_start)
                            video_label[id, c_start-start:end-start, ...] = self.label.get_video_label_single_speaker(session, spk, c_start, end)
                if end > silence_start:
                    video_fea[id, silence_start-start:end-start, ...], video_label[id, silence_start-start:end-start, ...] = self.get_slience_video(end - silence_start)
        num_speaker = len(speakers)
        for l in range(num_speaker, self.max_speaker):
            video_fea[l, :end - start, ...], video_label[l, :end - start, ...] = self.get_slience_video(end - start)
        return video_fea, video_label

    def __getitem__(self, idx):
        #[session, utt, cur_video_path, os.path.join(audio_dir, "misp2021_train_middle_raw_cmn_slide", f"{session}_Middle_0_RAW.fea")]
        session, utt, sample_video_path, audio_path = self.feature_list[idx]
        real_session = os.path.basename(audio_path).split(".")[0]
        start, end = utt.split('-')
        start, end = int(start), int(end)
        _, audio_fea = self.load_fea(audio_path, start*4, end*4)
        mask_label, speakers = self.label.get_mixture_utterance_label(session, start=start*4, end=end*4)
        num_speaker, T, C = mask_label.shape
        
        video_fea, video_label = self.get_video_feature(session, speakers, start, end, sample_video_path)
        audio_embedding = []
        for spk in speakers:
            audio_embedding.append(self.speaker_embedding.get_speaker_embedding("{}-{}".format(real_session, spk)))
        if num_speaker < self.max_speaker:
            data_set_speaker = list(self.speaker_embedding.speaker_embedding.keys())
            for speaker in speakers:
                try:
                    if not self.differ_speaker:
                        data_set_speaker.remove("{}-{}".format(session, speaker))
                    else:
                        for spk in self.speaker_embedding.spk2sessionspk[speaker]:
                            data_set_speaker.remove(spk)
                except:
                    print(speaker)
            for speaker in np.random.choice(data_set_speaker, self.max_speaker - num_speaker, replace=False):
                audio_embedding.append(self.speaker_embedding.get_speaker_embedding(speaker))
        audio_embedding = np.stack(audio_embedding)

        if num_speaker < self.max_speaker:
            append_label = np.zeros([self.max_speaker - num_speaker, T, C])
            append_label[:, :, 0] = 1
            mask_label = np.vstack([mask_label, append_label])

        if num_speaker > self.max_speaker:
            speaker_index = np.array(range(num_speaker))
            np.random.shuffle(speaker_index)
            audio_embedding = audio_embedding[speaker_index][:self.max_speaker]
            mask_label = mask_label[speaker_index][:self.max_speaker]
        '''
        returns:
        audio_fea: [T, F]
        audio_embedding: [num_speaker, video_embedding]
        video_fea: [num_speaker, T, 96, 96]
        video_label: [num_speaker, T, C]
        mask_label: [num_speaker, T, C]
        '''
        return audio_fea, audio_embedding, video_fea, video_label, mask_label

class Label_Generate_From_RTTM():
    def __init__(self, oracle_rttm, differ_silence_inference_speech=False, max_speaker=8):
        self.differ_silence_inference_speech = differ_silence_inference_speech
        self.frame_label, self.frame_label_video = self.get_label(oracle_rttm)
        self.max_speaker = max_speaker

    def get_label(self, oracle_rttm):
        '''
        SPEAKER session0_CH0_0L 1  116.38    3.02 <NA> <NA> 5683 <NA>
        '''
        files = open(oracle_rttm)
        MAX_len = {}
        rttm = {}
        rttm_video = {}
        for line in files:
            line = line.split(" ")
            session = line[1]
            if not session in MAX_len.keys():
                MAX_len[session] = 0
            start = np.int(np.float(line[3]) * 100)
            end = np.int(np.float(line[4]) * 100) + start
            if end > MAX_len[session]:
                MAX_len[session] = end + 800
        files.close()
        files = open(oracle_rttm)
        for line in files:
            line = line.split(" ")
            session = line[1]
            spk = line[-3]
            if not session in rttm.keys():
                rttm[session] = {}
                rttm_video[session] = {}
            if not spk in rttm[session].keys():
                rttm[session][spk] = np.zeros([MAX_len[session], 2], dtype=np.int8)
                rttm_video[session][spk] = np.zeros([MAX_len[session]//4, 2], dtype=np.int8)
            start = np.int(np.float(line[3]) * 100)
            end = np.int(np.float(line[4]) * 100) + start
            rttm[session][spk][start: end, 1] = 1
            rttm_video[session][spk][start//4: end//4, 1] = 1
        for session in rttm.keys():
            for spk in rttm[session].keys():
                rttm[session][spk][:, 0] = 1 - rttm[session][spk][:, 1]
                rttm_video[session][spk][:, 0] = 1 - rttm_video[session][spk][:, 1]
        files.close()
        return rttm, rttm_video
            
    def mixture_num_speaker(self, session):
        return len(self.frame_label[session])

    def get_session_length(self, session):
        for spk in self.frame_label[session].keys():
            return len(self.frame_label[session][spk])
    
    def get_mixture_utterance_label(self, session, start=0, end=None):
        speakers = []
        mixture_utternce_label = []
        for spk in self.frame_label[session].keys():
            speakers.append(spk)
            mixture_utternce_label.append(self.frame_label[session][spk][start:end, :])
        return np.vstack(mixture_utternce_label).reshape(len(speakers), end - start, -1), speakers

    def get_video_label_single_speaker(self, session, speaker, start=0, end=None):
        try:
            return self.frame_label_video[session][speaker][start: end, :]
        except:
            print("{} {} not in labels! return []".format(session, speaker))
            return []

class LoadIVector():
    def __init__(self, speaker_embedding_txt):
        self.speaker_embedding, self.spk2sessionspk, self.session2spk = self.load_ivector(speaker_embedding_txt)

    def load_ivector(self, speaker_embedding_txt):
        SCP_IO = open(speaker_embedding_txt)
        speaker_embedding = {}
        raw_lines = [l for l in SCP_IO]
        SCP_IO.close()
        already_speaker_list = []
        spk2sessionspk = {}
        session2spk = {}
        for i in range(len(raw_lines) // 2):
            speaker = raw_lines[2*i].split()[0]
            session, real_speaker = speaker.split('-')
            if session not in session2spk.keys():
                session2spk[session] = []
            if real_speaker not in session2spk[session]:
                session2spk[session].append(real_speaker)
            if real_speaker not in spk2sessionspk.keys():
                spk2sessionspk[real_speaker] = []
            spk2sessionspk[real_speaker].append(speaker)
            if speaker not in already_speaker_list:
                ivector = np.array(raw_lines[2*i+1].split()[:-1], np.float32)
                speaker_embedding[speaker] = ivector
                already_speaker_list.append(speaker)
        return speaker_embedding, spk2sessionspk, session2spk

    def get_speaker_embedding(self, speaker):
        if not speaker in self.speaker_embedding.keys():
            print("{} not in sepaker embedding list".format(speaker))
            exit()
        return self.speaker_embedding[speaker]