# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import numpy as np
import sys

sys.path.append(".")
sys.path.append("..")

from .extract_lip_embedding_resnet import VideoFrontend
from .conformer import ConformerBlock
from .vsd_net import Visual_VAD_Conformer_Net

class LSTM_Projection(nn.Module):
    def __init__(self, input_size, hidden_size, linear_dim, num_layers=1, bidirectional=True, dropout=0):
        super(LSTM_Projection, self).__init__()
        self.hidden_size = hidden_size
        self.LSTM = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, bidirectional=bidirectional, dropout=dropout)
        self.forward_projection = nn.Linear(hidden_size, linear_dim)
        self.backward_projection = nn.Linear(hidden_size, linear_dim)
        self.relu = nn.ReLU(True)

    def forward(self, x, nframes):
        '''
        x: [batchsize, Time, Freq]
        nframes: [len_b1, len_b2, ..., len_bN]
        '''
        packed_x = nn.utils.rnn.pack_padded_sequence(x, nframes, batch_first=True)
        packed_x_1, hidden = self.LSTM(packed_x)
        x_1, l = nn.utils.rnn.pad_packed_sequence(packed_x_1, batch_first=True)
        forward_projection = self.relu(self.forward_projection(x_1[..., :self.hidden_size]))
        backward_projection = self.relu(self.backward_projection(x_1[..., self.hidden_size:]))
        # x_2: [batchsize, Time, linear_dim*2]
        x_2 = torch.cat((forward_projection, backward_projection), dim=2)
        return x_2


class CNN2D_BN_Relu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(CNN2D_BN_Relu, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=kernel_size//2)
        self.bn = nn.BatchNorm2d(out_channels) #(N,C,H,W) on C
        self.relu = nn.ReLU(True)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        return out

class AIVECTOR_ConformerVEmbedding_SD(nn.Module):
    '''
    INPUT (MFCC)
    IDCT: MFCC to FBANK
    Batchnorm
    Stats pooling: batchnorm-cmn
    Combine inputs:  (batchnorm, batchnorm-cmn Speaker Detection Block
    4 layers CNN: Conv2d (in channels, out channels, kernel size, stride=1, padding=0 dilation=1, groups=1, bias=True, padding mode=zeros) 1 layer Splice-embedding:  (Convld SD, ivector-k) Linear layer
    2 layers Shared-blstm
    Attention
    1 layers CNN: Convld (in channels, out channels, kernel size, stride=1, padding=0 dilation=1, groups=1, bias=True, padding mode=zeros) Attention layer
    1 layers Combine-speaker
    1 layers BLSTM
    1 layers Dense: 4 dependent FC layers
    '''
    def __init__(self, configs):
        super(AIVECTOR_ConformerVEmbedding_SD, self).__init__()
        self.input_size = configs["input_dim"]
        self.cnn_configs = configs["cnn_configs"]
        self.speaker_embedding_size = configs["speaker_embedding_dim"]
        self.Linear_dim = configs["Linear_dim"]
        self.Shared_BLSTM_size = configs["Shared_BLSTM_dim"]
        self.Linear_Shared_layer1_dim = configs["Linear_Shared_layer1_dim"]
        self.Linear_Shared_layer2_dim = configs["Linear_Shared_layer2_dim"]
        #self.attention_hidden_dim = configs["attention_hidden_dim"]
        self.BLSTM_size = configs["BLSTM_dim"]
        self.BLSTM_Projection_dim = configs["BLSTM_Projection_dim"]
        self.output_size = configs["output_dim"]
        self.output_speaker = configs["output_speaker"]

        #self.idct = torch.from_numpy(np.load('./dataset/idct.npy').astype(np.float32)).cuda()  #if input is mfcc
        self.batchnorm = nn.BatchNorm2d(1)
        self.average_pooling = nn.AvgPool1d(configs["average_pooling"], stride=1, padding=configs["average_pooling"]//2)
        # Speaker Detection Block
        self.Conv2d_SD = nn.Sequential()
        self.Conv2d_SD.add_module('CNN2D_BN_Relu_SD1', CNN2D_BN_Relu(self.cnn_configs[0][0], self.cnn_configs[0][1], self.cnn_configs[0][2], self.cnn_configs[0][3]))
        self.Conv2d_SD.add_module('CNN2D_BN_Relu_SD2', CNN2D_BN_Relu(self.cnn_configs[1][0], self.cnn_configs[1][1], self.cnn_configs[1][2], self.cnn_configs[1][3]))
        self.Conv2d_SD.add_module('CNN2D_BN_Relu_SD3', CNN2D_BN_Relu(self.cnn_configs[2][0], self.cnn_configs[2][1], self.cnn_configs[2][2], self.cnn_configs[2][3]))
        self.Conv2d_SD.add_module('CNN2D_BN_Relu_SD4', CNN2D_BN_Relu(self.cnn_configs[3][0], self.cnn_configs[3][1], self.cnn_configs[3][2], self.cnn_configs[3][3]))

        self.splice_size = configs["splice_size"]
        self.Linear = nn.Linear(self.splice_size, self.Linear_dim)
        self.relu = nn.ReLU(True)
        self.Shared_BLSTMP_1 = LSTM_Projection(input_size=self.Linear_dim, hidden_size=self.Shared_BLSTM_size, linear_dim=self.Linear_Shared_layer1_dim, num_layers=1, bidirectional=True, dropout=0)
        self.Shared_BLSTMP_2 = LSTM_Projection(input_size=self.Linear_Shared_layer1_dim*2, hidden_size=self.Shared_BLSTM_size, linear_dim=self.Linear_Shared_layer2_dim, num_layers=1, bidirectional=True, dropout=0)

        self.conbine_speaker_size = self.Linear_Shared_layer2_dim * 2 * self.output_speaker
        self.BLSTMP = LSTM_Projection(input_size=self.conbine_speaker_size, hidden_size=self.BLSTM_size, linear_dim=self.BLSTM_Projection_dim, num_layers=1, bidirectional=True, dropout=0)
        '''
        if configs["dropout"] > 0:
            self.dropout = True
            self.Dropout = nn.Dropout(configs["dropout"])
        else:
            self.dropout = False
        '''
        FC = {}
        for i in range(self.output_speaker):
            FC[str(i)] = nn.Linear(self.BLSTM_Projection_dim*2, self.output_size)
        self.FC = nn.ModuleDict(FC)


    def forward(self, x, audio_embedding, video_embedding, nframes):
        '''
        x: Batch * Freq * Time
        audio_embedding : Batch * speaker(4) * Embedding
        video_embedding : Batch * Speaker * Time * Embedding
        nframe: descend order
        '''
        #print(x.shape)
        #print(embedding.shape)
        batchsize, Freq, Time = x.shape
        speaker, audio_embedding_dim = audio_embedding.shape[1:]
        video_embedding_dim = video_embedding.shape[-1]

        if type(nframes) == torch.Tensor:
            nframes = list(nframes.detach().cpu().numpy())
        # ****************IDCT: MFCC to FBANK*****************
        # [batchsize, Freq, Time] -> [Freq, batchsize, Time] -> [Freq, -1] 
        #x_1 = x.transpose(0, 1).reshape(Freq, -1)
        # MFCC to FBANK
        #x_2 = self.idct.mm(x_1).reshape(Freq, batchsize, Time).transpose(0, 1)

        # ************batchnorm statspooling*****************
        #batchnorm [batchsize, Freq, Time] -> [ batchsize, 1, Freq, Time]
        x_3 = self.batchnorm(x.reshape(batchsize, 1, Freq, Time)).squeeze(dim=1)
        #batchnorm_cmn
        #w = Time / torch.Tensor (nframes)
        #x_3_mean = torch.mean(x_3, dim=3) * w[:, None, None ]
        x_3_mean = self.average_pooling(x_3)
        #(batchnorm, batchnorm_cmn): 2 * [batchsize, Freq, Time] -> [batchs ize, 2, Freq, Time]
        x_4 = torch.cat((x_3, x_3_mean), dim=1).reshape(batchsize, 2, Freq, Time)
        #(batchnorm, batchnorm_cmn) -> (bn_d0, bnc_d0, bn_d1, bnc_d1,..., bn_d40, bnc_d40)

        # **************CNN*************
        # [batchsize, 2ï¼Freq, Time] -ã[batchsize, Conv-4-out-filters, Freq, Time]
        x_5 = self.Conv2d_SD(x_4)
        #print(x_5.shape)
        #[batchsize, Conv-4-out-filters, Freq, Time] -ã [ batchsize, Conv-4-out-filters*Freq, Time ]
        x_6 = x_5.reshape(batchsize, -1, Time)
        Freq = x_6.shape[1]
        #*********************************************need to check***************** ***
        #print(x_1.repeat(1, speaker, 1).shape)
        x_6_reshape = x_6.repeat(1, speaker, 1).reshape(batchsize * speaker, Freq, Time)
        
        #audio_embedding: Batch * speaker * Embedding -> (Batch * speaker) * Embedding * Time
        audio_embedding_reshape = audio_embedding.reshape(-1, audio_embedding_dim)[..., None].expand(batchsize * speaker, audio_embedding_dim, Time)
        #
        video_embedding_reshape = video_embedding.reshape(batchsize * speaker, Time // 4, video_embedding_dim).repeat_interleave(4, dim=1).permute(0, 2, 1)

        #print(embedding_reshape.shape).repeat_interleave(4, dim=1).permute(0, 2, 1) #(B*Speaker, -1, AT)
        x_7 = torch.cat((x_6_reshape, audio_embedding_reshape, video_embedding_reshape), dim=1)
        '''
        x_7: (Batch * speaker) * (Freq + Embedding) * Time
        '''
        #**************Linear*************
        #(Batch * speaker) * (Freq + Embedding) * Time -ã(Batch * speaker) * Time * Linear_dim
        x_8 = self.relu(self.Linear(x_7.transpose(1, 2)))
        #Shared_BLSTMP_1 (Batch * speaker) * Time * Linear_dim =ã(Batch * speaker) * Time * (Linear_Shared_layer1_dim * 2)
        lens = [n for n in nframes for i in range(speaker)] 
        x_9 = self.Shared_BLSTMP_1(x_8, lens)
        #Shared_BLSTMP_2 (Batch * speaker) * Time * (Linear_Shared_layer1_dim * 2) =ã (Batch * speaker) * Time * (Linear_Shared_layer2_dim * 2)
        x_10 = self.Shared_BLSTMP_2(x_9, lens)

        #Combine-Speaker: (Batch * speaker) * Time * (Linear_Shared_layer2_dim * 2) => Batch * Time * (speaker * Linear_Shared_layer2_dim * 2)
        x_11 = x_10.reshape(batchsize, speaker, Time, -1).transpose(1, 2).reshape(batchsize, Time, -1)
        '''
        batchsize * Time * (1stspeaker 2ndspeaker 3rdspeaker 4thspeaker)
        '''

        #BLSTM: Batch * Time * (speaker * Linear_Shared_layer2_dim * 2) => Batch * Time * (BLSTM_Projection_dim * 2)
        x_12 = self.BLSTMP(x_11, nframes)

        #Dimension reduction: remove the padding frames; Batch * Time * (BLSTM_Projection_dim * 2) => (Batch * Time) * (BLSTM_Projection_dim * 2)
        lens = [k for i, m in enumerate(nframes) for k in range(i * Time, m + i * Time)]
        x_13 = x_12.reshape(batchsize * Time, -1)[lens, :]
        '''
        1stBatch(1st sentence) length1 * (BLSTM_size * 2)
        2ndBatch(2nd sentence) length2 * (BLSTM_size * 2)
        ...
        '''
        #if self.dropout:
        #    x_13 = self.Dropout(x_13)
            
        out = []
        for i in self.FC:
            out.append(self.FC[i](x_13))
        return out



class AIVECTOR_ConformerVEmbedding_SD_JOINT(nn.Module):
    def __init__(self, configs):
        super(AIVECTOR_ConformerVEmbedding_SD_JOINT, self).__init__()

        self.v_embedding = Visual_VAD_Conformer_Net()

        self.av_sd = AIVECTOR_ConformerVEmbedding_SD(configs)

    def forward(self, audio_fea, audio_embedding, video_fea, nframes):
        '''
        x: Batch * Freq * Time
        audio_embedding : Batch * speaker(4) * Embedding
        video_embedding : Batch * Speaker * Time * Embedding
        nframe: descend order
        '''
        '''
        returns:
        audio_fea: [B, F, T]
        audio_embedding: [B, num_speaker, video_embedding]
        video_fea: [B, num_speaker, T, 96, 96]
        video_label: [B*num_speaker*T, C]
        mask_label: [B*num_speaker*T, C]
        '''
        B, num_speaker, T, _, _ = video_fea.shape
        if type(nframes) == torch.Tensor:
            nframes = list(nframes.detach().cpu().numpy())
        v_nframes = [ k // 4 for i in range(num_speaker) for k in nframes ]
        v_nframes = torch.from_numpy(np.array(v_nframes, dtype=np.int32))
        video_fea = video_fea.reshape(B*num_speaker, T, 96, 96)
        v_out, v_embedding = self.v_embedding(video_fea, v_nframes, return_embedding=True)
        v_embedding = v_embedding.reshape(B, num_speaker, T, -1)
        av_out = self.av_sd(audio_fea, audio_embedding, v_embedding, nframes)

        return v_out, av_out


if __name__=='__main__':

    import config
    model = A_ConformerVEmbedding_ConformerDeocoding_SD(config.configs3_6Speakers_ConformerVEmbedding_ConformerDeocoding_2Classes)
    '''
    self.Encoder_Audio = nn.Sequential()
    self.Encoder_Audio.add_module('Encoder_Audio_Linear', nn.Linear(configs["input_dim"], configs["encoder_conformer"]["dim"]))
    self.Encoder_Audio.add_module('Encoder_Audio_ReLU', nn.ReLU())
    for i in range(configs["encoder_conformer_layers"]):
        self.Encoder_Audio.add_module(f'Encoder_Audio_Conformer{i}', self.make_conformer_block(configs["encoder_conformer"]))
    
    self.Combine_FC = nn.Sequential()
    self.Combine_FC.add_module('Combine_FC_Linear', nn.Linear(configs["encoder_conformer"]["dim"]+configs["speaker_embedding"], configs["share_conformer"]["dim"]))
    self.Combine_FC.add_module('Combine_FC_ReLU', nn.ReLU())

    self.Share_Conformer = nn.Sequential()
    for i in range(configs["share_conformer_layers"]):
        self.Share_Conformer.add_module(f'Share_Conformer{i}', self.make_conformer_block(configs["share_conformer"]))

    conbine_speaker_size = configs["share_conformer"]["dim"] * configs["num_speaker"]
    self.combine_BLSTMP = LSTM_Projection(input_size=conbine_speaker_size, hidden_size=configs["BLSTM_dim"], linear_dim=configs["BLSTM_Projection_dim"], num_layers=1, bidirectional=True, dropout=0)
    '''
    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params:,} total parameters.')
