# -*- coding: utf-8 -*-
import torch.nn as nn
import torch
import numpy as np
try:
    from .extract_lip_embedding_resnet import VideoFrontend
    from .conformer import ConformerBlock
except:
    from extract_lip_embedding_resnet import VideoFrontend
    from conformer import ConformerBlock

class LSTM_Encoder(nn.Module):
    def __init__(self,feature_dim,hidden_size,num_layers):
        super(LSTM_Encoder, self).__init__()
        self.feature_dim = feature_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.stack_rnn = nn.LSTM(input_size=self.feature_dim, hidden_size=self.hidden_size, batch_first=False, bidirectional=False, num_layers=1)

    def forward(self, cur_inputs, current_frame):
        packed_input = nn.utils.rnn.pack_padded_sequence(cur_inputs, current_frame)
        rnn_out, _ = self.stack_rnn(packed_input)
        rnn_out, _ = nn.utils.rnn.pad_packed_sequence(rnn_out) 
               
        return rnn_out


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

class VideoBackend(nn.Module):
    def __init__(self,args=1):
        super(VideoBackend, self).__init__()
        # self.lip_encoder = VideoFrontend()
        # self.conv_av1 = nn.Conv2d(256,256, kernel_size=(1, 5), stride=(1,2), padding=(0,2))
        # self.conv_av2 = nn.Conv2d(256,512, kernel_size=(1, 5), stride=(1,2), padding=(0,2))
        #self.feature_dim = args.input_dim
        #self.hidden_size = args.hidden_sizes
        #self.num_layers = args.lstm_num_layers
        self.hidden_size = 256
        self.num_layers = 1
        self.encoder = LSTM_Encoder(256, self.hidden_size, self.num_layers)
        #[[2, 64, 3, 1], [64, 64, 3, 1], [64, 128, 3, (2, 1)], [128, 128, 3, 1]]
        self.Conv2d = nn.Sequential()
        self.Conv2d.add_module('CNN2D_BN_Relu_1', CNN2D_BN_Relu(1, 32, 5, (1, 1)))
        self.Conv2d.add_module('CNN2D_BN_Relu_2', CNN2D_BN_Relu(32, 128, 5, (2, 1)))
        self.Conv2d.add_module('CNN2D_BN_Relu_3', CNN2D_BN_Relu(128, 256, 5, (2, 1)))

        #self.conv1 = nn.Conv2d(1,32, kernel_size=(5,5), stride=(1,1), padding=(2,2))
        #self.conv2 = nn.Conv2d(32, 128, kernel_size=(5,5), stride=(2,2), padding=(2,2))
        #self.max_pool1 = nn.MaxPool2d(2, stride=(1,2))
        #self.conv3 = nn.Conv2d(128, 256, kernel_size=(5,5), stride=(1,2), padding=(2,2))
        self.fc1 = nn.Linear(16384, 1024)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, 2)
       

    def forward(self, lip_inputs, current_frame):
        if type(current_frame) == torch.Tensor:
            current_frame = list(current_frame.detach().cpu().numpy())
        # #lstm layer
        encoder_output = self.encoder(lip_inputs, current_frame) # (T, B, 256)
        encoder_output = encoder_output.permute(1,2,0) 
        # print(encoder_output.shape) # (B,64,T)
        batchsize, _, Time = encoder_output.shape
        
        '''
        1stBatch(1st sentence) length1 * (BLSTM_size * 2)
        2ndBatch(2nd sentence) length2 * (BLSTM_size * 2)
        ...
        '''

        # #CNN detector and classifier 
        cnn_input = encoder_output.unsqueeze(1) 
        #cnn_out = self.conv1(cnn_input)
        #cnn_out = self.conv2(cnn_out)
        #cnn_out = self.max_pool1(cnn_out)
        #cnn_out = self.conv3(cnn_out)
        cnn_out = self.Conv2d(cnn_input).reshape(batchsize, -1, Time).permute(0,2,1) 
        #print(cnn_out.shape)


        #Dimension reduction: remove the padding frames; Batch * Time * (BLSTM_Projection_dim * 2) => (Batch * Time) * (BLSTM_Projection_dim * 2)
        lens = [k for i, m in enumerate(current_frame) for k in range(i * Time, m + i * Time)]
        cnn_out = cnn_out.reshape(batchsize * Time, -1)[lens, :]
        #cnn_out = (cnn_out.mean(-2)).permute(0,2,1) 
        fc_out = self.fc1(cnn_out)
        fc_out = self.dropout(fc_out)
        fc_out = self.fc2(fc_out)
        fc_out = self.fc3(fc_out)

        #max_pool2 = nn.MaxPool2d((cnn_out.shape[1],1))
        #cnn_out = max_pool2(cnn_out) 
        #cnn_out = cnn_out.squeeze(-1)
 
        return fc_out

class Visual_VAD_Net(nn.Module):
    def __init__(self, args=1):
        super(Visual_VAD_Net, self).__init__()
        self.lip_encoder = VideoFrontend()
        self.lip_decoder = VideoBackend(args)

    def forward(self, video_inputs, current_frame):
        #video_inputs = video_inputs.permute(1, 0, 2, 3)
        lip_inputs = self.lip_encoder(video_inputs) #(B, T, 256)
        lip_inputs = lip_inputs.permute(1, 0, 2) # (T, B, 256)

        lip_outputs = self.lip_decoder(lip_inputs, current_frame)

        return lip_outputs


class Visual_VAD_FC_Net(nn.Module):
    def __init__(self):
        super(Visual_VAD_FC_Net, self).__init__()
        self.lip_encoder = VideoFrontend()
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(256, 32)
        self.fc2 = nn.Linear(32, 2)

    def forward(self, video_inputs, current_frame):
        lip_encoder = self.lip_encoder(video_inputs) #(B, T, 256)
        batchsize, Time, _ = lip_encoder.shape
        if type(current_frame) == torch.Tensor:
            current_frame = list(current_frame.detach().cpu().numpy())
        lens = [k for i, m in enumerate(current_frame) for k in range(i * Time, m + i * Time)]
        lip_encoder = lip_encoder.reshape(batchsize * Time, -1)[lens, :]

        dropout = self.dropout(lip_encoder)
        fc1 = self.fc1(dropout)
        fc2 = self.fc2(fc1)

        return fc2


class Visual_VAD_Conformer_Net(nn.Module):
    def __init__(self):
        super(Visual_VAD_Conformer_Net, self).__init__()
        self.lip_encoder = VideoFrontend()
        #self.dropout = nn.Dropout(0.2)
        self.conformer1 = ConformerBlock(
                                        dim = 256,
                                        dim_head = 64,
                                        heads = 4,
                                        ff_mult = 4,
                                        conv_expansion_factor = 2,
                                        conv_kernel_size = 31,
                                        attn_dropout = 0.,
                                        ff_dropout = 0.,
                                        conv_dropout = 0.
                                    )
        self.conformer2 = ConformerBlock(
                                        dim = 256,
                                        dim_head = 64,
                                        heads = 4,
                                        ff_mult = 4,
                                        conv_expansion_factor = 2,
                                        conv_kernel_size = 31,
                                        attn_dropout = 0.,
                                        ff_dropout = 0.,
                                        conv_dropout = 0.
                                    )
        self.conformer3 = ConformerBlock(
                                        dim = 256,
                                        dim_head = 64,
                                        heads = 4,
                                        ff_mult = 4,
                                        conv_expansion_factor = 2,
                                        conv_kernel_size = 31,
                                        attn_dropout = 0.,
                                        ff_dropout = 0.,
                                        conv_dropout = 0.
                                    )
        self.decoder = LSTM_Encoder(256, 256, 1)
        self.fc1 = nn.Linear(256, 32)
        self.fc2 = nn.Linear(32, 2)

    def forward(self, video_inputs, current_frame, return_embedding=False):
        lip_encoder = self.lip_encoder(video_inputs) #(B, T, 256)
        batchsize, Time, _ = lip_encoder.shape

        #dropout = self.dropout(lip_encoder)
        conformer1 = self.conformer1(lip_encoder)
        conformer2 = self.conformer2(conformer1)
        conformer3 = self.conformer3(conformer2)

        if type(current_frame) == torch.Tensor:
            current_frame = list(current_frame.detach().cpu().numpy())
        decoder = self.decoder(conformer3.permute(1, 0, 2), current_frame).permute(1, 0, 2) 

        lens = [k for i, m in enumerate(current_frame) for k in range(i * Time, m + i * Time)]

        fc1 = self.fc1(decoder.reshape(batchsize * Time, -1)[lens, :])
        fc2 = self.fc2(fc1)

        if return_embedding:
            return fc2, decoder
        else:
            return fc2

class Visual_VAD_Conformer_Embedding(nn.Module):
    def __init__(self):
        super(Visual_VAD_Conformer_Embedding, self).__init__()
        self.lip_encoder = VideoFrontend()
        #self.dropout = nn.Dropout(0.2)
        self.conformer1 = ConformerBlock(
                                        dim = 256,
                                        dim_head = 64,
                                        heads = 4,
                                        ff_mult = 4,
                                        conv_expansion_factor = 2,
                                        conv_kernel_size = 31,
                                        attn_dropout = 0.,
                                        ff_dropout = 0.,
                                        conv_dropout = 0.
                                    )
        self.conformer2 = ConformerBlock(
                                        dim = 256,
                                        dim_head = 64,
                                        heads = 4,
                                        ff_mult = 4,
                                        conv_expansion_factor = 2,
                                        conv_kernel_size = 31,
                                        attn_dropout = 0.,
                                        ff_dropout = 0.,
                                        conv_dropout = 0.
                                    )
        self.conformer3 = ConformerBlock(
                                        dim = 256,
                                        dim_head = 64,
                                        heads = 4,
                                        ff_mult = 4,
                                        conv_expansion_factor = 2,
                                        conv_kernel_size = 31,
                                        attn_dropout = 0.,
                                        ff_dropout = 0.,
                                        conv_dropout = 0.
                                    )
        self.decoder = LSTM_Encoder(256, 256, 1)
        self.fc1 = nn.Linear(256, 32)
        self.fc2 = nn.Linear(32, 2)

    def forward(self, video_inputs, current_frame):
        lip_encoder = self.lip_encoder(video_inputs) #(B, T, 256)
        batchsize, Time, _ = lip_encoder.shape

        #dropout = self.dropout(lip_encoder)
        conformer1 = self.conformer1(lip_encoder)
        conformer2 = self.conformer2(conformer1)
        conformer3 = self.conformer3(conformer2)

        if type(current_frame) == torch.Tensor:
            current_frame = list(current_frame.detach().cpu().numpy())
        decoder = self.decoder(conformer3.permute(1, 0, 2), current_frame).permute(1, 0, 2)
        return decoder

if __name__ == '__main__':
    nnet = Visual_VAD_Net()
    inputs = torch.from_numpy(np.random.randn(20, 2, 96, 96).astype(np.float32))
    nframe = [20, 20]
    a = nnet(inputs, nframe)
    print(a.shape)
