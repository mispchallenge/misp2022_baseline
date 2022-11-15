#!/usr/bin/env python
# _*_ coding: UTF-8 _*_
import torch
import numpy as np
import torch.nn as nn
from .network_comformer_encoder import ConformerEncoder
from .network_common_module import unify_time_dimension

# from network_comformer_encoder import ConformerEncoder
# from network_common_module import unify_time_dimension

  
class CoformerFusion(nn.Module): #[B,512,T] + [B,512,T] -> [B,256*3,T]
    def __init__(self, fuse_type, fuse_setting):
        super(CoformerFusion, self).__init__()
        self.fuse_type = fuse_type
        if self.fuse_type == 'cat':
            self.out_channels = np.sum(fuse_setting['input_size'])
        elif self.fuse_type == 'comformer':
            fuse_setting['input_size'] = int(np.sum(fuse_setting['input_size']))
            default_fuse_setting = dict( 
                num_blocks=6, 
                input_size=1024,
                output_size=256*3,    # dimension of attention
                attention_heads=4,
                linear_units=2048,  # the number of units of position-wise feed forward
                dropout_rate=0.1,
                positional_dropout_rate=0.1,
                attention_dropout_rate=0.0,
                input_layer="linear", # linear won't subsampling ,don't choose conv2d.
                normalize_before=True,
                pos_enc_layer_type="rel_pos",
                selfattention_layer_type="rel_selfattn",
                activation_type="swish",
                macaron_style=True,
                use_cnn_module=True,
                cnn_module_kernel=15
            )

            default_fuse_setting.update(fuse_setting)
            self.fusion = ConformerEncoder(**default_fuse_setting)
            self.out_channels = default_fuse_setting['output_size']
        else:
            raise NotImplementedError('unknown fuse_type')

    def forward(self, audios, videos, length=None): #length: must be the longer ones. e.g: audio 100fps video 25 fps , audio is chosen.
        if self.fuse_type == 'cat':
            x = torch.cat(unify_time_dimension(*audios, *videos), dim=1)
        elif self.fuse_type == 'comformer':
            x = torch.cat(unify_time_dimension(*audios, *videos), dim=1).transpose(1,2) # [B,512,T] + [B,512,T]  -> [B,1024,T] 
            x, length , _ = self.fusion(x, length)  # [B,1024,T]  -> [B,256*3,T]
        else:
            raise NotImplementedError('unknown fuse_type')
        return x.transpose(1,2), length


if __name__ == '__main__':
    lengths =  torch.randint(40, 50, (8,))
    lengths[0] = 50
    audio_visual_fusion  = CoformerFusion(fuse_type="comformer",fuse_setting={"input_size":[512,512]})
    used_z, length  = audio_visual_fusion([torch.ones(8, 512, 50)], [torch.ones(8, 512, 50)], lengths)
    print(used_z.shape,length)

