#!/usr/bin/env python
# _*_ coding: UTF-8 _*_
import torch
import torch.nn as nn
from .network_resnet_conv2d import ResNet2D
from .network_resnet_conv1d import ResNet1D
from .network_resnet_pyconv2d import PyResNet2D
from .network_resnet_hpconv2d import HPResNet2D
from .network_densenet_conv2d import DenseNet2D
from .network_tcn_conv1d import MultiscaleMultibranchTCN
from .network_common_module import variable_activate, expend_params
from .network_feature_extract import FeatureExtractor
from .network_audio_visual_fusion import AudioVisualFuse
from .network_specaug import SpecAug


class AudioVisualClassifier(nn.Module):
    def __init__(self, input_types, extractor_types, extractor_settings, frontend_types, frontend_settings, backbone_types, backbone_settings, 
                 fuse_type, fuse_setting, backend_type=None, backend_setting=None,
                 skip_convert=True, extract_feature=False, **other_params):
        super(AudioVisualClassifier, self).__init__()
        self.extract_feature = extract_feature
        self.skip_convert = expend_params(value=skip_convert, length=2)

        assert input_types[0] == 'wave' and input_types[1] == 'frames', \
            'unknown input_types: {}'.format(input_types)
        assert extractor_types[0] in ['fbank', 'fbank_cnn'] and extractor_types[1] == 'gray_crop_flip', \
            'unknown extractor_types: {}'.format(extractor_types)

        self.audio_extractor = FeatureExtractor(
            extractor_type=extractor_types[0], extractor_setting=extractor_settings[0])
        self.visual_extractor = FeatureExtractor(
            extractor_type=extractor_types[1], extractor_setting=extractor_settings[1])

        self.audio_frontend = ClassifierFrontend(
            frontend_type=frontend_types[0],
            frontend_setting={**frontend_settings[0], 'in_channels': self.audio_extractor.output_size})
        self.visual_frontend = ClassifierFrontend(frontend_type=frontend_types[1],
                                                  frontend_setting=frontend_settings[1])

        self.audio_backbone = ClassifierBackbone(
            backbone_type=backbone_types[0],
            backbone_setting={**backbone_settings[0], 'in_channels': self.audio_frontend.out_channels})
        self.visual_backbone = ClassifierBackbone(
            backbone_type=backbone_types[1],
            backbone_setting={**backbone_settings[1], 'in_channels': self.visual_frontend.out_channels})

        self.audio_visual_fusion = AudioVisualFuse(
            fuse_type=fuse_type, fuse_setting={
                **fuse_setting,
                'in_channels': [self.audio_backbone.out_channels, self.visual_backbone.out_channels]})

        if not self.extract_feature:
            backend_setting.update({'in_channels': self.audio_visual_fusion.out_channels})
            self.backend = ClassifierBackend(backend_type=backend_type, backend_setting=backend_setting)

    def forward(self, wave, frames, length=None):
        # audio_process
        if not self.skip_convert[0]:
            wave = wave.float()
        audio_x, length = self.audio_extractor(wave, length)
        audio_x, length = self.audio_frontend(audio_x, length) #1dconv 
        audio_x, length = self.audio_backbone(audio_x, length) #resnet1d
        audio_t = audio_x.shape[2]  #[b,c,t]
        # video_process
        if not self.skip_convert[1]:
            frames = frames.float()
        visual_y, _ = self.visual_extractor(frames)
        visual_y, _ = self.visual_frontend(((visual_y - 0.421) / 0.165).unsqueeze(dim=1))
        y_b, y_c, y_t, y_h, y_w = visual_y.shape  #[b,c,t,h,w]
        visual_y, _ = self.visual_backbone(visual_y.transpose(1, 2).reshape(y_b * y_t, y_c, y_h, y_w)) #[b*t,c,h,w]->[b*t,c]
        visual_y = visual_y.view(y_b, y_t, -1).transpose(1, 2) #[b*t,c]-> [b,c,t]
        # import pdb; pdb.set_trace()
        # t expend
        # zhewang18:可能因为新音视频长度问题，导致下面assert异常
        if audio_t % y_t != 0:
            reserved = min(audio_t, 4*y_t)
            reserved = reserved - reserved % 4
            audio_x = audio_x[:,:,:reserved]
            visual_y = visual_y[:,:,:reserved//4]
            audio_t = reserved
            y_t = reserved//4
            # print('audio_length={} but visual_length={}'.format(audio_t, y_t))
        assert audio_t % y_t == 0 and audio_t // y_t >= 1, 'length error, audio_length={} but visual_length={}'.format(audio_t, y_t)
        if audio_t // y_t > 1:
            visual_y = torch.stack([visual_y for _ in range(audio_t // y_t)], dim=-1).reshape(y_b, -1, audio_t) #video upsampling and aligment
        fused_z, length = self.audio_visual_fusion([audio_x], [visual_y], length) ##[b,c,t]->[b,c,t]  length
        if self.extract_feature: 
            pass
        else:
            fused_z, length = self.backend(fused_z, length)
        return fused_z, length


class AudioClassifier(nn.Module):
    def __init__(self, input_type, extractor_type, extractor_setting, frontend_type, frontend_setting, 
                 backbone_type, backbone_setting, backend_type=None, backend_setting=None, skip_convert=False, extract_feature=False, **other_params):
        super(AudioClassifier, self).__init__()

        self.extract_feature = extract_feature
        self.skip_convert = skip_convert

        assert input_type == 'wave', 'unknown input_type: {}'.format(input_type)
        assert extractor_type in ['fbank', 'fbank_cnn'], 'unknown extractor_type: {}'.format(extractor_type)

        self.extractor = FeatureExtractor(extractor_type=extractor_type, extractor_setting=extractor_setting)

        self.frontend = ClassifierFrontend(
            frontend_type=frontend_type,
            frontend_setting={**frontend_setting, 'in_channels': self.extractor.output_size})

        self.backbone = ClassifierBackbone(
            backbone_type=backbone_type,
            backbone_setting={**backbone_setting, 'in_channels': self.frontend.out_channels})

        if not self.extract_feature:
            self.backend = ClassifierBackend(
                backend_type=backend_type,
                backend_setting={**backend_setting, 'in_channels': self.backbone.out_channels})

    def forward(self, wave, length=None):
        if not self.skip_convert:
            wave = wave.float()
        x, length = self.extractor(wave, length)
        x, length = self.frontend(x, length)
        x, length = self.backbone(x, length)
        if self.extract_feature:
            pass
        else:
            x, length = self.backend(x, length)
        return x, length


class VisualClassifier(nn.Module):
    def __init__(self, input_type, extractor_type, extractor_setting, frontend_type, frontend_setting, backbone_type,
                 backbone_setting, backend_type=None, backend_setting=None, skip_convert=True, extract_feature=False,
                 **other_params):
        super(VisualClassifier, self).__init__()

        self.extract_feature = extract_feature
        self.skip_convert = skip_convert

        assert input_type == 'frames', 'unknown input_type: {}'.format(input_type)
        assert extractor_type == 'gray_crop_flip', 'unknown extractor_type: {}'.format(input_type)

        self.extractor = FeatureExtractor(extractor_type=extractor_type, extractor_setting=extractor_setting)

        self.frontend = ClassifierFrontend(frontend_type=frontend_type, frontend_setting=frontend_setting)

        self.backbone = ClassifierBackbone(
            backbone_type=backbone_type,
            backbone_setting={**backbone_setting, 'in_channels': self.frontend.out_channels})

        if not self.extract_feature:
            self.backend = ClassifierBackend(
                backend_type=backend_type,
                backend_setting={**backend_setting, 'in_channels': self.backbone.out_channels})

    def forward(self, frames, length=None):
        if not self.skip_convert:
            frames = frames.float()
        x, length = self.extractor(frames, length)
        x = (x-0.421)/0.165
        x, length = self.frontend(x.unsqueeze(dim=1), length)
        x_b, x_c, x_t, x_h, x_w = x.shape
        x, length = self.backbone(x.transpose(1, 2).reshape(x_b * x_t, x_c, x_h, x_w), length)
        x = x.view(x_b, x_t, -1).transpose(1, 2)
        if self.extract_feature:
            pass
        else:
            x, length = self.backend(x, length)
        return x, length


class ClassifierFrontend(nn.Module):
    def __init__(self, frontend_type, frontend_setting):
        super(ClassifierFrontend, self).__init__()
        if frontend_type == 'conv3d':
            self.out_channels = frontend_setting.get('out_channels', 64)
            frontend_conv3d_kernel = frontend_setting.get('conv3d_kernel', (5, 7, 7))
            frontend_conv3d_stride = frontend_setting.get('conv3d_stride', (1, 2, 2))
            frontend_conv3d_padding = frontend_setting.get(
                'conv3d_padding', [(kernel_item - 1) // 2 for kernel_item in frontend_conv3d_kernel])
            frontend_act_type = frontend_setting.get('act_type', 'relu')
            frontend_pool3d_kernel = frontend_setting.get('pool3d_kernel', (1, 3, 3))
            frontend_pool3d_stride = frontend_setting.get('pool3d_stride', (1, 2, 2))
            frontend_pool3d_padding = frontend_setting.get(
                'pool3d_padding', [(kernel_item - 1) // 2 for kernel_item in frontend_pool3d_kernel])
            self.frontend = nn.Sequential(
                nn.Conv3d(
                    in_channels=1, out_channels=self.out_channels, kernel_size=frontend_conv3d_kernel,
                    stride=frontend_conv3d_stride, padding=frontend_conv3d_padding, bias=False),
                nn.BatchNorm3d(self.out_channels),
                variable_activate(act_type=frontend_act_type, in_channels=self.out_channels),
                nn.MaxPool3d(kernel_size=frontend_pool3d_kernel, stride=frontend_pool3d_stride,
                             padding=frontend_pool3d_padding)
            )
            self.length_retract = frontend_conv3d_stride[0]*frontend_pool3d_stride[0]
        elif frontend_type == 'conv1d':
            frontend_in_channels = frontend_setting.get('in_channels', 40)
            self.out_channels = frontend_setting.get('out_channels', 64)
            frontend_conv1d_kernel = frontend_setting.get('conv1d_kernel', 1)
            frontend_conv1d_stride = frontend_setting.get('conv1d_stride', 1)
            frontend_conv1d_padding = frontend_setting.get('conv1d_padding', (frontend_conv1d_kernel - 1) // 2)
            frontend_act_type = frontend_setting.get('act_type', 'relu')
            self.frontend = nn.Sequential(
                nn.Conv1d(
                    in_channels=frontend_in_channels, out_channels=self.out_channels, bias=False,
                    kernel_size=frontend_conv1d_kernel, stride=frontend_conv1d_stride, padding=frontend_conv1d_padding),
                nn.BatchNorm1d(self.out_channels),
                variable_activate(act_type=frontend_act_type, in_channels=self.out_channels))
            self.length_retract = frontend_conv1d_stride
        else:
            raise NotImplementedError('unknown frontend_type')

    def forward(self, x, length=None):
        if length is not None:
            length = (length / self.length_retract).long()
        return self.frontend(x), length


class ClassifierBackbone(nn.Module):
    def __init__(self, backbone_type, backbone_setting):
        super(ClassifierBackbone, self).__init__()
        type2backbone = {'resnet2d': ResNet2D, 'resnet1d': ResNet1D, 'pyresnet2d': PyResNet2D, 'hpresnet2d': HPResNet2D,
                         'densenet2d': DenseNet2D}
        if backbone_type in ['resnet1d', 'resnet2d', 'pyresnet2d', 'hpresnet2d']:
            if backbone_type == 'resnet1d':
                default_backbone_setting = {
                    'block_type': 'basic', 'block_num': 2, 'act_type': 'prelu',
                    'hidden_channels': [64, 128, 256, 512], 'stride': [1, 2, 2, 2], 'expansion': 1,
                    'downsample_type': 'norm'}
            elif backbone_type == 'resnet2d':
                default_backbone_setting = {
                    'block_type': 'basic', 'block_num': 2, 'act_type': 'prelu',
                    'hidden_channels': [64, 128, 256, 512], 'stride': [1, 2, 2, 2], 'expansion': 1,
                    'downsample_type': 'norm'}
            elif backbone_type == 'pyresnet2d':
                default_backbone_setting = {
                    'block_num': 2, 'hidden_channels': [64, 128, 256, 512],
                    'pyramid_level': [4, 3, 2, 1], 'kernel_size': [[3, 5, 7, 9], [3, 5, 7], [3, 5], [3]],
                    'groups': [[32, 32, 32, 32], [32, 32, 32], [32, 32], [32]], 'stride': [1, 2, 2, 2],
                    'act_type': 'relu', 'expansion': 1, 'downsample_type': 'norm'}
            else:  # hpresnet
                default_backbone_setting = {
                    'block_num': 2, 'hidden_channels': [64, 128, 256, 512],
                    'split_num': 4, 'kernel_size': [[1, 3, 5, 7] for _ in range(4)], 'stride': [1, 2, 2, 2],
                    'act_type': 'prelu', 'expansion': 2, 'stride_type': 'dw-3x3', 'downsample_type': 'norm'}
            backbone_setting = {**default_backbone_setting, **backbone_setting}
            if isinstance(backbone_setting['hidden_channels'], list):
                backbone_hidden_channels = backbone_setting['hidden_channels'][-1]
            else:
                backbone_hidden_channels = backbone_setting['hidden_channels']
            if isinstance(backbone_setting['expansion'], list):
                backbone_expansion = backbone_setting['expansion'][-1]
            else:
                backbone_expansion = backbone_setting['expansion']
            self.out_channels = int(backbone_hidden_channels * backbone_expansion)
        elif backbone_type == 'densenet2d':
            default_backbone_setting = {
                'block_num': 2, 'act_type': 'relu', 'hidden_channels': 12,
                'stride': 1, 'expansion': 4, 'out_channels': 256, 'reduction': 0.5, 'drop_rate': 0.2}
            backbone_setting = {**default_backbone_setting, **backbone_setting}
            self.out_channels = backbone_setting['out_channels']
        else:
            raise NotImplementedError('unknown backbone_type')
        self.backbone = type2backbone[backbone_type](**backbone_setting)

    def forward(self, x, length=None):
        y, length = self.backbone(x, length)
        return y, length


class ClassifierBackend(nn.Module):
    def __init__(self, backend_type, backend_setting):
        super(ClassifierBackend, self).__init__()
        if backend_type == 'tcn':
            default_backend_setting = {
                'hidden_channels': [256 * 3, 256 * 3, 256 * 3], 'num_classes': 500,
                'kernel_size': [3, 5, 7], 'dropout': 0.2, 'act_type': 'prelu', 'dwpw': False, 'consensus_type': 'mean',
                'consensus_setting': {}}
        else:
            raise NotImplementedError('unknown backend_type')
        backend_setting = {**default_backend_setting, **backend_setting}
        self.backend = MultiscaleMultibranchTCN(**backend_setting)

    def forward(self, x, length=None):
        y, length = self.backend(x, length)
        return y, length
