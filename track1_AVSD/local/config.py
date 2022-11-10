# -*- coding: utf-8 -*-


configs_SC_Single_Speaker_ivectors = {"input_dim": 40,
"average_pooling": 301,
"cnn_configs": [[2, 64, 3, 1], [64, 64, 3, 1], [64, 128, 3, (2, 1)], [128, 128, 3, 1]], 
"speaker_embedding_dim": 100,
"splice_size": 20*128+100,
"Linear_dim": 384,
"Shared_BLSTM_dim": 896,
"Linear_Shared_layer1_dim": 160,
"Linear_Shared_layer2_dim": 160,
"BLSTM_dim": 896,
"BLSTM_Projection_dim": 160,
"output_dim": 3,
"output_speaker": 1
}

configs_SC_Multiple_Speaker_ivectors_3Classes = {"input_dim": 40,
"average_pooling": 301,
"cnn_configs": [[2, 64, 3, 1], [64, 64, 3, 1], [64, 128, 3, (2, 1)], [128, 128, 3, 1]], 
"speaker_embedding_dim": 100,
"splice_size": 20*128+100,
"Linear_dim": 384,
"Shared_BLSTM_dim": 896,
"Linear_Shared_layer1_dim": 160,
"Linear_Shared_layer2_dim": 160,
"BLSTM_dim": 896,
"BLSTM_Projection_dim": 160,
"output_dim": 3,
"output_speaker": 8
}

configs_SC_Multiple_Speaker_ivectors_2Classes = {"input_dim": 40,
"average_pooling": 301,
"cnn_configs": [[2, 64, 3, 1], [64, 64, 3, 1], [64, 128, 3, (2, 1)], [128, 128, 3, 1]], 
"speaker_embedding_dim": 100,
"splice_size": 20*128+100,
"Linear_dim": 384,
"Shared_BLSTM_dim": 896,
"Linear_Shared_layer1_dim": 160,
"Linear_Shared_layer2_dim": 160,
"BLSTM_dim": 896,
"BLSTM_Projection_dim": 160,
"output_dim": 2,
"output_speaker": 2
}

configs_SC_Multiple_2Speakers_ivectors_2Classes = {"input_dim": 40,
"average_pooling": 301,
"cnn_configs": [[2, 64, 3, 1], [64, 64, 3, 1], [64, 128, 3, (2, 1)], [128, 128, 3, 1]], 
"speaker_embedding_dim": 100,
"splice_size": 20*128+100,
"Linear_dim": 384,
"Shared_BLSTM_dim": 896,
"Linear_Shared_layer1_dim": 160,
"Linear_Shared_layer2_dim": 160,
"BLSTM_dim": 896,
"BLSTM_Projection_dim": 160,
"output_dim": 2,
"output_speaker": 2
}

configs_SC_Multiple_2Speakers_xvectors_2Classes = {"input_dim": 40,
"average_pooling": 301,
"cnn_configs": [[2, 64, 3, 1], [64, 64, 3, 1], [64, 128, 3, (2, 1)], [128, 128, 3, 1]], 
"speaker_embedding_dim": 512,
"splice_size": 20*128+512,
"Linear_dim": 384,
"Shared_BLSTM_dim": 896,
"Linear_Shared_layer1_dim": 160,
"Linear_Shared_layer2_dim": 160,
"BLSTM_dim": 896,
"BLSTM_Projection_dim": 160,
"output_dim": 2,
"output_speaker": 2
}

configs2_SC_Multiple_2Speakers_xvectors_2Classes = {"input_dim": 40,
"average_pooling": 301,
"cnn_configs": [[2, 64, 3, 1], [64, 64, 3, 1], [64, 128, 3, (2, 1)], [128, 128, 3, 1]], 
"speaker_embedding_dim": 512,
"splice_size": 20*128+512,
"Linear_dim": 1024,
"Shared_BLSTM_dim": 1024,
"Linear_Shared_layer1_dim": 320,
"Linear_Shared_layer2_dim": 320,
"BLSTM_dim": 1024,
"BLSTM_Projection_dim": 320,
"output_dim": 2,
"output_speaker": 2
}

configs_SC_Spectrogram_Multiple_2Speakers_ivectors_2Classes = {"input_dim": 257,
"average_pooling": 301,
"cnn_configs": [[2, 64, 3, 1], [64, 64, 3, (2, 1)], [64, 128, 3, (2, 1)], [128, 128, 3, (2, 1)]], 
"speaker_embedding_dim": 100,
"splice_size": 33*128+100,
"Linear_dim": 384,
"Shared_BLSTM_dim": 896,
"Linear_Shared_layer1_dim": 160,
"Linear_Shared_layer2_dim": 160,
"BLSTM_dim": 896,
"BLSTM_Projection_dim": 160,
"output_dim": 2,
"output_speaker": 2
}

configs2_SC_Spectrogram_Multiple_2Speakers_ivectors_2Classes = {"input_dim": 257,
"average_pooling": 301,
"cnn_configs": [[2, 64, 3, (2, 1)], [64, 64, 3, (2, 1)], [64, 128, 3, (2, 1)], [128, 128, 3, (2, 1)]], 
"speaker_embedding_dim": 100,
"splice_size": 17*128+100,
"Linear_dim": 384,
"Shared_BLSTM_dim": 896,
"Linear_Shared_layer1_dim": 160,
"Linear_Shared_layer2_dim": 160,
"BLSTM_dim": 896,
"BLSTM_Projection_dim": 160,
"output_dim": 2,
"output_speaker": 2
}

configs3_SC_Spectrogram_Multiple_2Speakers_ivectors_2Classes = {"input_dim": 257,
"average_pooling": 301,
"cnn_configs": [[2, 64, 3, 1], [64, 64, 3, 1], [64, 128, 3, (2, 1)], [128, 128, 3, 1]], 
"speaker_embedding_dim": 100,
"splice_size": 129*128+100,
"Linear_dim": 384,
"Shared_BLSTM_dim": 896,
"Linear_Shared_layer1_dim": 160,
"Linear_Shared_layer2_dim": 160,
"BLSTM_dim": 896,
"BLSTM_Projection_dim": 160,
"output_dim": 2,
"output_speaker": 2
}

configs4_SC_Spectrogram_Multiple_2Speakers_ivectors_2Classes = {"input_dim": 257,
"cnn_configs": [[1, 256, 3, 1], [256, 128, 3, 1], [128, 128, 3, (2, 1)], [128, 128, 3, 1]], 
"speaker_embedding_dim": 100,
"splice_size": 129*128+100,
"Linear_dim": 384,
"Shared_BLSTM_dim": 896,
"Linear_Shared_layer1_dim": 160,
"Linear_Shared_layer2_dim": 160,
"BLSTM_dim": 896,
"BLSTM_Projection_dim": 160,
"output_dim": 2,
"output_speaker": 2
}

configs5_SC_Spectrogram_Multiple_2Speakers_ivectors_2Classes = {"input_dim": 257,
"average_pooling": 301,
"cnn_configs": [[2, 64, 3, 1], [64, 64, 3, 1], [64, 128, 3, (2, 1)], [128, 128, 3, 1]], 
"speaker_embedding_dim": 100,
"splice_size": 129*128+100,
"Linear_dim": 1024,
"Shared_BLSTM_dim": 1024,
"Linear_Shared_layer1_dim": 320,
"Linear_Shared_layer2_dim": 320,
"BLSTM_dim": 1024,
"BLSTM_Projection_dim": 320,
"output_dim": 2,
"output_speaker": 2
}

configs_SC_Multiple_3Speakers_ivectors_2Classes = {"input_dim": 40,
"average_pooling": 301,
"cnn_configs": [[2, 64, 3, 1], [64, 64, 3, 1], [64, 128, 3, (2, 1)], [128, 128, 3, 1]], 
"speaker_embedding_dim": 100,
"splice_size": 20*128+100,
"Linear_dim": 384,
"Shared_BLSTM_dim": 896,
"Linear_Shared_layer1_dim": 160,
"Linear_Shared_layer2_dim": 160,
"BLSTM_dim": 896,
"BLSTM_Projection_dim": 160,
"output_dim": 2,
"output_speaker": 3
}

configs_SC_Multiple_4Speakers_xvectors_2Classes = {"input_dim": 40,
"average_pooling": 301,
"cnn_configs": [[2, 64, 3, 1], [64, 64, 3, 1], [64, 128, 3, (2, 1)], [128, 128, 3, 1]], 
"speaker_embedding_dim": 256,
"splice_size": 20*128+256,
"Linear_dim": 384,
"Shared_BLSTM_dim": 896,
"Linear_Shared_layer1_dim": 160,
"Linear_Shared_layer2_dim": 160,
"BLSTM_dim": 896,
"BLSTM_Projection_dim": 160,
"output_dim": 2,
"output_speaker": 4
}

configs2_SC_Multiple_4Speakers_xvectors_2Classes = {"input_dim": 40,
"average_pooling": 301,
"cnn_configs": [[2, 64, 3, 1], [64, 64, 3, 1], [64, 128, 3, (2, 1)], [128, 128, 3, 1]], 
"speaker_embedding_dim": 256,
"splice_size": 20*128+256,
"Linear_dim": 512,
"Shared_BLSTM_dim": 1024,
"Linear_Shared_layer1_dim": 240,
"Linear_Shared_layer2_dim": 240,
"BLSTM_dim": 896,
"BLSTM_Projection_dim": 160,
"output_dim": 2,
"output_speaker": 4
}

configs_SC_Multiple_4Speakers_ivectors_2Classes = {"input_dim": 40,
"average_pooling": 301,
"cnn_configs": [[2, 64, 3, 1], [64, 64, 3, 1], [64, 128, 3, (2, 1)], [128, 128, 3, 1]], 
"speaker_embedding_dim": 100,
"splice_size": 20*128+100,
"Linear_dim": 384,
"Shared_BLSTM_dim": 896,
"Linear_Shared_layer1_dim": 160,
"Linear_Shared_layer2_dim": 160,
"BLSTM_dim": 896,
"BLSTM_Projection_dim": 160,
"output_dim": 2,
"output_speaker": 4
}

configs_SC_Multiple_4Speakers_AEmbedding_VEmbedding_2Classes = {"input_dim": 40,
"average_pooling": 301,
"cnn_configs": [[2, 64, 3, 1], [64, 64, 3, 1], [64, 128, 3, (2, 1)], [128, 128, 3, 1]], 
"speaker_embedding_dim": 256,
"splice_size": 20*128+100+256,
"Linear_dim": 512,
"Shared_BLSTM_dim": 896,
"Linear_Shared_layer1_dim": 160,
"Linear_Shared_layer2_dim": 160,
"BLSTM_dim": 896,
"BLSTM_Projection_dim": 160,
"dropout": 0.0,
"output_dim": 2,
"output_speaker": 4
}

configs_SC_Multiple_5Speakers_ivectors_2Classes = {"input_dim": 40,
"average_pooling": 301,
"cnn_configs": [[2, 64, 3, 1], [64, 64, 3, 1], [64, 128, 3, (2, 1)], [128, 128, 3, 1]], 
"speaker_embedding_dim": 100,
"splice_size": 20*128+100,
"Linear_dim": 384,
"Shared_BLSTM_dim": 896,
"Linear_Shared_layer1_dim": 160,
"Linear_Shared_layer2_dim": 160,
"BLSTM_dim": 896,
"BLSTM_Projection_dim": 160,
"output_dim": 2,
"output_speaker": 5
}

configs_SC_Multiple_8Speakers_ivectors_2Classes = {"input_dim": 40,
"average_pooling": 301,
"cnn_configs": [[2, 64, 3, 1], [64, 64, 3, 1], [64, 128, 3, (2, 1)], [128, 128, 3, 1]], 
"speaker_embedding_dim": 100,
"splice_size": 20*128+100,
"Linear_dim": 384,
"Shared_BLSTM_dim": 896,
"Linear_Shared_layer1_dim": 160,
"Linear_Shared_layer2_dim": 160,
"BLSTM_dim": 896,
"BLSTM_Projection_dim": 160,
"output_dim": 2,
"output_speaker": 8
}

configs_SC_Multiple_6Speakers_ivectors_2Classes = {"input_dim": 40,
"average_pooling": 301,
"cnn_configs": [[2, 64, 3, 1], [64, 64, 3, 1], [64, 128, 3, (2, 1)], [128, 128, 3, 1]], 
"speaker_embedding_dim": 100,
"splice_size": 20*128+100,
"Linear_dim": 384,
"Shared_BLSTM_dim": 896,
"Linear_Shared_layer1_dim": 160,
"Linear_Shared_layer2_dim": 160,
"BLSTM_dim": 896,
"BLSTM_Projection_dim": 160,
"output_dim": 2,
"output_speaker": 6
}

configs_SC_Multiple_6Speakers_video_2Classes = {"input_dim": 40,
"average_pooling": 301,
"cnn_configs": [[2, 64, 3, 1], [64, 64, 3, 1], [64, 128, 3, (2, 1)], [128, 128, 3, 1]], 
"speaker_embedding_dim": 100,
"splice_size": 20*128+256,
"Linear_dim": 384,
"Shared_BLSTM_dim": 896,
"Linear_Shared_layer1_dim": 160,
"Linear_Shared_layer2_dim": 160,
"BLSTM_dim": 512,
"BLSTM_Projection_dim": 160,
"output_dim": 2,
"output_speaker": 6
}

configs_SC_Multiple_6Speakers_VEmbedding_2Classes = {"input_dim": 40,
"average_pooling": 301,
"cnn_configs": [[2, 64, 3, 1], [64, 64, 3, 1], [64, 128, 3, (2, 1)], [128, 128, 3, 1]], 
"speaker_embedding_dim": 256,
"splice_size": 20*128+256,
"Linear_dim": 384,
"Shared_BLSTM_dim": 896,
"Linear_Shared_layer1_dim": 160,
"Linear_Shared_layer2_dim": 160,
"BLSTM_dim": 896,
"BLSTM_Projection_dim": 160,
"dropout": 0.0,
"output_dim": 2,
"output_speaker": 6
}

configs_SC_Multiple_6Speakers_AEmbedding_VEmbedding_2Classes = {"input_dim": 40,
"average_pooling": 301,
"cnn_configs": [[2, 64, 3, 1], [64, 64, 3, 1], [64, 128, 3, (2, 1)], [128, 128, 3, 1]], 
"speaker_embedding_dim": 256,
"splice_size": 20*128+100+256,
"Linear_dim": 512,
"Shared_BLSTM_dim": 896,
"Linear_Shared_layer1_dim": 160,
"Linear_Shared_layer2_dim": 160,
"BLSTM_dim": 896,
"BLSTM_Projection_dim": 160,
"dropout": 0.0,
"output_dim": 2,
"output_speaker": 6
}

configs_SC_Multiple_6Speakers_AEmbedding_Ivector_VEmbedding_2Classes = {"resnet_out_dim": 8192,
"aembedding_dim": 256,
"splice_size": 256+100+256,
"Linear_dim": 512,
"Shared_BLSTM_dim": 896,
"Linear_Shared_layer1_dim": 160,
"Linear_Shared_layer2_dim": 160,
"BLSTM_dim": 896,
"BLSTM_Projection_dim": 160,
"dropout": 0.0,
"output_dim": 2,
"output_speaker": 6
}

configs_6Speakers_ConformerVEmbedding_ConformerDeocoding_2Classes = {"input_dim": 40,
"num_speaker": 6,
"encoder_conformer": {"dim":256, "dim_head":64, "heads":4, "ff_mult":4, "conv_expansion_factor":2, "conv_kernel_size":31, "attn_dropout":0., "ff_dropout":0., "conv_dropout":0.},
"encoder_conformer_layers": 3,
"speaker_embedding": 256,
"share_conformer": {"dim":512, "dim_head":64, "heads":8, "ff_mult":4, "conv_expansion_factor":2, "conv_kernel_size":31, "attn_dropout":0., "ff_dropout":0., "conv_dropout":0.},
"share_conformer_layers": 2,
"BLSTM_dim": 896,
"BLSTM_Projection_dim": 320,
"dropout": 0.0,
"decoder_fc": [128, 2]
}

configs2_6Speakers_ConformerVEmbedding_ConformerDeocoding_2Classes = {"input_dim": 40,
"num_speaker": 6,
"encoder_conformer": {"dim":256, "dim_head":64, "heads":4, "ff_mult":4, "conv_expansion_factor":2, "conv_kernel_size":31, "attn_dropout":0., "ff_dropout":0., "conv_dropout":0.},
"encoder_conformer_layers": 3,
"speaker_embedding": 256,
"share_conformer": {"dim":256, "dim_head":64, "heads":4, "ff_mult":4, "conv_expansion_factor":2, "conv_kernel_size":31, "attn_dropout":0., "ff_dropout":0., "conv_dropout":0.},
"share_conformer_layers": 2,
"BLSTM_dim": 512,
"BLSTM_Projection_dim": 160,
"dropout": 0.0,
"decoder_fc": [64, 2]
}

configs3_6Speakers_ConformerVEmbedding_ConformerDeocoding_2Classes = {"input_dim": 40,
"num_speaker": 6,
"encoder_conformer": {"dim":256, "dim_head":64, "heads":4, "ff_mult":4, "conv_expansion_factor":2, "conv_kernel_size":31, "attn_dropout":0., "ff_dropout":0., "conv_dropout":0.},
"encoder_conformer_layers": 3,
"speaker_embedding": 256,
"share_conformer": {"dim":256, "dim_head":64, "heads":4, "ff_mult":4, "conv_expansion_factor":2, "conv_kernel_size":31, "attn_dropout":0., "ff_dropout":0., "conv_dropout":0.},
"share_conformer_layers": 4,
"BLSTM_dim": 896,
"BLSTM_Projection_dim": 160,
"dropout": 0.0,
"decoder_fc": [128, 2]
}

configs_6Speakers_ConformerVEmbedding_ConformerDeocoding_Dropout_2Classes = {"input_dim": 40,
"num_speaker": 6,
"encoder_conformer": {"dim":256, "dim_head":64, "heads":4, "ff_mult":4, "conv_expansion_factor":2, "conv_kernel_size":31, "attn_dropout":0.2, "ff_dropout":0.2, "conv_dropout":0.2},
"encoder_conformer_layers": 3,
"speaker_embedding": 256,
"share_conformer": {"dim":512, "dim_head":64, "heads":8, "ff_mult":4, "conv_expansion_factor":2, "conv_kernel_size":31, "attn_dropout":0.2, "ff_dropout":0.2, "conv_dropout":0.2},
"share_conformer_layers": 2,
"BLSTM_dim": 896,
"BLSTM_Projection_dim": 320,
"dropout": 0.2,
"decoder_fc": [128, 2]
}

configs_ECAPA_SC_Multiple_8Speakers_ivectors_2Classes = {"input_dim": 40,
"average_pooling": 301,
"cnn_configs": [[2, 64, 3, 1], [64, 64, 3, 1], [64, 128, 3, (2, 1)], [128, 128, 3, 1]], 
"speaker_embedding_dim": 100,
"splice_size": 1536+100,
"Linear_dim": 384,
"Shared_BLSTM_dim": 896,
"Linear_Shared_layer1_dim": 160,
"Linear_Shared_layer2_dim": 160,
"BLSTM_dim": 896,
"BLSTM_Projection_dim": 160,
"output_dim": 2,
"output_speaker": 8
}

configs_MC_STC_ivectors = {"input_dim": 40,
"average_pooling": 301,
"cnn_configs": [[2, 64, 3, 1], [64, 64, 3, 1], [64, 128, 3, (2, 1)], [128, 128, 3, 1]],
"speaker_embedding_dim": 100,
"splice_size": 20*128+100,
"Linear_dim": 384,
"Shared_BLSTM_dim": 896,
"Linear_Shared_layer1_dim": 160,
"Linear_Shared_layer2_dim": 160,
"cnn_attention": [160*2, 320, 3, 1],
"BLSTM_dim": 896,
"BLSTM_Projection_dim": 160,
"output_dim": 2,
"output_speaker": 8
}

configs_SC_3_layers_BLSTMP_ivectors = {
    "1": [140, 896, 160],
    "2": [320, 896, 160],
    "3": [320, 896, 160],
    "FC": [320, 3]
}


configs = {
    "configs_SC_Multiple_2Speakers_ivectors_2Classes": configs_SC_Multiple_2Speakers_ivectors_2Classes,
    "configs_SC_Multiple_8Speakers_ivectors_2Classes": configs_SC_Multiple_8Speakers_ivectors_2Classes,
    "configs_SC_Multiple_6Speakers_ivectors_2Classes": configs_SC_Multiple_6Speakers_ivectors_2Classes,
    "configs_SC_Multiple_4Speakers_ivectors_2Classes": configs_SC_Multiple_4Speakers_ivectors_2Classes,
    "configs_SC_Multiple_4Speakers_xvectors_2Classes": configs_SC_Multiple_4Speakers_xvectors_2Classes,
    "configs2_SC_Multiple_4Speakers_xvectors_2Classes": configs2_SC_Multiple_4Speakers_xvectors_2Classes,
    "configs_ECAPA_SC_Multiple_8Speakers_ivectors_2Classes": configs_ECAPA_SC_Multiple_8Speakers_ivectors_2Classes
}