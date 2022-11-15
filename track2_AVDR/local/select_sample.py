#!/usr/bin/env python
# _*_ coding: UTF-8 _*_
import os
import json
import codecs
import torch
import cv2
from scipy.io import wavfile
from tqdm import tqdm


def json2dic(jsonpath, dic=None):
    """
    read dic from json or write dic to json
    :param jsonpath: filepath of json
    :param dic: content dic or None, None means read
    :return: content dic for read while None for write
    """
    if dic is None:
        with codecs.open(jsonpath, 'r') as handle:
            output = json.load(handle)
        return output
    else:
        assert isinstance(dic, dict)
        with codecs.open(jsonpath, 'w') as handle:
            json.dump(dic, handle)
        return None


def text2lines(textpath, lines_content=None):
    """
    read lines from text or write lines to txt
    :param textpath: filepath of text
    :param lines_content: list of lines or None, None means read
    :return: processed lines content for read while None for write
    """
    if lines_content is None:
        with codecs.open(textpath, 'r') as handle:
            lines_content = handle.readlines()
        processed_lines = [*map(lambda x: x[:-1] if x[-1] in ['\n'] else x, lines_content)]
        return processed_lines
    else:
        processed_lines = [*map(lambda x: x if x[-1] in ['\n'] else '{}\n'.format(x), lines_content)]
        with codecs.open(textpath, 'w') as handle:
            handle.write(''.join(processed_lines))
        return None


if __name__ == '__main__':
    far_a_result = json2dic('/raw7/cv1/hangchen2/misp2021_avsr/exp/0_1_MISP2021_far_asr/predict_best_eval_addition/result_after_exp_tri3_far_audio_decode_log_likelihoods/result_cer.json')
    far_a_far_v_result = json2dic('/raw7/cv1/hangchen2/misp2021_avsr/exp/1_3_MISP2021_far_wave_far_lip_avsr/predict_best_eval_addition/result_after_exp_tri3_far_audio_decode_log_likelihoods/result_cer.json')
    far_a_middle_v_result = json2dic('/raw7/cv1/hangchen2/misp2021_avsr/exp/1_5_MISP2021_far_wave_middle_lip_avsr/predict_best_eval_addition/result_after_exp_tri3_far_audio_decode_log_likelihoods/result_cer.json')
    far_v_result = json2dic('/raw7/cv1/hangchen2/misp2021_avsr/exp/2_1_MISP2021_far_lip_vsr/predict_best_eval_addition/result_after_exp_tri3_far_audio_decode_log_likelihoods/result_cer.json')
    middle_v_result = json2dic('/raw7/cv1/hangchen2/misp2021_avsr/exp/2_2_MISP2021_middle_lip_vsr/predict_best_eval_addition/result_after_exp_tri3_near_audio_decode_log_likelihoods/result_cer.json')
    # reject_list = ['S286_R53_S286287288289_C03_I1_095300-096132', ]
    # reject_list = ['S285_R53_S283284285_C03_I0_087112-088084']
    reject_list = []
    keys = [*(set(far_a_result['keys']) & set(far_a_far_v_result['keys']) & set(far_a_middle_v_result['keys']) - set(reject_list))]
    # find best s
    best_s_cha = 0
    best_sample = ''
    id2class = {'C07': 'nonoverlap', 'C08': 'nonoverlap', 'C12': 'nonoverlap', 'C01': 'nonoverlap+tv', 'C02': 'nonoverlap+tv', 
                'C09': 'nonoverlap+tv', 'C04': 'overlap', 'C03': 'overlap', 'C10': 'overlap', 'C05': 'overlap+tv', 'C06': 'overlap+tv', 
                'C11': 'overlap+tv'}
    for key in keys:
        _, _, _, config_id, _, _ = key.split('_')
        class_id = id2class[config_id]
        if class_id in ['overlap', 'overlap+tv']:
        # if class_id in ['nonoverlap+tv', 'overlap+tv']:
            # far_a_s = far_a_result['key2path'][key]['csid'][1]
            # far_a_far_v_s = far_a_far_v_result['key2path'][key]['csid'][1]
            # far_a_middle_v_s = far_a_middle_v_result['key2path'][key]['csid'][1]
            far_a_csid = far_a_result['key2path'][key]['csid']
            far_a_middle_v_csid = far_a_middle_v_result['key2path'][key]['csid']
            t = float(far_a_csid[0] + far_a_csid[1] + far_a_csid[3])
            middle_v_csid = middle_v_result['key2path'][key]['csid']
            if far_a_csid[1] + far_a_csid[2] + far_a_csid[3] >= 1 and far_a_middle_v_csid[1] + far_a_middle_v_csid[2] + far_a_middle_v_csid[3] == 0:
                # far_a_far_v_c = far_a_far_v_result['key2path'][key]['csid'][0]
                
                # far_a_i = far_a_result['key2path'][key]['csid'][2]
                # far_a_far_v_i = far_a_far_v_result['key2path'][key]['csid'][2]
                # far_a_middle_v_i = far_a_middle_v_result['key2path'][key]['csid'][2]
                # middle_v_i = middle_v_result['key2path'][key]['csid'][2]
                
                # if far_a_middle_v_c - far_a_c > best_s_cha:
                #     best_sample = key
                #     best_s_cha = far_a_middle_v_c - far_a_c
                # if middle_v_csid[0] > best_s_cha:
                #     best_sample = key
                #     best_s_cha = middle_v_csid[0]
                # if far_a_i - far_a_middle_v_i > best_s_cha:
                #     best_sample = key
                #     best_s_cha = far_a_i - far_a_middle_v_i
                # if (far_a_middle_v_csid[0] - far_a_csid[0]) / t > best_s_cha:
                #     best_sample = key
                #     best_s_cha = (far_a_middle_v_csid[0] - far_a_csid[0]) / t
                if (far_a_csid[2] - far_a_middle_v_csid[2]) > best_s_cha:
                    best_sample = key
                    best_s_cha = far_a_csid[2] - far_a_middle_v_csid[2]
    
    os.makedirs('./sample_{}'.format(best_sample), exist_ok=True)
    print(best_sample)
    
    # best_sample = 'S425_R72_S424425_C07_I1_026404-026496'
    
    # text
    text_out = []
    text_out.append('ref: {}'.format(far_a_result['key2path'][best_sample]['ref']))
    text_out.append('far_a: {}'.format(far_a_result['key2path'][best_sample]['hyp']))
    text_out.append('far_a: {}'.format(far_a_result['key2path'][best_sample]['op']))
    text_out.append('far_a_far_v: {}'.format(far_a_far_v_result['key2path'][best_sample]['hyp']))
    text_out.append('far_a_far_v: {}'.format(far_a_far_v_result['key2path'][best_sample]['op']))
    text_out.append('far_a_middle_v: {}'.format(far_a_middle_v_result['key2path'][best_sample]['hyp']))
    text_out.append('far_a_middle_v: {}'.format(far_a_middle_v_result['key2path'][best_sample]['op']))
    text_out.append('far_v: {}'.format(far_v_result['key2path'][best_sample]['hyp']))
    text_out.append('far_v: {}'.format(far_v_result['key2path'][best_sample]['op']))
    text_out.append('middle_v: {}'.format(middle_v_result['key2path'][best_sample]['hyp']))
    text_out.append('middle_v: {}'.format(middle_v_result['key2path'][best_sample]['op']))
    
    text2lines(textpath='./sample_{}/text'.format(best_sample), lines_content=text_out)
    # audio
    far_audio = torch.load(far_a_middle_v_result['key2path'][best_sample]['far_wave']).numpy()
    wavfile.write('./sample_{}/far.wav'.format(best_sample), 16000, far_audio)
    near_audio = torch.load(far_a_middle_v_result['key2path'][best_sample]['near_wave']).numpy()
    wavfile.write('./sample_{}/near.wav'.format(best_sample), 16000, near_audio)
    
    # far video
    far_lip = torch.load(far_a_middle_v_result['key2path'][best_sample]['far_lip']).numpy()[:, 32:64, 32:64, :]
    # far_gray_lip =  0.114 * far_lip[:, :, 0] + 0.587 * far_lip[:, :, 1] + 0.299 * far_lip[:, :, 2]
    os.makedirs('./sample_{}/far_lip'.format(best_sample), exist_ok=True)
    for i in tqdm(range(far_lip.shape[0])):
        cv2.imwrite('./sample_{}/far_lip/{}.jpg'.format(best_sample, i), cv2.resize(far_lip[i], (256, 256), interpolation=cv2.INTER_CUBIC))
    
    # middle video
    middle_lip = torch.load(far_a_middle_v_result['key2path'][best_sample]['middle_lip']).numpy()[:, :, :, :]
    # far_gray_lip =  0.114 * far_lip[:, :, 0] + 0.587 * far_lip[:, :, 1] + 0.299 * far_lip[:, :, 2]
    os.makedirs('./sample_{}/middle_lip'.format(best_sample), exist_ok=True)
    for i in tqdm(range(middle_lip.shape[0])):
        cv2.imwrite('./sample_{}/middle_lip/{}.jpg'.format(best_sample, i), cv2.resize(middle_lip[i], (256, 256), interpolation=cv2.INTER_CUBIC))
    
    

    
    
    
    
    
    
    