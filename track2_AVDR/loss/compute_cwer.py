#!/usr/bin/env python
# _*_ coding: UTF-8 _*_
import argparse
import codecs
import json
import os

from tqdm import tqdm


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
    
    


def compute_cer(src_seq, tgt_seq):
    "计算编辑距离，并计算插入，删除，替换错误"
    "计算两个序列的编辑距离，用来计算字符错误率"
    insert = 0
    delete = 0
    substitute = 0
    l_src, l_tgt = len(src_seq), len(tgt_seq)
    if l_src == 0 or l_tgt == 0:
        return l_tgt, l_src, l_tgt, 0
    # construct matrix of size (l_src + 1, l_tgt + 1)
    dist = [[0] * (l_tgt + 1) for i in range(l_src + 1)]
    route = [[0,0] * (l_tgt + 1) for i in range(l_src + 1)]
    for i in range(1, l_tgt + 1):
        dist[0][i] = dist[0][i-1] + 1
    for i in range(1, l_src + 1):
        dist[i][0] = dist[i-1][0] + 1
    for i in range(1, l_src + 1):
        for j in range(1, l_tgt + 1):
            if src_seq[i - 1] == tgt_seq[j - 1]:
                cost = 0
            else:
                cost = 1
            dist[i][j] = min(dist[i][j-1] + 1, dist[i-1][j] + 1, dist[i-1][j-1] + cost)
            if dist[i][j] == dist[i-1][j]+1:
                route[i][j] = [i-1,j]
            elif dist[i][j] == dist[i][j-1]+1:
                route[i][j] = [i,j-1]
            elif dist[i][j] == dist[i-1][j-1]+cost:
                route[i][j] = [i-1, j-1]
    #路径回溯
    i = l_src
    j = l_tgt
    while i > 0 and j > 0:
        i_back = route[i][j][0]
        j_back = route[i][j][1]
        if j_back == j and (not i_back == i):
            insert = insert+abs(i-i_back)
        elif i_back == i and (not j_back == j):
            delete = delete + abs(j-j_back)
        elif i_back == i-1 and j_back == j-1:
            if not src_seq[i - 1] == tgt_seq[j - 1]:
                substitute += 1
        i = i_back
        j = j_back
        if i == 0 and j == 0:
            break
    if not i==0:
        insert = insert+abs(i)
    if not j==0:
        delete = delete+abs(j)            
    return l_tgt, insert, delete, substitute




def input_interface(src_file, tgt_file, result_file):
    src_lines = sorted(text2lines(textpath=src_file))
    src_dic = {}
    for src_line in src_lines:
        key, *value = src_line.split(' ')
        src_dic[key] = value
    
    tgt_lines = sorted(text2lines(textpath=tgt_file))
    tgt_dic = {}
    for tgt_line in tgt_lines:
        key, *value = tgt_line.split(' ')
        tgt_dic[key] = value
    
    result_dic = {}
    # all_result = {'entire': 0, 'insert': 0, 'delete': 0, 'substitute': 0}
    for key, tgt_list in tqdm(tgt_dic.items()):
        src_list = src_dic.get(key, [''])
        entire, insert, delete, substitute = compute_cer(src_seq=''.join(src_list), 
                                                         tgt_seq=''.join(tgt_list))
        result_dic[key] = {'entire': entire, 'insert': insert, 'delete': delete, 'substitute': substitute}
        # for j in ['entire', 'insert', 'delete', 'substitute']:
        #     all_result[j] += result_dic[key][j]
    
    # print('cer|error/entire|insert,delete,substitut')
    # print('{}|{}/{}|{},{},{}'.format(
    #     float(all_result['insert'] + all_result['delete'] + all_result['substitute']) / float(all_result['entire']),
    #     all_result['insert'] + all_result['delete'] + all_result['substitute'], all_result['entire'],
    #     all_result['insert'], all_result['delete'], all_result['substitute']))
    if result_file == 'default':
        result_file = os.path.join(os.path.split(src_file)[0], 'cer.json')
    else:
        if not os.path.exists(os.path.split(result_file)[0]):
                os.makedirs(os.path.split(result_file)[0], exist_ok=True)
    json2dic(jsonpath=result_file, dic=result_dic)
    return None


def cer_statistics(cer_json):
    cer_dic = json2dic(cer_json)
    all_result = {'entire': 0, 'insert': 0, 'delete': 0, 'substitute': 0}
    for key, value in cer_dic.items():
        for j in ['entire', 'insert', 'delete', 'substitute']:
            all_result[j] += value[j]
    result_lines = ['cer|error/entire|insert,delete,substitut', 
                    '{}|{}/{}|{},{},{}'.format(
                        float(all_result['insert'] + all_result['delete'] + all_result['substitute']) / float(all_result['entire']),
                        all_result['insert'] + all_result['delete'] + all_result['substitute'], all_result['entire'],
                        all_result['insert'], all_result['delete'], all_result['substitute'])]
    print(result_lines[0])
    print(result_lines[1])
    text2lines(textpath=os.path.join(os.path.split(cer_json)[0], 'RESULT'), lines_content=result_lines)
    return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser('compute_cwer')
    
    parser.add_argument('src_file', type=str, default='/yrfs1/intern/hangchen2/experiment/EASE',
                        help='directory of experiment')
    parser.add_argument('tgt_file', type=str, default='./exp_yml', help='directory of config yaml')
    
    parser.add_argument('-r', '--result_file', type=str, default='default', help='config id')
    parser.add_argument('-e', '--error_type', type=str, default='cer', help='config id')
    
    args = parser.parse_args()

    if args.error_type == 'cer':
        if args.result_file == 'default':
            result_file = os.path.join(os.path.split(args.src_file)[0], 'cer.json')
        else:
            result_file = args.result_file
        input_interface(src_file=args.src_file, tgt_file=args.tgt_file, result_file=result_file)
        cer_statistics(result_file)
    elif args.error_type == 'wer':
        raise NotImplementedError('waiting...')
    else:
        raise ValueError('unknown error type: {}'.format(args.error_type))
