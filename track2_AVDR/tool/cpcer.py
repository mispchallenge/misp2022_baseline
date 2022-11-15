import argparse
import numpy as np
import codecs
import os
import sys
import json
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment

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

def gendict_new_cpCER(src_content,tgt_content):
    src_lines = sorted(src_content)
    src_dic = {}
    for src_line in src_lines:
        key,*value = src_line.split(' ')
        src_dic[key] = value

    tgt_lines = sorted(tgt_content)
    tgt_dic = {}
    for tgt_line in tgt_lines:
        key,*value = tgt_line.split(' ')
        tgt_dic[key] = value
    
    if len(src_content) > len(tgt_content):
        raise ValueError('The number of speakers should be <= ground truth.')
    if len(src_content) < len(tgt_content):
        # speakers may miss
        for i in range(len(tgt_content)-len(src_content)):   
            sessionid =  "_".join(list(src_dic.keys())[0].split("_")[1:])
            spkid_sessionid = str(-1-i) + "_" + sessionid
            src_dic[spkid_sessionid] = [] 

    return src_dic,tgt_dic

def compute_cer_web(src_seq,tgt_seq):
    src_seq = src_seq.replace("<UNK>","*")
    hypothesis = list(src_seq)
    reference = list(tgt_seq)
    len_hyp = len(hypothesis)
    len_ref = len(reference)
    cost_matrix = np.zeros((len_hyp + 1, len_ref + 1), dtype=np.int16)

  
    ops_matrix = np.zeros((len_hyp + 1, len_ref + 1), dtype=np.int8)

    for i in range(len_hyp + 1):
        cost_matrix[i][0] = i
    for j in range(len_ref + 1):
        cost_matrix[0][j] = j


    for i in range(1, len_hyp + 1):
        for j in range(1, len_ref + 1):
            if hypothesis[i-1] == reference[j-1]:
                cost_matrix[i][j] = cost_matrix[i-1][j-1]
            else:
                substitution = cost_matrix[i-1][j-1] + 1
                insertion = cost_matrix[i-1][j] + 1
                deletion = cost_matrix[i][j-1] + 1
                compare_val = [substitution, insertion, deletion]   

                min_val = min(compare_val)
                operation_idx = compare_val.index(min_val) + 1
                cost_matrix[i][j] = min_val
                ops_matrix[i][j] = operation_idx

    match_idx = []  
    i = len_hyp
    j = len_ref
    nb_map = {"N": len_ref, "C": 0, "W": 0, "I": 0, "D": 0, "S": 0}
    while i >= 0 or j >= 0:
        i_idx = max(0, i)
        j_idx = max(0, j)

        if ops_matrix[i_idx][j_idx] == 0:     # correct
            if i-1 >= 0 and j-1 >= 0:
                match_idx.append((j-1, i-1))
                nb_map['C'] += 1
           
            i -= 1
            j -= 1
        elif ops_matrix[i_idx][j_idx] == 2:   # insert
            i -= 1
            nb_map['I'] += 1
        elif ops_matrix[i_idx][j_idx] == 3:   # delete
            j -= 1
            nb_map['D'] += 1
        elif ops_matrix[i_idx][j_idx] == 1:   # substitute
            i -= 1
            j -= 1
            nb_map['S'] += 1

        
        if i < 0 and j >= 0:
            nb_map['D'] += 1
        elif j < 0 and i >= 0:
            nb_map['I'] += 1

    match_idx.reverse()
    wrong_cnt = cost_matrix[len_hyp][len_ref]
    nb_map["W"] = wrong_cnt

    return nb_map["N"],nb_map["I"],nb_map["D"],nb_map["S"] 


def compute_cpCER(src_dic,tgt_dic):
    length = len(tgt_dic)
    cost_matrix = np.zeros((length, length))
    src_list = sorted(list(src_dic.values()))
    tgt_list = sorted(list(tgt_dic.values()))
    for i in range(length):
        for j in range(length):
            entire,insert,delete,substitute = compute_cer_web(src_seq=''.join(src_list[i]),tgt_seq=''.join(tgt_list[j]))
            cost_matrix[i][j] = (insert + delete + substitute)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    cost = cost_matrix[row_ind, col_ind].sum()
    entire = 0
    for i in range(length):
        entire += len(list(''.join(tgt_list[i])))
    return cost, entire

if __name__ == '__main__':    
    parser = argparse.ArgumentParser('cer')
    parser.add_argument('-s', '--src_directory', type=str, default='data/dev_far_audio/Result',
                       help='directory of experiment')
    parser.add_argument('-r', '--ref_directory', type=str, default='data/dev_far_audio/Ground_Truth',
                        help='directory of experiment')
    parser.add_argument('-f', '--score_file', type=str, default='data/dev_far_audio/score.txt',
                        help='directory of experiment')
    args = parser.parse_args()
    src_file_list = os.listdir(args.src_directory)
    ref_file_list = os.listdir(args.ref_directory)
    assert len(src_file_list) == len(ref_file_list), "The number of the sessions doesn't match. You may omit some sessions!"
    
    cost_total, entire_total = 0, 0
    for file in ref_file_list:
        src_path = os.path.join(args.src_directory, file)
        ref_path = os.path.join(args.ref_directory, file)
        content_src, content_ref = [], []
        with open(src_path, 'r', encoding='utf-8') as f_src:
            for line in f_src.readlines():
                line = line.strip()
                content_src.append(line)
        with open(ref_path, 'r', encoding='utf-8') as f_ref:
            for line in f_ref.readlines():
                line = line.strip()
                content_ref.append(line)
        content_src = sorted(content_src)
        content_ref = sorted(content_ref)
        src_dict,ref_dict = gendict_new_cpCER(content_src, content_ref)
        cost, entire = compute_cpCER(src_dict,ref_dict)
        cost_total += cost
        entire_total += entire
    cpCER = cost_total/entire_total
    score_path = args.score_file
    with open(score_path,"w") as f:
         print("cpCER: {:.4f}".format(cpCER), file=f)
    
    print("cpCER: {:.4f}".format(cpCER))
    