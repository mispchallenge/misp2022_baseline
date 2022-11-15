import argparse
import numpy as np
import codecs
import os
import sys
import json
from tqdm import tqdm

def text2lines(textpath, lines_content=None,encoding='utf-8'):
    """
    read lines from text or write lines to txt
    :param textpath: filepath of text
    :param lines_content: list of lines or None, None means read
    :return: processed lines content for read while None for write
    """
   
    if lines_content is None:
        with codecs.open(textpath, 'r',encoding=encoding) as handle:
            lines_content = handle.readlines()
            processed_lines = []
            for line_content in lines_content:
                line_content = line_content.replace('\n', '').replace('\r', '')
                if line_content.strip():
                     processed_lines.append(line_content)
        return processed_lines
    else:
        processed_lines = [*map(lambda x: x if x[-1] in ['\n','\r'] else '{}\n'.format(x), lines_content)]
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


def gendict(src_file,tgt_file):
    src_lines = sorted(text2lines(textpath=src_file))
    src_dic = {}
    for src_line in src_lines:
        key,*value = src_line.split(' ')
        src_dic[key] = value

    tgt_lines = sorted(text2lines(textpath=tgt_file))
    tgt_dic = {}
    for tgt_line in tgt_lines:
        key,*value = tgt_line.split(' ')
        tgt_dic[key] = value
    
    # speakers may miss
    for key in tgt_dic.keys():   
        if key not in src_dic.keys():
            src_dic[key] = '' 

    return src_dic,tgt_dic

def gendict_new(src_content,tgt_content):
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
    
    # speakers may miss
    for key in tgt_dic.keys():   
        if key not in src_dic.keys():
            src_dic[key] = '' 

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


def compute_cer(src_dic,tgt_dic):
    # result_dic = {}
    sum_entire = 0
    sum_insert = 0
    sum_delete = 0
    sum_substitute = 0
    result_dic = {}
    

    #check input keys, it must include reference keys 
    if not sorted(list(src_dic.keys()))==sorted(list(set(tgt_dic.keys()) & set((src_dic.keys())))):
        watch_out = "In your reftext, some segments are missing.!"
    else: watch_out = ""
    
    miss_utter = []
    for key,src_list in tqdm(list(src_dic.items())): #只看src拥有的
        tgt_list =  tgt_dic.get(key,None)
        if tgt_list:
            entire,insert,delete,substitute = compute_cer_web(src_seq=''.join(src_list),tgt_seq=''.join(tgt_list))
            result_dic[key] = {'entire': entire, 'insert': insert, 'delete': delete, 'substitute': substitute}
            sum_entire += entire
            sum_insert += insert
            sum_delete += delete
            sum_substitute += substitute
            # result_dic[key] ={'entire':entire,'insert':insert,'delete':delete,'substitute':substitute}
        else:
            miss_utter.append(key)
    return [(sum_insert+sum_delete+sum_substitute)/sum_entire,sum_entire,sum_insert,sum_delete,sum_substitute],watch_out,miss_utter, result_dic

def deal_with_result_decode_and_text(src_path, ref_path):
    content_result_decode = []
    content_text = []
    with open(src_path, 'r', encoding='utf-8') as f_rd:
        for line in f_rd.readlines():
            line = line.strip()
            content_result_decode.append(line)
    with open(ref_path, 'r', encoding='utf-8') as f_t:
        for line in f_t.readlines():
            line = line.strip()
            content_text.append(line)
    content_result_decode = sorted(content_result_decode)
    content_text = sorted(content_text)

    output_from_result_decode = []
    output_from_text = []
    for line in content_result_decode:
        file_name = "_".join(line.split(" ")[0].split("_")[:-1])
        if len(output_from_result_decode) == 0 or output_from_result_decode[-1].split(" ")[0] != file_name:
            # import pdb;pdb.set_trace()
            output_from_result_decode.append(" ".join([file_name] + line.split(" ")[1:]))
        else:
            output_from_result_decode[-1] += " ".join(line.split(" ")[1:])
    for line in content_text:
        file_name = "_".join(line.split(" ")[0].split("_")[:-1])
        if len(output_from_text) == 0 or output_from_text[-1].split(" ")[0] != file_name:
            # import pdb;pdb.set_trace()
            output_from_text.append(" ".join([file_name] + line.split(" ")[1:]))
        else:
            output_from_text[-1] += " ".join(line.split(" ")[1:])

    return output_from_result_decode, output_from_text

if __name__ == '__main__':    
    parser = argparse.ArgumentParser('cer')
    parser.add_argument('-s', '--src_path', type=str, default='data/dev_far_audio/result_decode.txt',
                       help='directory of experiment')
    parser.add_argument('-r', '--ref_path', type=str, default='data/dev_far_audio/text',
                        help='directory of experiment')
    args = parser.parse_args()
    src_path = args.src_path
    ref_path = args.ref_path
    output_from_result_decode, output_from_text = deal_with_result_decode_and_text(args.src_path, args.ref_path)
    src_dict,ref_dict = gendict_new(output_from_result_decode, output_from_text)
    result,watch_out,miss_utter,result_dic = compute_cer(src_dict,ref_dict)
    if watch_out:
        print(watch_out)
        print("missing segments:")
        for i in miss_utter:
            print(i)
    score_path = os.path.join(os.path.split(src_path)[0], 'score.txt')
    with open(score_path,"w") as f:
         print("CCER: {:.4f}".format(result[0]),file=f)
    print("CCER: {:.4f}".format(result[0]))
    json2dic(jsonpath=os.path.join(os.path.split(src_path)[0], 'cer.json'), dic=result_dic)
    
