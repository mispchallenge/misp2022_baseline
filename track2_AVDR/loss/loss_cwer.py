#!/usr/bin/env python
# _*_ coding: UTF-8 _*_
import json
import codecs
import argparse
import numpy as np


def compute_edit_distance(hypothesis: list, reference: list):
    insert, delete, substitute = 0, 0, 0
    correct = 0
    len_hyp, len_ref = len(hypothesis), len(reference)
    
    if len_hyp == 0 or len_ref ==0:
        return len_ref, len_hyp, len_ref, 0
    
    cost_matrix = np.zeros((len_hyp + 1, len_ref + 1), dtype=np.int16)
    # record all process，0-equal；1-insertion；2-deletion；3-substitution
    ops_matrix = np.zeros((len_hyp + 1, len_ref + 1), dtype=np.int8)
    for i in range(len_hyp + 1):
        cost_matrix[i][0] = i
    for j in range(len_ref + 1):
        cost_matrix[0][j] = j
        
     # create cost matrix and operation matrix, i: hyp, j: ref
    for i in range(1, len_hyp + 1):
        for j in range(1, len_ref + 1):
            if hypothesis[i-1] == reference[j-1]:
                cost_matrix[i][j] = cost_matrix[i-1][j-1]
            else:
                substitution = cost_matrix[i-1][j-1] + 1
                insertion = cost_matrix[i-1][j] + 1
                deletion = cost_matrix[i][j-1] + 1
                compare_val = [substitution, insertion, deletion]   # priority

                min_val = min(compare_val)
                operation_idx = compare_val.index(min_val) + 1
                cost_matrix[i][j] = min_val
                ops_matrix[i][j] = operation_idx
    
    match_idx = []  # save all aligned element subscripts in hyp and ref
    i = len_hyp
    j = len_ref
    while i >= 0 or j >= 0:
        i_idx = max(0, i)
        j_idx = max(0, j)

        if ops_matrix[i_idx][j_idx] == 0:     # correct
            if i-1 >= 0 and j-1 >= 0:
                match_idx.append((j-1, i-1))
                correct += 1
            i -= 1
            j -= 1
        elif ops_matrix[i_idx][j_idx] == 2:   # insert
            i -= 1
            insert += 1
        elif ops_matrix[i_idx][j_idx] == 3:   # delete
            j -= 1
            delete += 1
        elif ops_matrix[i_idx][j_idx] == 1:   # substitute
            i -= 1
            j -= 1
            substitute += 1

        if i < 0 and j >= 0:
            delete += 1
        elif j < 0 and i >= 0:
            insert += 1

    match_idx.reverse()
    error = cost_matrix[len_hyp][len_ref]
    return len_ref, insert, delete, substitute


def mlf2dic(mlf_path):
    results = {}
    fi = codecs.open(mlf_path, "r")
    lines = fi.readlines()
    word_list = []
    utt_name=''
    for line_idx in range(len(lines)):
        line = lines[line_idx].strip()
        if '.rec' in line:
            word_list = []
            utt_name = line.replace('"', '').replace('.rec', '')
        elif line == '.':
            if lines[line_idx + 1].strip() == '.':
                results[utt_name] = word_list
        else:
            word_list.append(line)
    fi.close()
    return results


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
