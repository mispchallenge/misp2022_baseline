#!/usr/bin/env python
# _*_ coding: UTF-8 _*_
import os
import glob
import argparse


def search(root_dir, regular_expression):
    search_expression = os.path.join(root_dir, regular_expression)
    if search_expression[-1] != '*':
        search_expression = search_expression + '*'
    possible_results_list = glob.glob(search_expression)
    return possible_results_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser('match_search')
    parser.add_argument('root_dir', type=str, default='/yrfs1/intern/hangchen2/experiment/EASE',
                        help='root directory for search')
    parser.add_argument('regular_expression', type=str, default='5_1', help='regular expression for match')
    args = parser.parse_args()
    
    result_candidates = search(root_dir=args.root_dir, regular_expression=args.regular_expression)
    
    if len(result_candidates) == 0:
        print('')
    elif len(result_candidates) > 1:
        print('|'.join(result_candidates))
    else:
        print(result_candidates[0])
