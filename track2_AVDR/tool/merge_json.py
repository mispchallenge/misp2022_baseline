#!/usr/bin/env python
# _*_ coding: UTF-8 _*_
import os
import json
import codecs
import argparse


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser('merge_json')
    parser.add_argument('-o', '--output_json', type=str, default='./output.json', help='directory of experiment')
    parser.add_argument('input_json', type=str, nargs='+', default=['train'], help='select run mode')
    args = parser.parse_args()
    
    output_dic = {'keys': [], 'duration': [], 'key2path': {}}
    for jsonpath in args.input_json:
        input_dic = json2dic(jsonpath=jsonpath)
        output_dic['keys'] = output_dic['keys'] + input_dic['keys']
        output_dic['duration'] = output_dic['duration'] + input_dic['duration']
        output_dic['key2path'] = {**output_dic['key2path'], **input_dic['key2path']}
    if not os.path.exists(os.path.split(args.output_json)[0]):
            os.makedirs(os.path.split(args.output_json)[0], exist_ok=True)
    json2dic(jsonpath=args.output_json, dic=output_dic)
    