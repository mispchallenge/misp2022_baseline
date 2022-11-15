#!/usr/bin/env python
# _*_ coding: UTF-8 _*_
import os
import argparse


def beamformit_worker(beamformit_tool, config_file, source_dir, channel_scp):
    f = open(channel_scp)
    for line in f:
        show_id = line.split(' ')[0]
        store_dir = os.path.split(line.split(' ')[1])[0]
        print('*' * 50)
        print(store_dir)
        print(show_id)
        print('*' * 50)
        if not os.path.exists(store_dir):
            os.makedirs(store_dir, exist_ok=True)
        cmd = '{} -s {} -c {} --config_file {} -source_dir {} --result_dir {}'.format(
            beamformit_tool, show_id, channel_scp, config_file, source_dir, store_dir)
        os.system(cmd)
    f.close()
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser('')
    parser.add_argument('beamformit_tool', type=str, default='',
                        help='path of beamformit tool')
    parser.add_argument('config_file', type=str, default='./conf/all_conf.cfg',
                        help='path of config file')
    parser.add_argument('source_dir', type=str, default='',
                        help='wpe data dir')
    parser.add_argument('channel_scp', type=str, default='',
                        help='path of config file')

    args = parser.parse_args()
    beamformit_worker(beamformit_tool=args.beamformit_tool, config_file=args.config_file, source_dir=args.source_dir,
                      channel_scp=args.channel_scp)
