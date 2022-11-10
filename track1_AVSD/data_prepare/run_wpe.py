#!/usr/bin/env python
# _*_ coding: UTF-8 _*_
import os
import numpy as np
import scipy.io.wavfile as wf
from nara_wpe.wpe import wpe_v8 as wpe
from nara_wpe.utils import stft, istft
import sys
import argparse


parser = argparse.ArgumentParser() 
parser.add_argument('--session_list')
parser.add_argument('--audio_path')
parser.add_argument('--wpe_save_path')
args = parser.parse_args()


def do_wpe(input_list, output_list, WIN_LEN=5 * 60 * 16000):
    sampling_rate = 16000
    iterations = 5
    stft_options = dict(
        size=512,
        shift=128,
        window_length=None,
        fading=True,
        pad=True,
        symmetric_window=False
    )
    signal_list = []
    for f in input_list:
        _, data = wf.read(f)
        if data.dtype == np.int16:
            data = np.float32(data) / 32768
        signal_list.append(data)
    try:
        y = np.stack(signal_list, axis=0)
    except:
        mlen = len(signal_list[0])
        for i in range(1, len(signal_list)):
            mlen = min(mlen, len(signal_list[i]))
        for i in range(len(signal_list)):
            signal_list[i] = signal_list[i][:mlen]
        y = np.stack(signal_list, axis=0)
    z = []
    for s in range(0, y.shape[1], WIN_LEN):
        Y = stft(y[:, s:(s+WIN_LEN)], **stft_options).transpose(2, 0, 1)
        Z = wpe(Y, iterations=iterations, statistics_mode='full').transpose(1, 2, 0)
        z.append(istft(Z, size=stft_options['size'], shift=stft_options['shift']))
    z = np.hstack(z)
    for d, out_path in enumerate(output_list):
        tmpwav = np.int16(z[d,:] * 32768)
        wf.write(out_path, sampling_rate, tmpwav)


def prepare(sessions, dataset="MISP"):
    for l in sessions:
        data_root, output_root, session = l.split()
        if dataset=="MISP":  # The current code is only for far-field data. You can change the following code to adapt to other data.
            do_for_wpe = False
            input_list = []
            output_list = []
            for ch in range(6):
                audio_path = os.path.join(data_root, f"{session}_Far_{ch}.wav")
                if os.path.isfile(audio_path):
                    do_for_wpe = True
                    input_list.append(audio_path)
                    output_list.append(os.path.join(output_root, f"{session}_Far_{ch}.wav"))
            if do_for_wpe:
                print(f"Processing MISP: {session}")
                do_wpe(input_list, output_list)
        elif dataset=="AMI":
            for arr in range(1, 3):
                do_for_wpe = False
                input_list = []
                output_list = []
                for ch in range(1, 9):
                    audio_path = os.path.join(data_root, f"{session}.Array{arr}-0{ch}.wav")
                    if os.path.isfile(audio_path):
                        do_for_wpe = True
                        input_list.append(audio_path)
                        output_list.append(os.path.join(output_root, f"{session}.Array{arr}-0{ch}.wav"))
                if do_for_wpe:
                    print(f"Processing AMI: {session}.Array{arr}")
                    do_wpe(input_list, output_list)
    return None



if __name__ == '__main__':
    session = ""
    fileHandler  =  open  (args.session_list,  "r")
    while  True:
        line  =  fileHandler.readline()
        if  not  line  :
            break;
        session = line
        print(args.audio_path+" "+args.wpe_save_path+" "+session)
        prepare([args.audio_path+" "+args.wpe_save_path+" "+session], dataset="MISP")  
    fileHandler.close()


