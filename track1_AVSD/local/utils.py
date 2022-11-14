#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import logging
import numpy as np
import torch
import os
import copy

def get_logger(filename):
    # Logging configuration: set the basic configuration of the logging system
    log_formatter = logging.Formatter(fmt='%(asctime)s [%(processName)s, %(process)s] [%(levelname)-5.5s]  %(message)s',
                                      datefmt='%m-%d %H:%M')
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    # File logger
    file_handler = logging.FileHandler("{}.log".format(filename))
    file_handler.setFormatter(log_formatter)
    file_handler.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    # Stderr logger
    std_handler = logging.StreamHandler(sys.stdout)
    std_handler.setFormatter(log_formatter)
    std_handler.setLevel(logging.DEBUG)
    logger.addHandler(std_handler)
    return logger


def cal_indicator(pre,label):
    TP = 0.0
    FN = 0.0
    TN = 0.0
    FP = 0.0
    TP += np.sum(pre==1.0 and label==1.0)
    FP += np.sum(pre==1.0 and label==-0.0)
    FN += np.sum(pre==-0.0 and label==1.0)
    TN += np.sum(pre==-0.0 and label==-0.0)
    '''
    for i, it in enumerate(pre):
        print(f"{it.shape} {label[i].shape}")
        TP += np.sum(it==1.0 and label[i][:, 1]==1.0)
        FP += np.sum(it==1.0 and label[i][:, 1]==-0.0)
        FN += np.sum(it==-0.0 and label[i][:, 1]==1.0)
        TN += np.sum(it==-0.0 and label[i][:, 1]==-0.0)
        if it == 1.0 and label[i] == 1.0:
            TP += 1.0
        elif it == 1.0 and label[i] == -0.0:
            FP += 1
        elif it == -0.0 and label[i] == 1.0:
            FN += 1
        elif it == -0.0 and label[i] == -0.0:
            TN += 1.0
    '''
    return TP, FP, TN, FN


#ANCHOR Checks of the directory exist and if not, creates a new directory
def checkdir(directory):
    try:
        os.makedirs(directory)
    except OSError:
        pass
