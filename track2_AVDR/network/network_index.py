#!/usr/bin/env python
# _*_ coding: UTF-8 _*_
import torch
import argparse

from .network_model import BaseModelWorker

from .network_classifier import AudioClassifier, VisualClassifier, AudioVisualClassifier


class AVSRWorker(BaseModelWorker):
    def __init__(self, log_type, logger=None):
        super(AVSRWorker, self).__init__(log_type, logger)

    def _build_map(self):
        self.name2network = {
            'asr': AudioClassifier, 'vsr': VisualClassifier, 'avsr': AudioVisualClassifier
        }
        return None
