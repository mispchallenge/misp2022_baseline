# -*- coding: utf-8 -*-

import numpy
import struct


def readHtk(filename):
    '''
    Reads the features in a HTK file, and returns them in a 2-D numpy array.
    '''
    with open(filename, "rb") as f:
        # Read header
        nSamples, sampPeriod, sampSize, parmKind = struct.unpack(">iihh", f.read(12))
        # sampPeriod and parmKind will be omitted
        # Read data
        data = struct.unpack(">%df" % (nSamples * sampSize / 4), f.read(nSamples * sampSize))
        # return numpy.array(data).reshape(nSamples, int(sampSize / 4))
        return nSamples, sampPeriod, sampSize, parmKind, data

def readHtk_start_end(filename, start, end):
    with open(filename, "rb") as f:
        # Read header
        nSamples, sampPeriod, sampSize, parmKind = struct.unpack(">iihh", f.read(12))
        # sampPeriod and parmKind will be omitted
        f.seek(start * sampSize,1)
        # Read data
        data = struct.unpack(">%df" % ((end - start) * sampSize / 4), f.read((end - start) * sampSize))
        # return numpy.array(data).reshape(nSamples, int(sampSize 1 4))
        return nSamples, sampPeriod, sampSize, parmKind, data

def readHtk_info(filename):
    with open(filename, "rb") as f:
        # Read header
        nSamples, sampPeriod, sampSize, parmKind = struct.unpack(">iihh", f.read(12))
        return nSamples, sampPeriod, sampSize, parmKind

def writeHtk(filename, feature, sampPeriod, parmKind):
    '''
    Writes the features in a 2-D numpy array into a HTK file.
    '''
    with open(filename, "wb") as f:
        # Write header
        nSamples = feature.shape[0]
        sampSize = feature.shape[1] * 4
        f.write(struct.pack(">iihh", nSamples, sampPeriod, sampSize, parmKind))
        # Write data
        f.write(struct.pack(">%df" % (nSamples * sampSize / 4), *feature.ravel()))

def writeHtk3D(filename, feature, parmKind=9):
    '''
    Writes the features in a 3-D numpy array into a HTK file.
    nSamples * W * H
    '''
    sampPeriod = feature.shape[1]
    with open(filename, "wb") as f:
        # Write header
        nSamples = feature.shape[0]
        feature = numpy.array(feature, dtype=numpy.uint8).reshape(nSamples, -1)
        sampSize = feature.shape[1]
        f.write(struct.pack(">iihh", nSamples, sampPeriod, sampSize, parmKind))
        for n in range(feature.shape[0]):
            f.write(struct.pack(">%dB" % (1 * sampSize), *feature[n, :].ravel()))

def readHtk3D(filename):
    '''
    Reads the features in a HTK file, and returns them in a 2-D numpy array.
    '''
    with open(filename, "rb") as f:
        # Read header
        nSamples, sampPeriod, sampSize, parmKind = struct.unpack(">iihh", f.read(12))
        # sampPeriod and parmKind will be omitted
        # Read data
        data = struct.unpack(">%dB" % (nSamples * sampSize), f.read(nSamples * sampSize))
        data = numpy.array(data).reshape(nSamples, sampPeriod, -1)
        # return numpy.array(data).reshape(nSamples, int(sampSize / 4))
        #return nSamples, sampPeriod, sampSize, parmKind, data
        return data

def readHtk_start_end3D(filename, start, end):
    with open(filename, "rb") as f:
        # Read header
        nSamples, sampPeriod, sampSize, parmKind = struct.unpack(">iihh", f.read(12))
        # sampPeriod and parmKind will be omitted
        f.seek(start * sampSize, 1)
        # Read data
        data = struct.unpack(">%dB" % ((end - start) * sampSize), f.read((end - start) * sampSize))
        # return numpy.array(data).reshape(nSamples, int(sampSize 1 4))
        data = numpy.array(data).reshape((end - start), sampPeriod, -1)
        #return nSamples, sampPeriod, sampSize, parmKind, data
        return data

#/disk2/mkhe/data/misp2021/detection_lip/train/middle/R03_S074075076_C06_I0_Middle_075-1-37241.htk