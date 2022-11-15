#!/usr/bin/env python
# _*_ coding: UTF-8 _*_
import os
import cv2
import json
import yaml
import torch
import codecs
import shutil
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from openpyxl import load_workbook
from .text_grid import TextGrid, read_textgrid_from_file, write_textgrid_to_file


def plot(figure_path, y_label, x_label, data):
    figure = plt.figure()
    os.makedirs(os.path.split(figure_path)[0], exist_ok=True)
    for k, x in data.tiers():
        if isinstance(x, list) or isinstance(x, np.ndarray):
            plt.plot(list(range(len(x))), x, label=k)
        elif isinstance(x, tuple):
            plt.plot(x[0], x[1], label=k)
        else:
            raise ValueError('unknown data value')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend(loc='upper right')
    plt.close('all')
    figure.savefig(figure_path, dpi=330, bbox_inches='tight')
    return None


def data2float(data):
    """
    convert dtype of a ndarray/tensor to float32
    :param data: ndarray or tensor
    :return: ndarray or tensor which dtype is float32
    """
    if isinstance(data, torch.Tensor):
        return data.float()
    elif isinstance(data, np.ndarray):
        return data.astype('float32')
    else:
        raise NotImplementedError('unknown data type')


def sph2numpy(sph_file):
    """
    read numpy array from sph file (wav in TIMIT)
    :param sph_file: filepath of sph
    :return: numpy array of sph file
    """
    with codecs.open(sph_file, 'rb') as sph_handle:
        sph_frames = sph_handle.read()
    np_array = np.frombuffer(sph_frames, dtype=np.int16, offset=1024)
    return np_array


def pcm2numpy(pcm_file):
    """
        read numpy array from pcm file (wav without head)
        :param pcm_file: filepath of pcm
        :return: numpy array of sph file
        """
    with codecs.open(pcm_file, 'rb') as pcm_handle:
        pcm_frames = pcm_handle.read()
    if len(pcm_frames) % 2 != 0:
        pcm_frames = pcm_frames[:-1]
    np_array = np.frombuffer(pcm_frames, dtype=np.int16, offset=0)
    return np_array


def text2lines(textpath, lines_content=None):
    """
    read lines from text or write lines to txt
    :param textpath: filepath of text
    :param lines_content: list of lines or None, None means read
    :return: processed lines content for read while None for write
    """
    if lines_content is None:
        with codecs.open(textpath, 'r') as handle:
            lines_content = handle.readlines()
        processed_lines = [*map(lambda x: x[:-1] if x[-1] in ['\n'] else x, lines_content)]
        return processed_lines
    else:
        processed_lines = [*map(lambda x: x if x[-1] in ['\n'] else '{}\n'.format(x), lines_content)]
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


def yaml2dic(yamlpath, dic=None):
    """
    read dic from yaml or write dic to yaml
    :param yamlpath: filepath of yaml
    :param dic: content dic or None, None means read
    :return: content dic for read while None for write
    """
    if dic is None:
        with codecs.open(yamlpath, 'r') as handle:
            return yaml.load(handle, Loader=yaml.FullLoader)
    else:
        with codecs.open(yamlpath, 'w') as handle:
            yaml.dump(dic, handle)
        return None


def video2numpy(videopath, frames_np=None, fps=25, is_color=True):
    """
    read numpy array from video (mp4 or avi) or write numpy array to video (only avi)
    :param videopath: filepath of video
    :param frames_np: numpy array of image frames or None, None means read
    :param fps: frame per second
    :param is_color: colorful image or gray image
    :return: numpy array for read while None for write
    """
    if frames_np is None:
        frames = []
        video_capture = cv2.VideoCapture(videopath)
        while video_capture.isOpened():
            ret, frame = video_capture.read()
            if ret:
                if not is_color:
                    frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))  # color of video which read by cv2 is BGR
                else:
                    frames.append(frame)
            else:
                break
        video_capture.release()
        frames_np = np.array(frames)
        return frames_np
    else:
        frames_np = frames_np.astype('uint8')
        frame_size = (frames_np.shape[2], frames_np.shape[1])
        frame_count = frames_np.shape[0]
        if len(frames_np.shape) == 3:
            is_color = False
        elif len(frames_np.shape) == 4:
            is_color = True
        else:
            raise ValueError('unknown shape for frames_np')
        out_writer = cv2.VideoWriter(
            videopath, cv2.VideoWriter_fourcc('X', 'V', 'I', 'D'), fps, frame_size, is_color)
        for frame_idx in range(frame_count):
            frame = frames_np[frame_idx]
            out_writer.write(frame)
        out_writer.release()
        return None


def excel2list(excelpath, max_row, max_col, min_row=1, min_col=1):
    workbook = load_workbook(filename=excelpath)
    sheet = workbook.active
    sheet_content = []
    for row in sheet.iter_rows(min_row=min_row, min_col=min_col, max_row=max_row, max_col=max_col):
        row_str = ''
        for cell in row:
            row_str += '{} '.format(cell.value)
        sheet_content.append(row_str[:-1])
    return sheet_content


def safe_store(file, data, mode='cover', ftype=None, **other_params):
    """
    integrate all existing write interface
    :param file: filepath to write
    :param data: data will be wrote
    :param mode: mode of write, ignore or cover
    :param ftype: file type
    :param other_params: reserved interface
    :return: None
    """
    def store_process(filename, content, **extra):
        store_dir = os.path.split(filename)[0]
        if not os.path.exists(store_dir):
            os.makedirs(store_dir, exist_ok=True)
        file_type = extra.get('ftype', os.path.split(filename)[-1].split('.')[-1])
        file_type = file_type.lower()
        if file_type == 'npy':
            np.save(filename, content)
        elif file_type == 'npz':
            if isinstance(content, dict):
                np.savez(filename, **content)
            else:
                np.savez(filename, content)
        elif file_type == 'json':
            json2dic(filename, dic=content)
        elif file_type == 'wav':
            fs = extra.get('fs', 16000)
            wavfile.write(filename, fs, content)
        elif file_type == 'yml':
            yaml2dic(filename, dic=content)
        elif file_type == 'avi':
            frame_per_second = extra.get('fps', 25)
            video2numpy(videopath=filename, frames_np=content, fps=frame_per_second)
        elif file_type in ['tar', 'pt', 'torch']:
            torch.save(content, filename)
        elif file_type in ['txt']:
            text2lines(textpath=filename, lines_content=content)
        elif file_type in ['textgrid']:
            write_textgrid_to_file(filepath=filename, textgrid=content)
        else:
            raise TypeError('unsupported store type')
        return None

    if ftype is None:
        pass
    else:
        other_params.update({'ftype': ftype})
    if os.path.exists(file):
        if mode == 'ignore':
            return False
        elif mode == 'cover':
            # os.remove(file)
            store_process(filename=file, content=data, **other_params)
        else:
            raise NotImplementedError('unknown mode')
    else:
        store_process(filename=file, content=data, **other_params)
    return None


def safe_copy(source, destination, keep_source=True, mode='cover', **other_params):
    """
    copy and paste file
    :param source: file be copy
    :param destination: file to paste
    :param keep_source: keep source or remove source
    :param mode: operation mode, ignore or cover
    :param other_params: reserved interface
    :return: None
    """
    def copy_process(origin, target, keep_origin, **extra):
        store_dir = os.path.split(target)[0]
        if not os.path.exists(store_dir):
            os.makedirs(store_dir, exist_ok=True)
        if keep_origin:
            shutil.copy(origin, target)
        else:
            shutil.move(origin, target)
        return None

    if not os.path.exists(source):
        raise FileExistsError('no source file')
    else:
        if os.path.exists(destination):
            if mode == 'ignore':
                return False
            elif mode == 'cover':
                # os.remove(destination)
                copy_process(origin=source, target=destination, keep_origin=keep_source, **other_params)
            else:
                raise NotImplementedError('unknown mode')
        else:
            copy_process(origin=source, target=destination, keep_origin=keep_source, **other_params)
    return None


def safe_load(file, ftype=None, **other_params):
    """
    integrate all existing read interface
    :param file: file be load
    :param ftype: file type
    :param other_params: reserved interface
    :return: file content
    """
    if not os.path.exists(file):
        raise FileExistsError('no such file {}'.format(file))
    else:
        file_type = os.path.split(file)[-1].split('.')[-1] if ftype is None else ftype
        file_type = file_type.lower()
        if file_type in ['npy', 'npz', 'numpy']:
            data = np.load(file, allow_pickle=True)
        elif file_type == 'json':
            data = json2dic(file)
        elif file_type == 'yml':
            data = yaml2dic(file)
        elif file_type == 'wav':
            _, data = wavfile.read(file)
        elif file_type == 'sph':
            data = sph2numpy(sph_file=file)
        elif file_type == 'pcm':
            data = pcm2numpy(pcm_file=file)
        elif file_type in ['avi', 'mp4']:
            is_color = other_params.get('is_color', True)
            data = video2numpy(videopath=file, is_color=is_color)
        elif file_type in ['pt', 'tar', 'torch']:
            data = torch.load(file, map_location=lambda storage, loc: storage)
        elif file_type in ['txt']:
            data = text2lines(textpath=file)
        elif file_type in ['xlsx']:
            assert 'max_row' in other_params and 'max_col' in other_params, 'lack of params: max_row and max_col'
            max_row, max_col = other_params['max_row'], other_params['max_col']
            min_row, min_col = other_params.get('min_row', 1), other_params.get('min_col', 1)
            data = excel2list(excelpath=file, max_row=max_row, max_col=max_col, min_row=min_row, min_col=min_col)
        elif file_type in ['textgrid']:
            data = read_textgrid_from_file(filepath=file)
        else:
            raise TypeError('unsupported file type')
    return data
