# -*- coding: utf-8 -*-

import numpy as np
import os
import argparse
import json


def write_rttm(session_label, output_path, fps=100):
    with open(output_path, "w") as OUT:
        for session in session_label.keys():
            for spk in session_label[session].keys():
                labels = session_label[session][spk]
                to_split = np.nonzero(labels[1:] != labels[:-1])[0]
                to_split += 1
                if labels[-1] == 1:
                    to_split = np.r_[to_split, len(labels)+1]
                if labels[0] == 1:
                    to_split = np.r_[0, to_split]
                for l in to_split.reshape(-1, 2):
                    #print(l)
                    #break
                    OUT.write("SPEAKER {} 1 {:.2f} {:.2f} <NA> <NA> {} <NA> <NA>\n".format(session, l[0]/fps, (l[1]-l[0])/fps, spk))

class Segmentation(object):
    """Stores segmentation for an utterances"""
    "this code comes from steps/segmentation/internal/sad_to_segments.py (reversed)"
    def __init__(self):
        self.segments = None
        self.filter_short_duration = 0

    def initialize_segments(self, rttm):
        self.segments = {}
        with open(rttm, 'r') as INPUT:
            for line in INPUT:
                line = line.split(" ")
                for l in line:
                    if l == "": line.remove(l)
                session = line[1]
                if line[-2] != "<NA>":
                    spk = line[-2]
                else:
                    spk = line[-3]
                if not session in self.segments.keys():
                    self.segments[session] = {}
                if not spk in self.segments[session].keys():
                    self.segments[session][spk] = []
                start = float(line[3])
                end = start + float(line[4])
                self.segments[session][spk].append([start, end])

    def filter_short_segments(self, min_dur):
        """Filters out segments with durations shorter than 'min_dur'."""
        if min_dur <= 0:
            return

        segments_kept = {}
        for session in self.segments.keys():
            segments_kept[session] = {}
            for spk in self.segments[session].keys():
                segments_kept[session][spk] = []
                for segment in self.segments[session][spk]:
                    dur = segment[1] - segment[0]
                    if dur < min_dur:
                        self.filter_short_duration += dur
                    else:
                        segments_kept[session][spk].append(segment)
        self.segments = segments_kept

    def pad_speech_segments(self, segment_padding, max_duration=float("inf")):
        """Pads segments by duration 'segment_padding' on either sides, but
        ensures that the segments don't go beyond the neighboring segments
        or the duration of the utterance 'max_duration'."""
        if segment_padding <= 0:
            return
        if max_duration == None:
            max_duration = float("inf")
        
        for session in self.segments.keys():
            for spk in self.segments[session].keys():
                for i, segment in enumerate(self.segments[session][spk]):
                    segment[0] -= segment_padding  # try adding padding on the left side
                    if segment[0] < 0.0:
                        # Padding takes the segment start to before the beginning of the utterance.
                        # Reduce padding.
                        segment[0] = 0.0
                    if i >= 1 and self.segments[session][spk][i - 1][1] > segment[0]:
                        # Padding takes the segment start to before the end the previous segment.
                        # Reduce padding.
                        segment[0] = self.segments[session][spk][i - 1][1]

                    segment[1] += segment_padding
                    if segment[1] >= max_duration:
                        # Padding takes the segment end beyond the max duration of the utterance.
                        # Reduce padding.
                        segment[1] = max_duration
                    if (i + 1 < len(self.segments[session][spk])
                            and segment[1] > self.segments[session][spk][i + 1][0]):
                        # Padding takes the segment end beyond the start of the next segment.
                        # Reduce padding.
                        segment[1] = self.segments[session][spk][i + 1][0]

    def merge_consecutive_segments(self, max_dur):
        """Merge consecutive segments (happens after padding), provided that
        the merged segment is no longer than 'max_dur'."""
        if max_dur <= 0 or not self.segments:
            return

        merged_segments = {}
        for session in self.segments.keys():
            merged_segments[session] = {}
            for spk in self.segments[session].keys():
                merged_segments[session][spk] = [self.segments[session][spk][0]]
                for segment in self.segments[session][spk][1:]:
                    #if segment[0] == merged_segments[session][spk][-1][1] and \
                    #        segment[1] - merged_segments[session][spk][-1][0] <= max_dur:
                    if segment[0] - merged_segments[session][spk][-1][1] <= max_dur:
                        # The segment starts at the same time the last segment ends,
                        # and the merged segment is shorter than 'max_dur'.
                        # Extend the previous segment.
                        merged_segments[session][spk][-1][1] = segment[1]
                    else:
                        merged_segments[session][spk].append(segment)

        self.segments = merged_segments

    def write_rttm(self, output_path):
        OUTPUT = open(output_path, 'w')
        for session in self.segments.keys():
            for spk in self.segments[session].keys():
                for segment in self.segments[session][spk]:
                    OUTPUT.write("SPEAKER {} 1 {:.2f} {:.2f} <NA> <NA> {} <NA> <NA>\n".format(session, segment[0], segment[1]-segment[0], spk))

def main(args):
    if os.path.isfile(args.session2spk):
        with open(args.session2spk) as IN:
            session2spk = json.load(IN)
    prob_array = [l for l in os.listdir(args.prob_array_dir)]
    MAX_DUR = {}
    for l in prob_array:
        if l.find(".npy") == -1: continue
        # R10_S186187188_C09_I2_Far_186-0-200
        # R10_S186187188_C09_I2-0-3200
        if len(l.split("_")) > 4:
            session = "_".join(l.split("_")[:-2])
        else:
            session = l.split("-")[0]
        end = int(l.split(".")[0].split("-")[-1])
        if session not in MAX_DUR.keys():
            MAX_DUR[session] = 0
        if MAX_DUR[session] < end:
            MAX_DUR[session] = end
    
    session_label = {}
    for l in prob_array:
        if l.find(".npy") == -1: continue
        # R10_S186187188_C09_I2_Far_186-0-200

        # R10_S186187188_C09_I2-0-3200
        if len(l.split("_")) > 4:
            session = "_".join(l.split("_")[:-2])
            speaker, start, end = l.split("_")[-1].split(".")[0].split("-")
            start = int(start)
            end = int(end)
            if session not in session_label.keys():
                session_label[session] = {}
            if speaker not in session_label[session].keys():
                session_label[session][speaker] = np.zeros(MAX_DUR[session]+100)
            prob_label = np.load(os.path.join(args.prob_array_dir, l))
            session_label[session][speaker][start:end][prob_label[:(end-start), 1] > args.threshold] = 1
        else:
            session = l.split("-")[0]
            start, end = l.split(".")[0].split("-")[-2:]
            start = int(int(start) / 100 * args.fps)
            end = int(int(end) / 100 * args.fps)
            prob_label = np.load(os.path.join(args.prob_array_dir, l))
            if session not in session_label.keys():
                session_label[session] = {}
                for spk in session2spk[session]:
                    session_label[session][spk] = np.zeros(MAX_DUR[session]+100)
            prob_label = np.load(os.path.join(args.prob_array_dir, l))
            for i, spk in enumerate(session2spk[session]):
                session_label[session][spk][start:end][prob_label[i, :(end-start), 1] > args.threshold] = 1
    write_rttm(session_label, os.path.join(args.rttm_dir, "rttm_th{:.2f}".format(args.threshold)), args.fps)
    segmentation = Segmentation()
    segmentation.initialize_segments(os.path.join(args.rttm_dir, "rttm_th{:.2f}".format(args.threshold)))
    segmentation.merge_consecutive_segments(args.max_dur)
    segmentation.filter_short_segments(args.min_dur)
    segmentation.write_rttm(os.path.join(args.rttm_dir, "rttm_th{:.2f}_pp".format(args.threshold)))


def make_argparse():
    # Set up an argument parser.
    parser = argparse.ArgumentParser(description='Prepare ivector extractor weights for ivector extraction.')
    parser.add_argument('--threshold', metavar='Float', type=float, default=0.5,
                        help='threshold.') 
    parser.add_argument('--min_dur', metavar='Float', type=float, default=0.0,
                        help='min_dur.') 
    parser.add_argument('--segment_padding', metavar='Float', type=float, default=0.0,
                        help='segment_padding.') 
    parser.add_argument('--max_dur', metavar='Float', type=float, default=0.0,
                        help='max_dur.') 
    parser.add_argument('--prob_array_dir', metavar='DIR', required=True,
                        help='prob_array_dir.')
    parser.add_argument('--session2spk', metavar='DIR', default="None",
                        help='session2spk.')
    parser.add_argument('--rttm_dir', metavar='DIR', required=True,
                        help='rttm_dir.')       
    parser.add_argument('--min_segments', metavar='DIR', type=int, default=0,
                        help='min_segments.')
    parser.add_argument('--fps', metavar='DIR', type=int, default=100,
                        help='min_segments.')
    return parser


if __name__ == '__main__':
    parser = make_argparse()
    args = parser.parse_args()
    main(args)
