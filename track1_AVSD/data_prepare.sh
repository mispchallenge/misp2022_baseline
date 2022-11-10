#!/usr/bin/env bash

# Dereverberation for far-field audio, and extract ROI of face and lips

stage=1
set=dev  # Defining the set: train, dev, or eval.

audio_path=/export/corpus/misp2022/Released/audio/${set}/far/  # Defining original audio data path
wpe_save_path=wpe/${set}  # Defining path to save WPE result
session_list=scp_dir/${set}.list  # A list containing all sessions to be processed. Each line is the name of the session, like: R11_S193194_C08_I2

mkdir -p $wpe_save_path

if [ $stage -le 1 ]; then # WPE
    python data_prepare/run_wpe.py \
      --session_list $session_list \
      --audio_path $audio_path \
      --wpe_save_path $wpe_save_path
fi

video_dir=/export/corpus/misp2022/Released/video/${set}/far/  # Defining video data path
roi_json_dir=/export/corpus/misp2022/Released/roi_coordinates/${set}/far/lip/  # Defining detection_results path
roi_store_dir=detection_roi/${set}/far  # Defining path to save face and lip ROIs

mkdir -p $roi_store_dir

if [ $stage -le 2 ]; then # lip and face ROI
    python data_prepare/prepare_far_video_roi_speaker_diarization.py \
      --set $set \
      --video_dir $video_dir \
      --roi_json_dir $roi_json_dir \
      --roi_store_dir $roi_store_dir
fi
