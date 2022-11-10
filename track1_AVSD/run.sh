#!/usr/bin/env bash

stage=6
gpu="0,1"  # Defining GPU
set=dev  # Defining the set: train, dev, or eval.

. ./utils/parse_options.sh

################################################################################
# single channel
################################################################################
ch=0  # Defining the audio channel

lip_train_scp=scp_dir/train.lip.scp  # # A file pointing to the training data lip ROIs
rttm_train=scp_dir/train_far_RTTM.rttm    # The oracle_RTTM file combining all training data sessions

lip_decode_scp=scp_dir/${set}.lip.scp  # A file pointing to the lip ROIs
oracle_vad=scp_dir/${set}_far_timestamp.lab  # The oracle_VAD timestamp file combining all sessions
oracle_rttm=scp_dir/${set}_far_RTTM.rttm  # The oracle_RTTM file combining all sessions
data=MISP2022_${set}_Far${ch}_WPE # Defining the data name
wav_dir=wpe/${set}/  # Defining the path of audio after WPE

vsd_model_path=exp/checkpoints/vsd/conformer_v_sd_2.model  # VSD model
vsd_prob_dir=exp/result_vsd/$set/prob
vsd_embedding_output_dir=exp/result_vsd/$set/vemb
vsd_output_rttm_dir=exp/result_vsd/$set/rttm

ivector_dir=exp/nnet3_cnceleb_ivector  # ivector output path
ivector_train=$ivector_dir/ivectors_misp2022_train_far_raw+wpe+enh/ivectors_spk.txt  # ivector of training data

avsd_model_path=exp/checkpoints/avsd/av_diarization_3.model  # AVSD model
avsd_prob_dir=exp/result_avsd/$data/prob
avsd_output_rttm_dir=exp/result_avsd/$data/rttm

if [ $stage -le 1 ]; then  # VSD decoder
    mkdir -p $vsd_prob_dir
    mkdir -p $vsd_embedding_output_dir
    CUDA_VISIBLE_DEVICES=$gpu python local/decode_VSD.py \
        --model_path $vsd_model_path \
        --prob_dir $vsd_prob_dir \
        --embedding_output_dir $vsd_embedding_output_dir \
        --lip_train_scp $lip_train_scp \
        --rttm_train $rttm_train \
        --lip_decode_scp $lip_decode_scp
fi

if [ $stage -le 2 ]; then  # get RTTM file and DER
    system=vsd
    local/prob_to_rttm.sh --system $system\
                          --set $set \
                          --prob_dir $vsd_prob_dir \
                          --oracle_vad $oracle_vad \
                          --rttm_dir $vsd_output_rttm_dir \
                          --oracle_rttm $oracle_rttm --fps 25
fi

if [ $stage -le 3 ]; then  # extract ivector
    mkdir -p $ivector_dir
    mkdir -p data/$data
    find $wav_dir | grep -E "_${ch}\.wav" > data/$data/wav.list
    awk -F "/" '{print $NF}' data/$data/wav.list | \
            awk -F "_" '{print $1"_"$2"_"$3"_"$4}' > data/$data/utt
    paste -d " " data/$data/utt data/$data/wav.list > data/$data/wav.scp
    paste -d " " data/$data/utt data/$data/utt > data/$data/utt2spk
    utils/fix_data_dir.sh data/$data
    local/extract_feature.sh --stage 1 --ivector_dir $ivector_dir \
        --data $data --rttm $vsd_output_rttm_dir/rttm_th0.50_pp_oraclevad \
        --max_speaker 6 --affix _VSD
fi

if [ $stage -le 4 ]; then  # AVSD decoder
    
    CUDA_VISIBLE_DEVICES=0,1 python local/decode_AVSD.py \
        --vsd_model_path $vsd_model_path \
        --avsd_model_path $avsd_model_path \
        --lip_train_scp $lip_train_scp \
        --rttm_train $rttm_train \
        --lip_decode_scp $lip_decode_scp \
        --ivector_train $ivector_train \
        --ivector_decode $ivector_dir/ivectors_${data}_VSD/ivectors_spk.txt \
        --fbank_dir fbank/HTK/${data}_cmn_slide \
        --prob_dir $avsd_prob_dir
fi
system=avsd
if [ $stage -le 5 ]; then  # get RTTM file and DER
    local/prob_to_rttm.sh --system $system\
                    --set $set \
                    --prob_dir $avsd_prob_dir/av_sd \
                    --oracle_vad $oracle_vad \
                    --session2spk $avsd_prob_dir/session2speaker \
                    --rttm_dir $avsd_output_rttm_dir \
                    --ch ch$ch \
                    --oracle_rttm $oracle_rttm --fps 100
fi

################################################################################
# Fusion of 6-channels using dover-lap
################################################################################

if [ $stage -le 6 ]; then
      #### If you did not execute the previous steps, you need to execute the content in the note
      #mkdir -p $vsd_prob_dir
      #mkdir -p $vsd_embedding_output_dir
      CUDA_VISIBLE_DEVICES=$gpu python local/decode_VSD.py \
          --model_path $vsd_model_path \
          --prob_dir $vsd_prob_dir \
          --embedding_output_dir $vsd_embedding_output_dir \
          --lip_train_scp $lip_train_scp \
          --rttm_train $rttm_train \
          --lip_decode_scp $lip_decode_scp

      system=vsd

      local/prob_to_rttm.sh --system $system\
          --set $set \
          --prob_dir $vsd_prob_dir \
          --oracle_vad $oracle_vad \
          --rttm_dir $vsd_output_rttm_dir \
          --oracle_rttm $oracle_rttm --fps 25
    
    for i in $( seq 0 5)
    do 
      ch=$i
      echo ch$ch
      data=MISP2022_${set}_Far${ch}_WPE 
    
      
      mkdir -p data/$data
      find $wav_dir | grep -E "_${ch}\.wav" > data/$data/wav.list
      awk -F "/" '{print $NF}' data/$data/wav.list | \
            awk -F "_" '{print $1"_"$2"_"$3"_"$4}' > data/$data/utt
      paste -d " " data/$data/utt data/$data/wav.list > data/$data/wav.scp
      paste -d " " data/$data/utt data/$data/utt > data/$data/utt2spk
      utils/fix_data_dir.sh data/$data
      local/extract_feature.sh --stage 1 --ivector_dir $ivector_dir \
          --data $data --rttm $vsd_output_rttm_dir/rttm_th0.55_pp_oraclevad \
          --max_speaker 6 --affix _VSD

      avsd_prob_dir=exp/result_avsd/$data/prob
      avsd_output_rttm_dir=exp/result_avsd/$data/rttm

      CUDA_VISIBLE_DEVICES=0,1 python local/decode_AVSD.py \
          --vsd_model_path $vsd_model_path \
          --avsd_model_path $avsd_model_path \
          --lip_train_scp $lip_train_scp \
          --rttm_train $rttm_train \
          --lip_decode_scp $lip_decode_scp \
          --ivector_train $ivector_train \
          --ivector_decode $ivector_dir/ivectors_${data}_VSD/ivectors_spk.txt \
          --fbank_dir fbank/HTK/${data}_cmn_slide \
          --prob_dir $avsd_prob_dir
          
      system=avsd
      
      local/prob_to_rttm.sh --system $system\
          --set $set \
          --prob_dir $avsd_prob_dir/av_sd \
          --oracle_vad $oracle_vad \
          --session2spk $avsd_prob_dir/session2speaker \
          --rttm_dir $avsd_output_rttm_dir \
          --ch ch$ch \
          --oracle_rttm $oracle_rttm --fps 100
    done
fi

if [ $stage -le 7 ]; then  # Utilize dover_lap to fuse multi-channel results
    data=MISP2022_${set}_Far_WPE
    mkdir -p exp/result_avsd/${data}-Fusion
    dover-lap exp/result_avsd/${data}-Fusion/fusion exp/result_avsd/MISP2022_${set}_Far*_WPE/rttm/rttm_th0.65_pp_oraclevad
    system=avsd
    i=fusion
    rttm_dir=exp/result_avsd/${data}-Fusion
    local/analysis_diarization.sh $system $i $set $oracle_rttm $rttm_dir/fusion | grep ALL
fi
