#!/usr/bin/env bash

stage=0

. ./utils/parse_options.sh
. ./cmd.sh
. ./path.sh

################################################################################
# Training
################################################################################
misp_dir=                       # data path

find detection_roi/train/far/lip | grep htk > scp_dir/train_far.lip.scp                    # Find the extracted lip ROIs
cat /export/corpus/slwu/rttm/train_combined/*.rttm > scp_dir/train_far_RTTM.rttm           # Connect the rttm files to one file
lip_train_scp=scp_dir/train_far.lip.scp                                                    # A file pointing to the training data lip ROIs
rttm_train=scp_dir/train_far_RTTM.rttm                                                     # The oracle_RTTM file combining all training data sessions

find detection_roi/dev/far/lip | grep htk > scp_dir/${set}_far.lip.scp                  # Find the extracted lip ROIs
cat /export/corpus/misp2022/Released/rttm/dev_new_combined/*.rttm > scp_dir/dev_far_RTTM.rttm  # Connect the dev rttm files to one file
lip_dev_scp=scp_dir/dev_far.lip.scp                                # A file pointing to the lip ROIs
rttm_dev=scp_dir/dev_far_RTTM.rttm                                 # The oracle_RTTM file combining all sessions

ivector_dir=exp/nnet3_cnceleb_ivector  # ivector output path

if [ $stage -le 0 ]; then  # VSD training

    python local/train_VSD.py --project VSD_MISP2022_Far \
                       --file_train_path $lip_train_scp \
                       --rttm_train_path $rttm_train \
                       --file_dev_path $lip_dev_scp \
                       --rttm_dev_path $rttm_dev
fi

# Find best model:
best_model=model/pretrained/conformer_v_sd_2.model

if [ $stage -le 1 ]; then  # Extract train set frame-level:

    python local/extract_visual_embedding.py --model_path $best_model \
                                      --output_dir output/visual_embedding \
                                      --file_train_path $lip_train_scp \
                                      --rttm_train_path $rttm_train
    
    find output/visual_embedding | grep htk > scp_dir/train.lip.embedding.scp

fi

if [ $stage -le 2 ]; then  # Audio and speaker embedding extract

    mkdir -p data/misp_train
    # Prepare data dir
    find $misp_dir/audio/train/far | grep -E "\.wav" | \
            awk -F "/" '{print $NF}' | \
            awk -F "_" '{print $1"_"$2"_"$3"_"$4}' | \
            sort|  uniq > data/misp_train/session.list
    # RAW audio
    rm -f data/misp_train/wav.scp
    for ch in `seq 0 5`;do
        awk -v dir=$misp_dir/audio/train/far -v c=$ch \
            '{print $0"_RAW_ch"c,dir"/"$0"_Far_"c".wav"}' \
            data/misp_train/session.list >> data/misp_train/wav.scp
    done
    audio_type="RAW"

    # If you did wpe or enhancement on misp data, then additional data could be used for training
    #for ch in `seq 0 5`;do
    #    awk -v dir=$misp_dir/wpe/train/far -v c=$ch \
    #        '{print $0"_wpe_ch"c,dir"/"$0"_Far_"c".wav"}' \
    #        data/misp_train/session.list >> data/misp_train/wav.scp
    #done
    #audio_type="$audio_type+WPE"

    
    awk '{print $1,$1}' data/misp_train/wav.scp > data/misp_train/utt2spk
    utils/fix_data_dir.sh data/misp_train

    # Extract audio feature
    local/extract_feature.sh --nj 16 --ivector_dir $ivector_dir --data misp_train \
            --rttm $rttm_train --max_speaker 6 --affix _oracle

fi

if [ $stage -le 3 ]; then  # AVSD training

    # Here we used visual frame-level embedding extracted in VSD training steps
    # We didn't evaluate with dev set when training. If you want to evaluate each saved
    # model, run the following decoding scripts.
    # Default batchsize is 48 on 4 3090 GPUs (memory: 24GB)
    # With 6 channels RAW (without any preprocessing), WPE and Enhanced audio, the 3rd epoch model is the best in previous experiments.

    python local/train_AVSD.py --project AVSD_MISP2022_Far \
                        --train_audio_fea_dir fbank/HTK/misp_train_cmn_slide \
                        --train_speaker_embedding $ivector_dir/ivectors_misp_train/ivectors_spk.txt \
                        --train_video_fea_scp scp_dir/train.lip.embedding.scp \
                        --rttm_train_path $rttm_train \
                        --audio_type $audio_type
                       
fi

################################################################################
# Decoding
################################################################################

gpu="0,1"  # Defining GPU
set=dev  # Defining the set: dev, or eval.
ch=0

find detection_roi/${set}/far/lip | grep htk > scp_dir/${set}_far.lip.scp                  # Find the extracted lip ROIs
cat /export/corpus/misp2022/Released/rttm/${set}/*.rttm > scp_dir/dev_far_RTTM.rttm   # Connect the rttm files to one file
lip_decode_scp=scp_dir/${set}_far.lip.scp                                 # A file pointing to the lip ROIs
oracle_rttm=scp_dir/${set}_far_RTTM.rttm                                # The oracle_RTTM file combining all sessions
oracle_vad=scp_dir/${set}_far_timestamp.lab                           # The oracle_VAD timestamp file combining all sessions
data=MISP2022_${set}_Far${ch}_WPE                                      # Defining the data name
wav_dir=wpe/${set}/                                                                # Defining the path of audio after WPE

vsd_model_path=model/pretrained/conformer_v_sd_2.model   # VSD model
vsd_prob_dir=exp/result_vsd/$set/prob
vsd_embedding_output_dir=exp/result_vsd/$set/vemb
vsd_output_rttm_dir=exp/result_vsd/$set/rttm

ivector_dir=exp/nnet3_cnceleb_ivector  # ivector output path
ivector_train=$ivector_dir/ivectors_misp_train/ivectors_spk.txt  # ivector of training data, this can be replace with the training set ivector you extracted in stage 2

avsd_model_path=model/pretrained/av_diarization_3.model   # AVSD model
avsd_prob_dir=exp/result_avsd/$data/prob
avsd_output_rttm_dir=exp/result_avsd/$data/rttm

################################################################################
# single channel
################################################################################

if [ $stage -le 4 ]; then     # VSD decoder
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

if [ $stage -le 5 ]; then  # get RTTM file and DER
    system=vsd
    local/prob_to_rttm.sh --system $system\
                          --set $set \
                          --prob_dir $vsd_prob_dir \
                          --oracle_vad $oracle_vad \
                          --rttm_dir $vsd_output_rttm_dir \
                          --oracle_rttm $oracle_rttm --fps 25
fi

if [ $stage -le 6 ]; then  # extract ivector
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

if [ $stage -le 7 ]; then  # AVSD decoder
    
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
if [ $stage -le 8 ]; then  # get RTTM file and DER
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

if [ $stage -le 9 ]; then

      #### If you did not execute the previous steps, you need to execute the content in the note
      #mkdir -p $vsd_prob_dir
      #mkdir -p $vsd_embedding_output_dir
      #CUDA_VISIBLE_DEVICES=$gpu python local/decode_VSD.py \
      #   --model_path $vsd_model_path \
      #  --prob_dir $vsd_prob_dir \
      #   --embedding_output_dir $vsd_embedding_output_dir \
      #    --lip_train_scp $lip_train_scp \
      #   --rttm_train $rttm_train \
      #   --lip_decode_scp $lip_decode_scp

      #system=vsd

      #local/prob_to_rttm.sh --system $system\
      #    --set $set \
      #    --prob_dir $vsd_prob_dir \
      #    --oracle_vad $oracle_vad \
      #    --rttm_dir $vsd_output_rttm_dir \
      #    --oracle_rttm $oracle_rttm --fps 25
    
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

if [ $stage -le 10 ]; then  # Utilize dover_lap to fuse multi-channel results
    data=MISP2022_${set}_Far_WPE
    mkdir -p exp/result_avsd/${data}-Fusion
    dover-lap exp/result_avsd/${data}-Fusion/fusion exp/result_avsd/MISP2022_${set}_Far*_WPE/rttm/rttm_th0.65_pp_oraclevad
    system=avsd
    i=fusion
    rttm_dir=exp/result_avsd/${data}-Fusion
    local/analysis_diarization.sh $system $i $set $oracle_rttm $rttm_dir/fusion | grep ALL
fi
