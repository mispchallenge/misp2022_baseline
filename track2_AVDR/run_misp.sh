#!/usr/bin/env bash
#
# This recipe is for misp2022 track2 (AVDR), it recognise
# a given evaluation utterance given ground truth
# diarization information
#
# Copyright  2022  USTC (Author: Hang Chen, Zhe Wang)
# Apache 2.0
#

# Begin configuration section.
nj=15
nnet_stage=0
oovSymbol="<UNK>"
boost_sil=1.0 # note from Dan: I expect 1.0 might be better (equivalent to not
              # having the option)... should test.
numLeavesTri1=7000
numGaussTri1=56000
numLeavesMLLT=10000
numGaussMLLT=80000
numLeavesSAT=12000
numGaussSAT=96000
# End configuration section

. ./utils/parse_options.sh

. ./cmd.sh
. ./path.sh
source ./bashrc

set -e

stage=$1

# path settings
beamformit_path=   # e.g. /path/BeamformIt-master
python_path=       # e.g. /path/python/bin
misp2022_corpus=   # e.g. /path/misp2022
enhancement_dir=${misp2022_corpus}_WPE
dict_dir=data/local/dict
data_roi=data/local/roi

###########################################################################
# Training
###########################################################################

##########################################################################
# wpe+beamformit
##########################################################################

# use nara-wpe and beamformit to enhance multichannel misp data
# notice:make sure you install nara-wpe and beamformit and you need to compile BeamformIt with the kaldi script install_beamformit.sh 
if [ $stage -le -1 ]; then
  for x in dev train ; do
    if [[ ! -f ${enhancement_dir}/audio/$x.done ]]; then
      local/enhancement.sh --stage 0 --python_path $python_path --beamformit_path $beamformit_path \
        $misp2022_corpus/audio/$x ${enhancement_dir}/audio/$x  || exit 1;
      touch ${enhancement_dir}/audio/$x.done
    fi
  done
fi

###########################################################################
# prepare dict
###########################################################################

# download DaCiDian raw resources, convert to Kaldi lexicon format
if [ $stage -le 0 ]; then
  local/prepare_dict.sh --python_path $python_path $dict_dir || exit 1;
fi

###########################################################################
# prepare audio data
###########################################################################

if [ $stage -le 1 ]; then
  for x in dev train; do
    for y in far; do
      ${python_path}python local/prepare_data.py -nj 1 feature/misp2022_avsr/${x}_${y}_audio_wpe/beamformit/wav/'*.wav' \
        released_data/misp2022_avsr/${x}_near_transcription/TextGrid/"*.TextGrid" data/${x}_${y}_audio|| exit 1;
      # spk2utt
      utils/utt2spk_to_spk2utt.pl data/${x}_${y}_audio/utt2spk | sort -k 1 | uniq > data/${x}_${y}_audio/spk2utt
      echo "word segmentation"
      ${python_path}python local/word_segmentation.py $dict_dir/word_seg_vocab.txt data/${x}_${y}_audio/text_sentence > data/${x}_${y}_audio/text
    done
  done
fi

###########################################################################
# prepare language module
###########################################################################

# L
if [ $stage -le 2 ]; then
  utils/prepare_lang.sh --position-dependent-phones false \
    $dict_dir "$oovSymbol" data/local/lang data/lang  || exit 1;
fi

# arpa LM
if [ $stage -le 3 ]; then
  local/train_lms_srilm.sh --train-text data/train_far_audio/text --dev-text data/dev_far_audio/text --oov-symbol "$oovSymbol" data/ data/srilm
fi

# prepare lang_test
if [ $stage -le 4 ]; then
  utils/format_lm.sh data/lang data/srilm/lm.gz data/local/dict/lexicon.txt data/lang_test
fi

mkdir -p exp

###########################################################################
# feature extraction
###########################################################################
if [ $stage -le 5 ]; then
  mfccdir=mfcc
  for x in dev train; do
    for y in far; do
      utils/fix_data_dir.sh data/${x}_${y}_audio
      steps/make_mfcc_pitch.sh --cmd "$train_cmd" --nj $nj data/${x}_${y}_audio feature/misp2022_avsr/${x}_${y}_mfcc_pitch_kaldi/log \
        feature/misp2022_avsr/${x}_${y}_mfcc_pitch_kaldi/ark
      steps/compute_cmvn_stats.sh data/${x}_${y}_audio feature/misp2022_avsr/${x}_${y}_mfcc_pitch_kaldi/log \
        feature/misp2022_avsr/${x}_${y}_mfcc_pitch_kaldi/ark
      utils/fix_data_dir.sh data/${x}_${y}_audio
    done
  done
  # subset the training data for fast startup
  # for x in 50 100; do
  #   utils/subset_data_dir.sh data/train_far ${x}000 data/train_far_${x}k
  # done
fi

###########################################################################
# mono phone train
###########################################################################
if [ $stage -le 6 ]; then
  for x in far; do
    steps/train_mono.sh --boost-silence $boost_sil --nj $nj --cmd "$train_cmd" data/train_${x}_audio data/lang exp/mono_${x}_audio || exit 1;
    # make graph
    utils/mkgraph.sh data/lang_test exp/mono_${x}_audio exp/mono_${x}_audio/graph || exit 1;
  done
fi

###########################################################################
# tr1 delta+delta-delta
###########################################################################
if [ $stage -le 7 ]; then
  for x in far; do
    # alignment
    steps/align_si.sh --boost-silence $boost_sil --cmd "$train_cmd" --nj $nj data/train_${x}_audio data/lang exp/mono_${x}_audio exp/mono_${x}_audio_ali || exit 1;
    # training
    steps/train_deltas.sh --boost-silence $boost_sil --cmd "$train_cmd" $numLeavesTri1 $numGaussTri1 data/train_${x}_audio data/lang exp/mono_${x}_audio_ali exp/tri1_${x}_audio || exit 1;
    # make graph
    utils/mkgraph.sh data/lang_test exp/tri1_${x}_audio exp/tri1_${x}_audio/graph || exit 1;
  done
fi

###########################################################################
# tri2 all lda+mllt
###########################################################################
if [ $stage -le 8 ]; then
  for x in far; do
    # alignment
    steps/align_si.sh --boost-silence $boost_sil --cmd "$train_cmd" --nj $nj data/train_${x}_audio data/lang exp/tri1_${x}_audio exp/tri1_${x}_audio_ali || exit 1;
    # training
    steps/train_lda_mllt.sh --boost-silence $boost_sil --cmd "$train_cmd" $numLeavesMLLT $numGaussMLLT data/train_${x}_audio data/lang exp/tri1_${x}_audio_ali exp/tri2_${x}_audio || exit 1;
    # make graph
    utils/mkgraph.sh data/lang_test exp/tri2_${x}_audio exp/tri2_${x}_audio/graph || exit 1;
  done
fi

###########################################################################
# tri3 all sat
###########################################################################
if [ $stage -le 9 ]; then
  for x in far; do
    # alignment
    steps/align_fmllr.sh --boost-silence $boost_sil --cmd "$train_cmd" --nj $nj data/train_${x}_audio data/lang \
      exp/tri2_${x}_audio exp/tri2_${x}_audio_ali || exit 1;
    # training
    steps/train_sat.sh --boost-silence $boost_sil --cmd "$train_cmd" $numLeavesSAT $numGaussSAT data/train_${x}_audio data/lang \
      exp/tri2_${x}_audio_ali exp/tri3_${x}_audio || exit 1;
    # make graph
    utils/mkgraph.sh data/lang_test exp/tri3_${x}_audio exp/tri3_${x}_audio/graph || exit 1;
  done

  # alignment 
  for x in dev train ; do
    for y in far ; do
      ali_dir=exp/tri3_${y}_audio_ali_${x}_${y}_audio
      output_dir=feature/misp2022_avsr/${x}_${y}_tri3_ali/pt
      steps/align_fmllr.sh --boost-silence $boost_sil --cmd "$train_cmd" --nj $nj data/${x}_${y}_audio data/lang \
        exp/tri3_${y}_audio $ali_dir || exit 1;
      mkdir -p ${output_dir}
      tool/format_ali.sh --python_path ${python_path} --nj ${nj} --cmd ${train_cmd} --frame_dur 0.02 --frame_shift 0.01 ${ali_dir} ${output_dir} || exit 1;
      echo "============================================================"
      echo "Create_json_file"
      ${python_path}python local/generate_key2shape.py $output_dir/'*.pt'
    done
  done

fi

###########################################################################
# segment wav to pt
###########################################################################
if [ $stage -le 10 ]; then
  for x in dev train ; do
    for y in far ; do
      data_dir=data/${x}_${y}_audio
      store_dir=feature/misp2022_avsr/${x}_${y}_audio_segment/pt
      echo "============================================================"
      echo "segment $data_dir, store in $store_dir"
      ${python_path}python tool/segment_wav_to_pt.py -nj $nj $data_dir $store_dir
      cat $store_dir/segment.log
      echo "============================================================"
      echo "Create_json_file"
      ${python_path}python local/generate_key2shape.py $store_dir/'*.pt'
    done
  done
fi

###########################################################################
# prepare video data
###########################################################################
if [ $stage -le 11 ]; then
  for x in dev train; do
    for y in far; do
      ${python_path}python local/prepare_data.py -nj 1 released_data/misp2022_avsr/${x}_${y}_video/mp4/'*.mp4' \
        released_data/misp2022_avsr/${x}_near_transcription/TextGrid/"*.TextGrid" data/${x}_${y}_video|| exit 1;
      # spk2utt
      utils/utt2spk_to_spk2utt.pl data/${x}_${y}_video/utt2spk | sort -k 1 | uniq > data/${x}_${y}_video/spk2utt
      echo "word segmentation"
      ${python_path}python local/word_segmentation.py $dict_dir/word_seg_vocab.txt data/${x}_${y}_video/text_sentence > data/${x}_${y}_video/text
    done
  done
fi

###########################################################################
# segment video roi to pt
###########################################################################
if [ $stage -le 12 ]; then
  for x in dev train; do
    for y in far ; do
      data_dir=data/${x}_${y}_video
      roi_json_dir=released_data/misp2022_avsr/${x}_${y}_detection_result
      store_dir=feature/misp2022_avsr/${x}_${y}_video_lip_segment/pt
      echo "============================================================"
      echo "segment $data_dir, store in $store_dir"
      ${python_path}python local/segment_video_roi.py --roi_type lip --roi_size 96 96 --need_speaker --roi_sum $data_dir $roi_json_dir $store_dir || exit 1
      cat $store_dir/segment.log
      echo "============================================================"
      echo "Create_json_file"
      ${python_path}python local/generate_key2shape.py $store_dir/'*.pt'
    done
  done
fi

###########################################################################
# prepare json file for DNN training
###########################################################################
if [ $stage -le 13 ]; then
  ${python_path}python local/index_file2json.py -s 0
fi

if [ $stage -le 14 ]; then
  CUDA_VISIBLE_DEVICES=3 ${python_path}python local/feature_cmvn.py 0
fi

if [ $stage -le 15 ]; then
  num_targets=$(tree-info exp/tri3_far_audio/tree |grep num-pdfs|awk '{print $2}')
  echo $num_targets
fi

###########################################################################
# DNN training
###########################################################################

if [ $stage -le 16 ]; then
  # you can add your submission script and the training script is as follows:
  # bash submit.sh --pn 1-3 --pd avsr_far --numg 4 --gputype TeslaV100-PCIE-12GB --logfile submit_log/1_3.log "run_gpu.sh 1_3 4"
  agi=0,1,2,3
  gpu_num=`echo ${agi//,/} | wc -L`
  CUDA_VISIBLE_DEVICES=$agi \
  ${python_path}python -m torch.distributed.launch --nproc_per_node=$gpu_num --master_port=12346 \
    local/run_gpu.py -c $step -m train -rs 123456 -be 0 -es 100 -sp 1 -pf 500 -ss train dev \
    -ms ce acc -si 2 -ci 1 -co max -pd dev -um -1
fi

###########################################################################
# Inference[using audio-visual speaker diarization results]
###########################################################################

###########################################################################
# prepare audio and video data
# use the RTTM file to generate the utt2spk and segments files
# Notice: you need to modify the path of the RTTM file!
###########################################################################

if [ $stage -le 17 ]; then
  for x in dev; do
    for y in far; do
      # The path of the RTTM file
      rttm_file_path=data/${x}_${y}.rttm
      store_audio_dir=data/${x}_${y}_audio_inference
      store_video_dir=data/${x}_${y}_video_inference
      mkdir $store_audio_dir
      mkdir $store_video_dir
      cp data/${x}_${y}_audio/wav.scp data/${x}_${y}_audio_inference/wav.scp
      cp data/${x}_${y}_video/wav.scp data/${x}_${y}_video_inference/wav.scp
      ${python_path}python tool/rttm_to_utt2spk_segments.py $rttm_file_path $store_audio_dir/utt2spk $store_audio_dir/segments
      cp $store_audio_dir/utt2spk $store_video_dir/utt2spk
      cp $store_audio_dir/segments $store_video_dir/segments
    done
  done
fi

###########################################################################
# segment audio and video data
###########################################################################

if [ $stage -le 18 ]; then
  for x in dev ; do
    for y in far ; do
      data_dir=data/${x}_${y}_audio_inference
      store_dir=feature/misp2022_avsr/${x}_${y}_audio_inference_segment/pt
      echo "============================================================"
      echo "segment $data_dir, store in $store_dir"
      ${python_path}python tool/segment_wav_to_pt.py -nj $nj $data_dir $store_dir
      cat $store_dir/segment.log
      echo "============================================================"
      echo "Create_json_file"
      ${python_path}python local/generate_key2shape.py $store_dir/'*.pt'
    done
  done
fi

if [ $stage -le 19 ]; then
  for x in dev; do
    for y in far ; do
      data_dir=data/${x}_${y}_video_inference
      roi_json_dir=released_data/misp2022_avsr/${x}_${y}_detection_result
      store_dir=feature/misp2022_avsr/${x}_${y}_video_inference_lip_segment/pt
      echo "============================================================"
      echo "segment $data_dir, store in $store_dir"
      ${python_path}python local/segment_video_roi.py --roi_type lip --roi_size 96 96 --need_speaker --roi_sum $data_dir $roi_json_dir $store_dir || exit 1
      cat $store_dir/segment.log
      echo "============================================================"
      echo "Create_json_file"
      ${python_path}python local/generate_key2shape.py $store_dir/'*.pt'
    done
  done
fi

###########################################################################
# prepare json file for DNN prediction
###########################################################################
if [ $stage -le 20 ]; then
  ${python_path}python local/index_file2json.py -s 1
fi

###########################################################################
# DNN prediction
###########################################################################

if [ $stage -le 21 ]; then
  agi=0,1,2,3
  gpu_num=`echo ${agi//,/} | wc -L`
  CUDA_VISIBLE_DEVICES=$agi \
  ${python_path}python -m torch.distributed.launch --nproc_per_node=$gpu_num --master_port=12345 \
  local/run_gpu.py -c 1_3 -m predict -rs 123456 -be 0 -es 100 -sp 1 -pf 500 -ss train dev \
  -ms ce acc -si 2 -ci 1 -co max -pd dev_inference -um -1
fi

###########################################################################
# decode
###########################################################################

if [ $stage -le 22 ]; then
  for x in 1_3; do
    for y in dev_inference; do
      local/decode_score_from_posteriors.sh --python_path $python_path --exp_root exp --predict_data ${y} --predict_item posteriori --used_model -1 \
        --ali_count feature/misp2022_avsr/train_far_tri3_ali/ali_train_pdf.counts --nj 16 --cmd "$decode_cmd" --stage 0 \
        ${x} exp/tri3_far_audio data/${y}_far_audio_inference
      # python local/sorce_replume_result.py --stage 1 s s result_cer.json
    done
  done
fi

###########################################################################
# Caculate CER(Global Speaker ID)
# Notice: copy the result_decode.txt file! 
###########################################################################
if [ $stage -le 23 ]; then
  for x in dev ; do
    for y in far ; do
      # copy exp/1_3_*/predict_best_dev_inference/result_*/result_decode.txt to data/dev_far_audio/result_decode.txt
      result_decode_file=data/${x}_${y}_audio/result_decode.txt
      text_file=data/${x}_${y}_audio/text
      ${python_path}python tool/cer.py -s $result_decode_file -r $text_file
    done
  done
fi

###########################################################################
# Caculate cpCER(Local Speaker ID)
# Notice: The directory contains files for each session!
###########################################################################
# if [ $stage -le 24 ]; then
#   for x in dev ; do
#     for y in far ; do
#       result_decode_directory=data/${x}_${y}_audio/Result
#       text_directory=data/${x}_${y}_audio/Ground_Truth
#       score_file=data/${x}_${y}_audio/score.txt
#       ${python_path}python tool/cpcer.py -s $result_decode_directory -r $text_directory -f $score_file
#     done
#   done
# fi
