#!/usr/bin/env bash

stage=0
gpu=0
model_path=
output_dir=
embedding_output_dir=
file_train_scp=scp_dir/MISP_Far/train.lip.scp
rttm_train=scp_dir/MISP_Far/train.rttm
file_decode_scp=

set -e

. ./utils/parse_options.sh

. ./cmd.sh
. ./path.sh


if [ $stage -le 1 ]; then
    CUDA_VISIBLE_DEVICES=$gpu python local/decode_Visual_VAD_Conformer.py \
        --model_path $model_path \
        --output_dir $output_dir \
        --embedding_output_dir $embedding_output_dir \
        --file_train_scp $file_train_scp \
        --rttm_train $rttm_train \
        --file_decode_scp $file_decode_scp
fi