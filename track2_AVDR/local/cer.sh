#!/usr/bin/env bash



ref=/yrfs2/cv1/hangchen2/code/misp2021_dili/exp/chain_train_far/tdnn1b_sp/decode_eval_far/scoring_kaldi/test_filt.txt
hyp=/yrfs2/cv1/hangchen2/code/misp2021_dili/exp/chain_train_far/tdnn1b_sp/decode_eval_far/scoring_kaldi/penalty_0.0/7.txt


source ./bashrc
set -e
. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

cat $ref | compute-wer --text --mode=present ark:$hyp  ark,p:- | grep WER  || exit 1;
