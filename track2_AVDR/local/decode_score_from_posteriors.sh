#!/bin/bash

# Begin configuration section.
python_path=/home/cv1/hangchen2/anaconda3/envs/py37/bin/
# hmm_dir=exp/tri3_far_audio
# data_dir=data/eval_far_video
exp_root=exp
predict_data=eval
predict_item=posterior
used_model=-1
ali_count=
nj=4
cmd=run.pl
stage=0
clean_all=false
out_dir=
scoring_opts=
# End configuration section.

echo "$0 $@" # Print the command line for logging

[ -f path.sh ] && . ./path.sh
set -e
. ./utils/parse_options.sh || exit 1

if [ $# -ne 3 ]; then
  echo "Usage: $0 [options] <exp_id> <hmm-dir> <data-dir> "
  echo "e.g.: $0 6_4 exp_tcd_timit/tri3_2500 kaldi_data_test_snr_inf"
  exit 1
fi

echo ===========================Start=================================
exp_id=`ls $exp_root | grep ^$1`
exp_dir=$exp_root/$exp_id
if [ $used_model -le -1 ]; then
  predict_dir=$exp_dir/predict_best_"$predict_data"
else
  predict_dir=$exp_dir/predict_epoch"$used_model"_"$predict_data"
fi
echo "Read posteriors from $predict_dir"

hmm_dir=$2
echo "Use final.mdl and graph in $hmm_dir"

if [ ! $ali_count ]; then
  input_type=log_posteriors
  use_ali_count=" --ali_count none"
else
  input_type=log_likelihoods
  use_ali_count=" --ali_count $ali_count"
  echo "Divide the posteriors by the prior probabilities, which is from $ali_count"
fi

data_dir=$3
echo "Use text in $data_dir"

flatted_exp_id=`echo $hmm_dir | tr '/' '_'`
[ ! $out_dir ] && out_dir=$predict_dir/result_after_"$flatted_exp_id"_decode_"$input_type"
echo "Output files to $out_dir"

echo =================================================================
if [ $stage -le 0 ]; then
  echo "Generate $nj $input_type arks in $predict_dir/ark_to_decode_$input_type..."
  ${python_path}python local/posterior_pt2ark.py -r $exp_root -c $exp_id -pd $predict_data -pi $predict_item -um $used_model\
    $use_ali_count --num_jobs $nj
  echo "Generation part done"
else
  echo "Skip generation part"
fi

echo =================================================================
if [ $stage -le 1 ]; then
  echo "Decode $nj arks to lat gz files in $out_dir/decode"
  mkdir -p $out_dir/decode
  $cmd JOB=1:$nj $out_dir/decode/result_decode.JOB.txt \
    latgen-faster-mapped --min-active=200 --max-active=7000 --max-mem=100000000 --beam=13.0 --lattice-beam=6.0 \
      --acoustic-scale=0.083333 --allow-partial=true --word-symbol-table=$hmm_dir/graph/words.txt $hmm_dir/final.mdl \
      $hmm_dir/graph/HCLG.fst "ark,s,cs: cat $predict_dir/ark_to_decode_$input_type/$input_type."JOB".ark |" \
      "ark:|gzip -c > $out_dir/decode/lat."JOB.gz || exit 1
  for i in $(seq 1 "$nj"); do grep -vE "LOG|latgen|#" $out_dir/decode/result_decode.$i.txt; done \
    > $out_dir/result_decode.txt
  echo "Decode part done"
else
  echo "Skip decode part"
fi

echo =================================================================
if [ $stage -le 2 ]; then
  echo "Score $nj lat"
  echo $nj > $out_dir/decode/num_jobs
  cp $hmm_dir/final.mdl $out_dir
  local/score.sh $scoring_opts --cmd "$cmd" $data_dir $hmm_dir/graph $out_dir/decode
  echo "Score part done"
else
  echo "Skip score part"
fi

echo =================================================================
if [ $stage -le 3 ]; then
  echo "Organize results"
  ${python_path}python local/sorce_replume_result.py --stage 0 $out_dir/decode/scoring_kaldi/cer_details/per_utt \
    ${predict_dir}/data_with_predicted.json $out_dir/result_cer.json
  echo "Organize part done"
else
  echo "Skip organize part"
fi

# echo =================================================================
# if $clean_all; then
#   echo "Clean extra files"
#   rm -rf $predict_dir/ark_to_decode_$input_type
#   rm -rf $out_dir/decode
#   rm -f $out_dir/final.mdl
#   echo "clean part done"
# else
#   echo "Skip clean part"
# fi
echo ==============================End================================
