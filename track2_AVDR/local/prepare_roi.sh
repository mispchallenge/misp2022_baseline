#!/usr/bin/env bash
# Copyright 2021 USTC (Authors: Hang Chen)
# Apache 2.0

# extract region of interest (roi) in the video, store as npz file, item name is "data"

set -e
# configs for 'chain'
python_path=
nj=15
roi_type=head
roi_size="96 96"
need_speaker=true
roi_sum=true
local=false
# End configuration section.
. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh


if [ $# != 3 ]; then
  echo "Usage: $0 <data-set> <roi-json-dir> <roi-store-dir>"
  echo " $0 data/train_far /path/roi data/train_far_sp_hires"
  exit 1;
fi

echo "$0 $@"  # Print the command line for logging

data_dir=$1
roi_json_dir=$2
roi_store_dir=$3
optional="--roi_type $roi_type --roi_size $roi_size"
if $need_speaker; then
  optional="$optional --need_speaker"
fi
if $roi_sum; then
  optional="$optional --roi_sum"
fi
mkdir -p $roi_store_dir
###########################################################################
# segment mp4 and crop roi, store as pt
###########################################################################
mkdir -p $roi_store_dir/log
for n in `seq $nj`; do
  cat <<-EOF > $roi_store_dir/log/roi.$n.sh
${python_path}python local/segment_video_roi.py $optional -ji $((n-1)) -nj $nj $data_dir $roi_json_dir $roi_store_dir
EOF
done
chmod a+x $roi_store_dir/log/roi.*.sh

if $local; then
  $train_cmd JOB=1:$nj $roi_store_dir/log/roi.JOB.log $roi_store_dir/log/roi.JOB.sh || exit 1;
else
  for n in `seq $nj`; do
    submit.sh --pn roi$n --pd roi$n --numg 1 --gputype TeslaM40 --logfile $roi_store_dir/log/roi.$n.log "$roi_store_dir/log/roi.$n.sh"
  done
fi
