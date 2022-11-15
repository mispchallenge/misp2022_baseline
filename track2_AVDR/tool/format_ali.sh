#!/usr/bin/env bash
# Copyright 2022 USTC (Authors: Hang Chen)
# Apache 2.0

# transform ali.gz to pt
python_path=
nj=
cmd=
frame_dur=0.02
frame_shift=0.01
echo "$0 $@"
. utils/parse_options.sh
. ./path.sh || exit 1;


if [ $# != 2 ]; then
  echo "Usage: $0 <ali-dir> <output-dir>"
  echo " $0 /path/misp /path/misp_WPE"
  exit 1;
fi
ali_dir=$1
output_dir=$2

echo "ali to pdf"
$cmd JOB=1:$nj $ali_dir/log/ali2pdf.JOB.log \
  ali-to-pdf $ali_dir/final.mdl "ark:gunzip -c $ali_dir/ali.JOB.gz|" \
  "ark,scp:$ali_dir/pdf.JOB.ark,$ali_dir/pdf.JOB.scp"
for n in $(seq $nj); do
  cat $ali_dir/pdf.$n.scp || exit 1;
done > $ali_dir/pdf.scp || exit 1;

echo "ark to pt"
${python_path}python tool/pdf_ark2pt.py -nj $nj $ali_dir/pdf.scp $frame_dur $frame_shift $output_dir

echo "analysis alignment"
num_pdf=$(hmm-info $ali_dir/final.mdl | awk '/pdfs/{print $4}')
echo $num_pdf > $output_dir/../num_pdf
labels_tr_pdf="ark:ali-to-pdf $ali_dir/final.mdl \"ark:gunzip -c $ali_dir/ali.*.gz |\" ark:- |"
analyze-counts --verbose=1 --binary=false --counts-dim=$num_pdf "$labels_tr_pdf" $output_dir/../ali_train_pdf.counts

echo "success format ali from $ali_dir to $output_dir"
