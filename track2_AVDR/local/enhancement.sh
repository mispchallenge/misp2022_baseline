#!/usr/bin/env bash
# Copyright 2018 USTC (Authors: Zhaoxu Nian, Hang Chen)
# Apache 2.0

# use nara-wpe and BeamformIt to enhance multichannel data

set -eo pipefail
# configs
stage=1
nj=1
python_path=
beamformit_path=

. ./path.sh || exit 1
. ./utils/parse_options.sh || exit 1
if [ $# != 2 ]; then
 echo "Usage: $0 <corpus-data-dir> <enhancement-dir>"
 echo " $0 /path/misp2021 /path/wpe_output"
 exit 1;
fi

data_root=$1
out_root=$2

echo "start speech enhancement"
# wpe
if [ $stage -le 0 ]; then
  echo "start wpe"
  ${python_path}python local/find_wav.py -nj $nj $data_root $out_root/log wpe Far
  for n in `seq $nj`; do
    cat <<-EOF > $out_root/log/wpe.$n.sh
    ${python_path}python local/run_wpe.py $out_root/log/wpe.$n.scp $data_root $out_root
EOF
  done
  chmod a+x $out_root/log/wpe.*.sh
  $train_cmd JOB=1:$nj $out_root/log/wpe.JOB.log $out_root/log/wpe.JOB.sh
  echo "finish wpe"
fi

# BeamformIt
if [ $stage -le 1 ]; then
  echo "start beamformit"
  ${python_path}python local/find_wav.py $out_root $out_root/log beamformit Far
  sed -i 's|${out_root}/||g' $out_root/log/beamformit.1.scp
  ${python_path}python local/run_beamformit.py $beamformit_path/BeamformIt conf/all_conf.cfg / $out_root/log/beamformit.1.scp
  echo "end beamformit"
fi
echo "end speech enhancement"
