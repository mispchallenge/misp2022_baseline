#!/bin/bash
set -e

# Begin configuration section
agi=
python_path=/home/cv1/hangchen2/anaconda3/envs/py37/bin/
# End configuration section

echo "$0 $@"  # Print the command line for logging

. ./utils/parse_options.sh

. ./cmd.sh
. ./path.sh
source ./bashrc

if [ $# != 1 -a $# != 2 ]; then
  echo "Usage: $0 [options] <step> (<gpu_num>)"
  echo "e.g.: $0 0"
  echo "Options: "
  echo "--agi <available gpu idxes>"
  exit
fi

step=$1
if [ $agi ]; then
  gpu_num=`echo ${agi//,/} | wc -L`
  cuda_opt=""
else
  gpu_num=$2
  agi=`seq -s , 0 $(($gpu_num-1))`
fi
train_list=(1_3)
predict_list=()

if [ $step == 0 ]; then
  echo "Welcome to experiment, good luck!"
elif [[ "${train_list[@]}" =~ "${step}" ]]; then
  CUDA_VISIBLE_DEVICES=$agi \
  ${python_path}python -m torch.distributed.launch --nproc_per_node=$gpu_num --master_port=12346 \
    local/run_gpu.py -c $step -m train -rs 123456 -be 0 -es 100 -sp 1 -pf 500 -ss train dev \
    -ms ce acc -si 2 -ci 1 -co max -pd dev -um -1
elif [[ "${predict_list[@]}" =~ "${step}" ]]; then
  CUDA_VISIBLE_DEVICES=$agi \
  ${python_path}python -m torch.distributed.launch --nproc_per_node=$gpu_num --master_port=12346 \
    local/run_gpu.py -c $step -m predict -rs 123456 -be 0 -es 100 -sp 1 -pf 500 -ss train dev \
    -ms ce acc -si 2 -ci 1 -co max -pd dev -um -1
fi

