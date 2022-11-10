#!/usr/bin/env bash
system=
set=
prob_dir=
oracle_vad=
rttm_dir=
oracle_rttm=
fps=
session2spk=None
ch=
set -e

. ./utils/parse_options.sh

mkdir -p $rttm_dir
for i in `seq 0.40 0.05 0.70`;do
    echo th$i
    python local/thresholding.py --threshold $i \
        --prob_array_dir $prob_dir --session2spk $session2spk \
        --rttm_dir ${rttm_dir}  --min_segments 0 \
        --min_dur 0.00 --segment_padding 0.0 --max_dur 0.301 --fps $fps

    python local/rttm_filter_with_vad.py \
        --input_rttm $rttm_dir/rttm_th${i} \
        --output_rttm $rttm_dir/rttm_th${i}_pp_oraclevad \
        --oracle_vad ${oracle_vad}
    
    if [ -f $oracle_rttm ];then
        echo -n "without oracle vad: "
        local/analysis_diarization.sh $system $i $set $oracle_rttm $rttm_dir/rttm_th${i} $ch | grep ALL
        echo -n "with oracle vad:    "
        local/analysis_diarization.sh $system $i $set $oracle_rttm $rttm_dir/rttm_th${i}_pp_oraclevad $ch | grep ALL
    fi
done
