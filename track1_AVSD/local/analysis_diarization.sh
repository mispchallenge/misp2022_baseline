#!/usr/bin/env bash
#

score_area=
collar=0
. ./utils/parse_options.sh
system=$1
i=$2
set=$3
ref_rttm_path=$4
hyp_rttm_path=$5
ch=$6
#tempdir=$( mktemp  -d  /tmp/eval_diarization.XXXXXX )

tempdir=exp/result_$system/DER_result_$set/$ch/tmp-${i}/
mkdir -p $tempdir

local/md-eval-22.pl $score_area -c $collar -afc -r $ref_rttm_path -s $hyp_rttm_path > ${tempdir}/temp.info
grep SCORED ${tempdir}/temp.info | cut -d "=" -f 2 | cut -d " " -f 1 > ${tempdir}/SCORED.list
grep MISSED ${tempdir}/temp.info | cut -d "=" -f 2 | cut -d " " -f 1 > ${tempdir}/MISSED.list
grep FALARM ${tempdir}/temp.info | cut -d "=" -f 2 | cut -d " " -f 1 > ${tempdir}/FALARM.list
grep "SPEAKER ERROR" ${tempdir}/temp.info | cut -d "=" -f 2 | cut -d " " -f 1 > ${tempdir}/SPEAKER.list
grep OVERALL ${tempdir}/temp.info | cut -d "=" -f 4 | cut -d ")" -f 1 > ${tempdir}/session.list
sed -i '$d' ${tempdir}/session.list
echo "ALL" >> ${tempdir}/session.list
for l in `cat ${tempdir}/session.list`;do
    grep $l $ref_rttm_path | awk '{print $8}' | sort | uniq | wc -l
done > ${tempdir}/oracle_spknum.list

for l in `cat ${tempdir}/session.list`;do
    grep $l $hyp_rttm_path | awk '{print $8}' | sort | uniq | wc -l
done > ${tempdir}/diarized_spknum.list

paste -d " " ${tempdir}/session.list ${tempdir}/SCORED.list ${tempdir}/MISSED.list \
             ${tempdir}/FALARM.list ${tempdir}/SPEAKER.list ${tempdir}/oracle_spknum.list \
             ${tempdir}/diarized_spknum.list > ${tempdir}/temp.details

awk '{printf "%s %.2f %.2f %.2f %.2f %d %d\n",$1,$4/$2*100,$3/$2*100,$5/$2*100,($3+$4+$5)/$2*100,$6,$7}' ${tempdir}/temp.details > ${tempdir}/temp.info1
echo "session FA MISS SPKERR DER ORACLE_SPKNUM DIARIZED_SPKNUM" > ${tempdir}/temp.details
grep -v "ALL" ${tempdir}/temp.info1 | sort -n -k 5 >> ${tempdir}/temp.details
grep "ALL" ${tempdir}/temp.info1 >> ${tempdir}/temp.details

column -t ${tempdir}/temp.details

#rm -rf ${tempdir}