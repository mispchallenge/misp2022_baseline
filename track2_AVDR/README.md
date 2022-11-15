# MISP2022 Challenge Track 2: Audio-Visual Diarization and Recognition (Baseline)
![Baseline_System](https://user-images.githubusercontent.com/88126124/201823221-47cad24f-a2a0-401a-a814-766ab927f88d.png)
Fig.1. The illustration of network structure

Fig. 1 shows the baseline AVDR system, which consists of an AVSD module followed by an AVSR module. The AVSD module also serves as the baseline system for Track 1. We elaborate the architecture and training process of the AVSD and AVSR modules, and provide the details about joining the AVSD and AVSR modules for decoding.

## AVSD Module Training
The training of AVSD module is consistent with the track1.

## AVSR Module Training
The training of AVSR module uses the oracle diarization results.
- **Data preparation**

  - **speech enhancement**

  Weighted Prediction Error(WPE) dereverberation and BeamformIt are used to reduce the reverberations of speech signals. The algorithms are implemented with open-source toolkit, [nara_wpe](https://github.com/fgnt/nara_wpe)[BeamformIt](https://github.com/xanguera/BeamformIt)

  - **prepare data and language directory for kaldi**

  For training, development, and test sets, we prepare data directories and the lexicon in the format expected by  [kaldi](http://kaldi-asr.org/doc/data_prep.html) respectively. Note that we choose [DaCiDian](https://github.com/aishell-foundation/DaCiDian.git) raw resource and convert it to kaldi lexicon format.

- **Language model**

  We segment MISP speech transcription for language model training by applying [DaCiDian](https://github.com/aishell-foundation/DaCiDian.git) as dict and [Jieba](https://github.com/fxsjy/jieba) open-source toolkit. For the language model, we choose a maximum entropy-based 3-gram model, which achieves the best perplexity, from n-gram(n=2,3,4) models trained on MISP speech transcripts with different smoothing algorithms and parameters sets. And the selected 3-gram model has 516600 unigrams, 432247 bigrams, and 915962 trigrams respectively.  Note that the temporary and final language models are stored in /data/srilm.

- **Acoustic model**

  The acoustic model of the ASR system is built largely following the Kaldi recipes which mainly contain two stages: GMM-HMM state model and DNN deep learning model.

  - **GMM-HMM**

    For features extraction, we extract 13-dimensional MFCC features plus 3-dimensional pitches. As a start point for triphone models, a monophone model is trained.  Then a triphone model is trained using delta features on the whole dataset. In the third triphone model training process, an MLLT-based global transform is estimated iteratively on the top of LDA feature to extract independent speaker features. For the fourth triphone model, feature space maximum likelihood linear regression (fMLLR) with speaker adaptive training (SAT) is applied in the training.

  - **DNN-HMM**

    Based on the tied-triphone state alignments from GMM, DNN is configured and trained to replace GMM. The input features are 40-dimensional FBank features with cepstral normalization and the 96 × 96 (*W × H*) lip ROI.

## Inference
The RTTM file as the output of the AVSD module contains the information of the Session, SPK<sup>−4</sup>, Tstart, and Tdur.

where:

Sessionk: k-th session
SPKi: i-th speaker
Tjstart: the start time of the j-th utterance for SPKi
Tjdur: the duration of the j-th utterance for SPKi
During the inference, the RTTM file is used for segmenting audio and video data in AVSR module.



## Results

| Models        | cpCER of Dev |
| ------------- | ------------ |
| ASD+ASR       | 80.44        |
| VSD+ASR       | 71.13        |
| VSD+AVSR      | 66.79        |
| **AVSD+AVSR**     | 66.07        |

The AVSD+AVSR model is our AVDR baseline system.


## Quick start

- **Setting Local System Jobs**

```
# Setting local system jobs (local CPU - no external clusters)
export train_cmd=run.pl
export decode_cmd=run.pl
```

- **Setting  Paths**

```
--- path.sh ---
# Defining Kaldi root directory
export KALDI_ROOT=
# Setting paths to useful tools
export PATH=
# Enable SRILM
. $KALDI_ROOT/tools/env.sh
# Variable needed for proper data sorting
export LC_ALL=C

--- run_misp.sh ---
# Defining corpus directory
misp2022_corpus=
# Defining path to beamforIt executable file
bearmformit_path = 
# Defining path to python interpreter
python_path = 
# the directory to host coordinate information used to crop ROI 
data_roi =
# dictionary directory 
dict_dir= 
```

- **Run Training**

```
./run_misp.sh 
# options:
		--stage      -1  change the number to start from different training stages
```

## Requirments

- **Kaldi**

- **Python Packages:**

  numpy

  tqdm

  [jieba](https://github.com/fxsjy/jieba)

- **Other Tools:**

  [nara_wpe](https://github.com/fgnt/nara_wpe)

  [Beamformit](https://github.com/xanguera/BeamformIt)

  SRILM

## Citation

If you find this code useful in your research, please consider to cite the following papers:

```bibtex
@inproceedings{chen2022first,
  title={The first multimodal information based speech processing (misp) challenge: Data, tasks, baselines and results},
  author={Chen, Hang and Zhou, Hengshun and Du, Jun and Lee, Chin-Hui and Chen, Jingdong and Watanabe, Shinji and Siniscalchi, Sabato Marco and Scharenborg, Odette and Liu, Di-Yuan and Yin, Bao-Cai and others},
  booktitle={ICASSP 2022-2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={9266--9270},
  year={2022},
  organization={IEEE}
}
@inproceedings{2022misptask2,
author={Chen, Hang and Du, Jun and Dai, Yusheng and Lee, Chin-Hui and Siniscalchi, Sabato Marco and Watanabe, Shinji and Scharenborg, Odette and Chen, Jingdong and Yin, Bao-Cai and Pan, jia},
booktitle={Proc. INTERSPEECH 2022},
title={Audio-Visual Speech Recognition in MISP2021 Challenge: Dataset Release and Deep Analysis},
year={2022}}

```

## License

It is noted that the code can only be used for comparative or benchmarking purposes. Users can only use code supplied under a [License](./LICENSE) for non-commercial purposes.

