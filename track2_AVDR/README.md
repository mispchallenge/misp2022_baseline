# MISP2022 Challenge Track 2: Audio-Visual Diarization and Recognition (Baseline)
![Baseline_System](https://user-images.githubusercontent.com/88126124/201823221-47cad24f-a2a0-401a-a814-766ab927f88d.png)
Fig.1. The illustration of network structure

Fig. 1 shows the baseline AVDR system, which consists of an AVSD module followed by an AVSR module. The AVSD module also serves as the baseline system for Track 1. In this section, we elaborate the architecture and training process of the AVSD and AVSR modules, and provide the details about joining the AVSD and AVSR modules for decoding.

## Data preparation
- **Speech enhancement** 

Weighted Prediction Error(WPE) dereverberation and BeamformIt are used to reduce the reverberations of speech signals. The algorithms are implemented with open-source toolkit, [nara_wpe](https://github.com/fgnt/nara_wpe)[BeamformIt](https://github.com/xanguera/BeamformIt)
- **Lip ROI extraction**

In the dataset, we provide the position detection results of the face and lips, which can be used to extract the ROI of the face or lips (only the lips are used here). The extracted lip ROI size is 96 × 96 (*W × H*) .
