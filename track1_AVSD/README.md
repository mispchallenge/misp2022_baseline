# MISP2022 Challenge Track 1: Audio-Visual Speaker Diarization (Baseline)
![image](https://user-images.githubusercontent.com/117905504/201367602-b1165b6e-f274-473f-917a-34de27dd8602.png)
Fig.1. The illustration of network structure
## Data preparation
- **Speech enhancement** 

Weighted Prediction Error(WPE) dereverberation is used to reduce the reverberations of speech signals. The algorithms is implemented with open-source toolkit, [nara_wpe](https://github.com/fgnt/nara_wpe)
- **Lip ROI extraction**

In the dataset, we provide the position detection results of the face and lips, which can be used to extract the ROI of the face or lips (only the lips are used here). The extracted lip ROI size is 96 × 96 (*W × H*) .

You can run `data_prepare.sh` to perform the above two steps. (You need to modify the file path in the script).

## Visual embedding module
The visual embedding module is illustrated in the top row of Fig.1. On the basis of the original [lipreading model](https://github.com/mpc001/Lipreading_using_Temporal_Convolutional_Networks), we add three conformer blocks with 256 encoder dims, 4 attention heads, 32 conv kernel size and a 256-cell BLSTM to compute the visual embedding for speakers. The whole network can be regarded as visual voice activity detection (V-VAD) module. After pre-training the visual network as a V-VAD task, V-embeddings are equipped with capability that represent the states of speaking or silent. By feeding the embedding into the fully connected layers, we can get the probability of whether a speaker speaks in each frame. Combining the results of each speaker, we will get the initial diarization results.
## Audio embedding module
The audio embedding module is illustrated in the bottom row of Fig.1. We firstly use the NARA-WPE method to dereverberate the original audio signals. Then, We extracted 40-dimensional FBANKs with 25 ms frame length and 10 ms frame shift. We extract the audio embedding from FBANKS through 4 layers 2D CNN. Then, a fully connected layer projects high dimensional CNNs output to low dimensional A-embeddings. Unlike visual network, we don’t pre-train audio network on any other tasks and directly optimize it with audio-visual decoding network.
## Speaker embedding module
The speaker embedding module is illustrated in the middle row of Fig.1. To reduce the impact of the unreliable visual embedding, we use 100-dimensional i-vector extractor which was trained on Cn-celeb to get the i-vectors as the speaker embedding. We compute i-vectors through the oracle labels for each speaker in the training stage. And in the inference stage, we compute i-vectors through the initial diarization results from the visual embedding extraction module.
## Decoder module
We firstly repeat the three embeddings for different times to solve the problem of different frame shift between audio and video. Then, we combine them to get the total embedding. In the decoding block, we use 2-layer BLSTM with projection to further extract the features. Finally, we use a 1-layer BLSTM with projection and the fully connected layer to get the speech or non-speech probabilities for each speaker respectively. All of BLSTM layers contained 896 cells. In the post-processing stage, we first perform thresholding with the probabilities to produce a preliminary result and adopt the same approaches in [previous work](https://ieeexplore.ieee.org/document/9747067). Furthermore, [DOVER-Lap](https://github.com/desh2608/dover-lap) is used to fuse the results of 6-channels audio.
## Training process
First, we use the parameters of the pre-trained lipreading and train the V-VAD model with a learning rate of 10<sup>−4</sup>. Then, we freeze the visual network
parameters and train the audio network and audio-visual decoding block on synchronized middle-ﬁeld audio and video with a learning rate of 10<sup>−4</sup>. Finally,we unfreeze the visual network parameters and train the whole network jointly on synchronized middle-ﬁeld audio and video with a learning rate of 10<sup>−5</sup>.
## Citation

If you find this code useful in your research, please consider to cite the following papers:

```bibtex
@inproceedings{he2022,
  title={End-to-End Audio-Visual Neural Speaker Diarization},
  author={He, Mao-kui and Du, Jun and Lee, Chin-Hui},
  booktitle={Proc. Interspeech 2022},
  pages={1461--1465},
  year={2022}
}

```

## License

It is noted that the code can only be used for comparative or benchmarking purposes. Users can only use code supplied under a [License](./LICENSE) for non-commercial purposes.
