# MISP2022 Challenge Track 1: Audio-Visual Speaker Diarization (Baseline)
![image](https://user-images.githubusercontent.com/117905504/201367602-b1165b6e-f274-473f-917a-34de27dd8602.png)
Fig.1. The illustration of network structure
## Data preparation
- **Speech enhancement** 

Weighted Prediction Error(WPE) dereverberation is used to reduce the reverberations of speech signals. The algorithms is implemented with open-source toolkit, [nara_wpe](https://github.com/fgnt/nara_wpe)
- **Lip ROI extraction**

In the dataset, we provide the position detection results of the face and lips, which can be used to extract the ROI of the face or lips (only the lips are used here). The extracted lip ROI size is 96 × 96 (*W × H*) .

You can run `data_prepare.sh` to perform the above two steps. (You need to modify the file path in the script).

## Visual embedding
The visual network is illustrated in the top row of Fig.1. On the basis of the original [lipreading model](https://github.com/mpc001/Lipreading_using_Temporal_Convolutional_Networks), we add three conformer blocks with 256 encoder dims, 4 attention heads, 32 conv kernel size and a 256-cell BLSTM to compute the visual embedding for speakers. The whole network can be regarded as visual voice activity detection (V-VAD) module. After pre-training the visual network as a V-VAD task, V-embeddings are equipped with capability that represent the states of speaking or silent. By feeding the embedding into the fully connected layers, we can get the probability of whether a speaker speaks in each frame. Combining the results of each speaker, we will get the initial diarization results.
## Audio embedding
The audio network is illustrated in the bottom row of Fig.1. We firstly use the NARA-WPE method to dereverberate the original audio signals. Then, We extracted 40-dimensional FBANKs with 25 ms frame length and 10 ms frame shift. We extract the audio embedding from FBANKS through 4 layers 2D CNN. Then, a fully connected layer projects high dimensional CNNs output to low dimensional A-embeddings. Unlike visual network, we don’t pre-train audio network on any other tasks and directly optimize it with audio-visual decoding network.
## Speaker embedding
To reduce the impact of the unreliable visual embedding, we use 100-dimensional i-vector extractor which was trained on Cn-celeb to get the i-vectors as the speaker embedding. We compute i-vectors through the oracle labels for each speaker in the training stage. And in the inference stage, we compute i-vectors through the initial diarization results from the visual embedding extraction module.
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
