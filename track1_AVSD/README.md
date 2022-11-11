# MISP2022 Challenge Track 1: Audio-Visual Speaker Diarization (Baseline)
![image](https://user-images.githubusercontent.com/117905504/201367602-b1165b6e-f274-473f-917a-34de27dd8602.png)
Fig.1. The illustration of network structure
## Data preparation
- **Speech enhancement** 

Weighted Prediction Error(WPE) dereverberation is used to reduce the reverberations of speech signals. The algorithms is implemented with open-source toolkit, [nara_wpe](https://github.com/fgnt/nara_wpe)
- **Lip ROI extraction**

In the dataset, we provide the position detection results of the face and lips, which can be used to extract the ROI of the face or lips (only the lips are used here). The extracted lip ROI size is 96 × 96 (*W × H*) .

You can run `data_prepare.sh` to perform the above two steps. (You need to modify the file path in the script).

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
