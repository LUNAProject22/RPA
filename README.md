# RCA: Region Conditioned Adaptation for Visual Abductive Reasoning (ACM Multimedia 2024)

This is the official implementation of the paper [RCA: Region Conditioned Adaptation for Visual Abductive Reasoning](https://arxiv.org/pdf/2303.10428). We achieved the top rank on the official Sherlock Abductive Reasoning [Leaderboard](https://leaderboard.allenai.org/sherlock/submissions/public) and the DHPR retrieval performance.

<div align="center">
  <img src="./images/overview.png" width="800px"/>
</div>

- [Updates](#updates)
- [Model Zoo](#model-zoo)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Contributors](#contributors)
- [Citing](#citing)
- [Acknowledgement](#Acknowledgement)

### July 19, 2024
* Release RCA-V1 version (the version used in paper) to public.

## Model Zoo
| Model | Backbone | Tuned (M↓) | im→txt (↓) | txt→im (↓) | P@1→I (↑) | GT/Auto-Box (↑) | Human Acc (↑) |
|----------|------------|-----------|--------------|-------------|------------|-------------|
| LXMERT [46] from [19] | F-RCNN | NA | 51.10 | 48.80 | 14.90 | 69.50 / 30.30 | 21.10 |
| UNITER [7] from [19] | NA | 40.40 | 40.00 | 19.80 | 73.00 / 33.30 | 22.90 |
| CPT [51] from [19] | RN50×64 | NA | 16.35 | 17.72 | 33.44 | 87.22 / 40.60 | 27.12 |
| CPT [51] from [19] | 149.62 | 19.85 | 21.64 | 30.56 | 85.33 / 36.60 | 21.31 |
| CPT [51] (our impl) | 149.77 | 19.46 | 21.14 | 31.19 | 85.00 / 38.84 | 23.09 |
| Full Fine-Tuning (R-CTX) | 149.77 | 15.63 | 18.20 | 33.76 | 86.19 / 40.78 | 27.32 |
| Our RCA (R-CTX) | ViT-B-16 | 42.26 | 15.59 | 18.04 | 33.83 | 86.36 / 40.79 | 26.39 |
| → Mixed Prompts | 42.26 | 14.39 | 16.91 | 34.84 | 87.73 / 41.64 | 26.11 |
| → Dual-Contrast Loss | 42.26 | 13.92 | 16.58 | 35.42 | 88.08 / 42.32 | 27.51 |
| CPT [51] (our impl) | 428.53 | 13.08 | 14.91 | 37.21 | 87.85 / 41.99 | 29.58 |
| Our RCA (R-CTX) | ViT-L-14 | 89.63 | 11.36 | 13.87 | 38.55 | 88.62 / 42.30 | 31.72 |
| → Mixed Prompts (336) | 89.63 | 10.48 | 12.95 | 39.68 | 89.66 / 43.61 | 31.23 |
| → Dual-Contrast Loss | 89.63 | 10.14 | 12.65 | 40.36 | 89.72 / 44.73 | 31.74 |


## Installation

## Quick Start
### Train

### Evaluate

## Contributors
RCA is coded and maintained by [Dr. Hao Zhang](https://hzhang57.github.io/).


## Citing
If you find the paper helpful for your work, please consider citing the following:

```
@inproceedings{hesselhwang2022abduction,
  title={{RCA: Region Conditioned Adaptation for Visual Abductive Reasoning}},
  author={Hao Zhang, Yeo Keat Ee, Basura Fernando},
  booktitle={ACM Multimedia},
  year={2024}
}
```

```
@inproceedings{hesselhwang2022abduction,
  title={{The Abduction of Sherlock Holmes: A Dataset for Visual Abductive Reasoning}},
  author={*Hessel, Jack and *Hwang, Jena D and Park, Jae Sung and Zellers, Rowan and Bhagavatula, Chandra and Rohrbach, Anna and Saenko, Kate and Choi, Yejin},
  booktitle={ECCV},
  year={2022}
}
```

```
@article{10568360,
  author={Charoenpitaks, Korawat and Nguyen, Van-Quang and Suganuma, Masanori and Takahashi, Masahiro and Niihara, Ryoma and Okatani, Takayuki},
  journal={IEEE Transactions on Intelligent Vehicles}, 
  title={Exploring the Potential of Multi-Modal AI for Driving Hazard Prediction}, 
  year={2024},
  volume={},
  number={},
  pages={1-11},
  keywords={Hazards;Cognition;Videos;Automobiles;Accidents;Task analysis;Natural languages;Vision;Language;Reasoning;Traffic Accident Anticipation},
  doi={10.1109/TIV.2024.3417353}
}

```
## Acknowledgement
Thanks for the following Github repositories:
- https://github.com/allenai/sherlock
- https://github.com/mlfoundations/open_clip
- https://github.com/DHPR-dataset/DHPR-dataset

