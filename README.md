# ConditionCDVAE+

Our study presents an improved deep generative model based on the Crystal Diffusion Variational  Autoencoder ([CDVAE](https://github.com/txie-93/cdvae)), ConditionCDVAE+, which enables the inverse design of van der Waals (vdW) heterostructures.
This is an implemented code from a paper titled "Deep generative model for the inverse design of 2D  heterostructures".
## Installation
Run the following command to install the environment:
```
conda env create -f environment.yml
```

Modify the following environment variables in `.env`.

- `PROJECT_ROOT`: path to the folder that contains this repo
- `HYDRA_JOBS`: path to a folder to store hydra outputs
- `WABDB`: path to a folder to store wabdb outputs

## Datasets

You can find a small sample of the dataset in `data/`, 
including the data used for ConditionCDVAE+ training. 
The complete data are available from the corresponding author upon reasonable request.

## Training and evaluation

training without condition command:
```
python ccdvaeplus/run.py data=j2dh-8 expname=j2dh-8 model=vae_withoutcondition
```

training  command:
```
python ccdvaeplus/run.py data=j2dh-8 expname=j2dh-8 model=vae
```

To generate materials without condition, run the following command:
```
python scripts/evaluate.py --model_path MODEL_PATH --tasks gen
```

To generate materials with condition:
```
python scripts/Condition_guided_eval.py --model_path MODEL_PATH --tasks gen --prop PROP or --formula FORMULA
```

compute reconstruction & generation metrics (only on random gen data):
```
python scripts/compute_metrics.py --root_path ROOT_PATH --tasks recon gen
```

# References

CDVAE
```
@inproceedings{
xie2022crystal,
title={Crystal Diffusion Variational Autoencoder for Periodic Material Generation},
author={Tian Xie and Xiang Fu and Octavian-Eugen Ganea and Regina Barzilay and Tommi S. Jaakkola},
booktitle={International Conference on Learning Representations},
year={2022},
url={https://openreview.net/forum?id=03RLpj-tc_}
}
```

EquiformerV2
```
@inproceedings{
    equiformer_v2,
    title={{EquiformerV2: Improved Equivariant Transformer for Scaling to Higher-Degree Representations}}, 
    author={Yi-Lun Liao and Brandon Wood and Abhishek Das* and Tess Smidt*},
    booktitle={International Conference on Learning Representations (ICLR)},
    year={2024},
    url={https://openreview.net/forum?id=mCOBKZmrzD}
}
```

LMF
```
@inproceedings{liuEfficientLowrankMultimodal2018,
  title = {Efficient {{Low-rank Multimodal Fusion With Modality-Specific Factors}}},
  booktitle = {Proceedings of the 56th {{Annual Meeting}} of the {{Association}} for {{Computational Linguistics}} ({{Volume}} 1: {{Long Papers}})},
  author = {Liu, Zhun and Shen, Ying and Lakshminarasimhan, Varun Bharadhwaj and Liang, Paul Pu and Bagher Zadeh, AmirAli and Morency, Louis-Philippe},
  year = {2018},
  pages = {2247--2256},
  publisher = {Association for Computational Linguistics},
  address = {Melbourne, Australia},
  doi = {10.18653/v1/P18-1209},
  url = {http://aclweb.org/anthology/P18-1209}
}
```

GAN
```
@article{zhengConditionalWassersteinGenerative2020,
  title = {Conditional {{Wasserstein}} Generative Adversarial Network-Gradient Penalty-Based Approach to Alleviating Imbalanced Data Classification},
  author = {Zheng, Ming and Li, Tong and Zhu, Rui and Tang, Yahui and Tang, Mingjing and Lin, Leilei and Ma, Zifei},
  year = {2020},
  journal = {Information Sciences},
  volume = {512},
  pages = {1009--1023},
  issn = {00200255},
  doi = {10.1016/j.ins.2019.10.014},
  url = {https://linkinghub.elsevier.com/retrieve/pii/S0020025519309715}
}
```