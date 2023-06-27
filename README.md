# Decoupled Consistency for Semi-supervised Medical Image Segmentation
by Faquan Chen, Jingjing Fei, Yaqi Chen, and Chenxi Huang*.
## Introduction
Official code for "Decoupled Consistency for Semi-supervised Medical Image Segmentation". (MICCAI 2023)
## Requirements
This repository is based on PyTorch 1.8.1, CUDA 10.1, and Python 3.6.13. All experiments in our paper were conducted on an NVIDIA GeForce RTX 1080ti GPU with an identical experimental setting.
## Usage
We provide code, data_split, and models for PROMISE12 and ACDC datasets.

Data could be got at PROMISE12 and ACDC.

To train a model,
```
python train_dcnet_prostate.py  #for Prostate training
python train_dcnet_acdc.py  #for ACDC training
```
To test a model,
```
python test_prostate.py  #for Prostate testing
python test_ACDC.py  #for ACDC testing
```
## Citation

## Acknowledgements
Our code is adapted from [MC-Net](https://github.com/ycwu1997/MC-Net), [SSNet](https://github.com/ycwu1997/SS-Net), and [SSL4MIS](https://github.com/HiLab-git/SSL4MIS). Thanks to these authors for their valuable works and hope our model can promote the relevant research as well.
## Questions
If you have any questions, welcome contact me at 'chenfaquan@stu.xmu.edu.cn'
