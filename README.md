# Refining Pre-Trained Motion Models

This repository is for the method introduced in the following paper accepted by ICRA2024:

Refining Pre-Trained Motion Models\
Xinglong Sun, Adam W. Harley, Leonidas J. Guibas

Arxiv Link: https://arxiv.org/pdf/2401.00850.pdf
## Introduction
Given the difficulty of manually annotating motion in video, the current best motion estimation methods are trained with synthetic data, 
and therefore struggle somewhat due to a train/test gap. Self-supervised methods hold the promise of training directly on real video, 
but typically perform worse. These include methods trained with warp error (i.e., color constancy) combined with smoothness terms, 
and methods that encourage cycle-consistency in the estimates (i.e., tracking backwards should yield the opposite trajectory as tracking forwards). 
In this work, we take on the challenge of improving state-of-the-art supervised models with self-supervised training. 
We find that when the initialization is supervised weights, most existing self-supervision techniques actually make performance worse instead of better, 
which suggests that the benefit of seeing the new data is overshadowed by the noise in the training signal. Focusing on obtaining a 
"clean" training signal from real-world unlabelled video, we propose to separate label-making and training into two distinct stages. 
In the first stage, we use the pre-trained model to estimate motion in a video, and then select the subset of motion estimates which we can verify with cycle-consistency. 
This produces a sparse but accurate pseudo-labelling of the video. In the second stage, we fine-tune the model to reproduce these outputs, 
while also applying augmentations on the input. We complement this boot-strapping method with simple techniques that densify and re-balance the pseudo-labels, 
ensuring that we do not merely train on ``easy'' tracks. We show that our method yields reliable gains over fully-supervised methods in real videos, 
for both short-term (flow-based) and long-range (multi-frame) pixel tracking. 

<div align="center">
  <img src="Figs/flowchart.png" width="100%">
  Overview of our method.
</div>

## Results on KITTI
<div align="center">
  <img src="Figs/result.png" width="100%">
  KITTI Depth Completion Results.
</div>
Link to our public results on KITTI test server:

https://www.cvlibs.net/datasets/kitti/eval_depth_detail.php?benchmark=depth_completion&result=c00c3b4d967f78cb9e1522ebd062f763b7668f7d

## Some Visualizations
<div align="center">
  <img src="Figs/viz1.png" width="100%">
  From left to right: groundtruth dense depth map, guidance RGB image, inference from our model, and inference from PENet.
</div>

<div align="center">
  <img src="Figs/viz2.png" width="100%">
  From left to right: groundtruth dense depth map, guidance RGB image, inference from our model, and inference from PENet.
</div>

## Prerequisites
### Datasets
Please follow the KITTI depth completion dataset downloading instruction here:

https://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_completion

## Train
To train the baseline/unpruned network, run:
```
python3 train.py
```
As mentioned in our paper, we study our deformable refinement module on top of the model backbone based on ENet from the paper PENet (https://arxiv.org/pdf/2103.00783.pdf). For faster convergence of ReDC, you could download the pretrained PENet model from here:
https://drive.google.com/file/d/1RDdKlKJcas-G5OA49x8OoqcUDiYYZgeM/view?usp=sharing

In train.py, we extract the backbone weights from PENet and initialize the backbone used in ReDC.

### Pretrained Models
We also release our pretrained model here: 
https://drive.google.com/file/d/1wE8QLI_fCpGVLKqhqBf5Wg8A08eHJDfv/view?usp=sharing

## Acknowledgement
Some dataloading and evaluation code is from:
https://github.com/JUGGHM/PENet_ICRA2021

## Citations
If you find this repo useful to your project or research, please cite our paper below:

@inproceedings{sun2023revisiting,\
  title={Revisiting deformable convolution for depth completion},\
  author={Sun, X and Ponce, J and Wang, Y-X},\
  booktitle={IEEE/RSJ International Conference on Intelligent Robots and Systems},\
  year={2023}\
}
