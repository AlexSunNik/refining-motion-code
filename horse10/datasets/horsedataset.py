import torch
import numpy as np
import os
import scipy.ndimage
import torchvision.transforms as transforms
import torch.nn.functional as F
from PIL import Image
import random
import glob
import json
import imageio
import cv2
from datasets.dataset import PointDataset
import pickle

# https://mmpose.readthedocs.io/en/latest/dataset_zoo/2d_animal_keypoint.html#horse-10

class HorseDataset(PointDataset):
    def __init__(self,
                 dataset_location,
                 S=8,
                 strides=[1,2,3],
                #  strides=[16],
                 crop_size=None,
                 use_augs=False,
                 is_training=True):
        print('loading horse dataset...')
        super().__init__(
            dataset_location=dataset_location,
            S=S,
            strides=strides,
            crop_size=crop_size, # raw data is 512,768
            use_augs=use_augs,
            is_training=is_training
        )

        self.dataset_location = dataset_location
        self.S = S
        self.anno_path = os.path.join(self.dataset_location, "seq_annotation.pkl")
        with open(self.anno_path, 'rb') as f:
            self.annotation = pickle.load(f)
        
        self.video_names = list(self.annotation.keys())
        print(f"found {len(self.annotation)} unique videos in {dataset_location}")

        self.all_video_names = []
        self.all_full_idx = []
        self.all_kp_idx = []

        rgb = None
        
        for video_name in self.video_names:
            video = self.annotation[video_name]
            S_local = len(video)
            stride = 1
            # for stride in strides:
            # for ii in range(0, max(S_local-self.S*stride,1), 8):
            for ii in range(0, max(S_local-self.S*stride,1), 32):
                full_idx = ii + np.arange(self.S)*stride
                full_idx = [ij for ij in full_idx if ij < S_local]
                if len(full_idx) > 8:
                    samples = [video[idx] for idx in full_idx]
                    visibs = []
                    trajs = []
                    for sample in samples:
                        visibs.append(np.squeeze(sample['keypoints_visible'], 0))
                        trajs.append(np.squeeze(sample['keypoints'], 0))
                    visibs = np.stack(visibs)
                    trajs = np.stack(trajs)

                    if rgb is None:
                        img_path = samples[0]['img_path']
                        img_path = self.dataset_location + img_path
                        rgb = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
                        H,W,C = rgb.shape
                        
                    for si in range(S):
                        # avoid 8px edge, since these are not really visible (according to adam)
                        oob_inds = np.logical_or(
                            np.logical_or(trajs[si,:,0] < 8, trajs[si,:,0] > W-8),
                            np.logical_or(trajs[si,:,1] < 8, trajs[si,:,1] > H-8))
                        visibs[si,oob_inds] = 0

                    vis01 = visibs[:2].sum(0) == 2
                    vis3 = visibs.sum(0) > 3
                    vis_ok = vis01 & vis3

                    S, N, _ = trajs.shape

                    for ni in range(N):
                        if vis_ok[ni]:
                            self.all_video_names.append(video_name)
                            self.all_full_idx.append(full_idx)
                            self.all_kp_idx.append(ni)
                        
        print(f"found {len(self.all_video_names)} samples in {dataset_location}")
            

    def getitem_helper(self, index):
        # print(index)
        # print('index', index)
        video_name = self.all_video_names[index]
        full_idx = self.all_full_idx[index]
        full_idx = full_idx[::4]
        # print(len(full_idx))
        ni = self.all_kp_idx[index]
        # print("index", index)
        # print(video_name)
        # print(full_idx)
        # print(ni)

        video = self.annotation[video_name]
        samples = [video[idx] for idx in full_idx]
        
        rgbs = []
        trajs = []
        visibs = []
        for sample in samples:
            img_path = sample['img_path']
            img_path = self.dataset_location + img_path
            rgbs.append(cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB))
            trajs.append(np.squeeze(sample['keypoints'], 0))
            visibs.append(np.squeeze(sample['keypoints_visible'], 0))

        rgbs = np.stack(rgbs, axis=0)
        trajs = np.stack(trajs, axis=0)
        visibs = np.stack(visibs, axis=0)

        S,H,W,C = rgbs.shape
        S,N,D = trajs.shape

        trajs = trajs[:,ni]
        visibs = visibs[:,ni]

        S = len(trajs)
        for si in range(1,S):
            if visibs[si]==0:
                trajs[si] = trajs[si-1]
        
        d = {
            'rgbs': rgbs.astype(np.uint8), # S, H, W, C
            'trajs': trajs.astype(np.int64), # S, 2
            'visibs': visibs.astype(np.float32), # S
            'video_name': video_name,
            'full_idx': full_idx
        }

        # print(d['rgbs'].shape)
        # print(d['trajs'].shape)
        # print(d['visibs'].shape)

        # print("index", index)
        return d

    def __len__(self):
        return len(self.all_video_names)