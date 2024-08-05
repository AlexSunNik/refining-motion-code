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
from torch.utils.data import Dataset
import imageio
import cv2
from datasets.dataset import PointDataset
import pickle
import albumentations as A

# https://mmpose.readthedocs.io/en/latest/dataset_zoo/2d_animal_keypoint.html#horse-10

class HorseDataset(Dataset):
    def __init__(self,
                 dataset_location,
                 S=8,
                 strides=[1,2,3],
                #  strides=[16],
                 crop_size=None,
                 use_augs=False,
                 is_training=True,
                 pkl_file=None):
        print('loading horse dataset...')
        self.strides = strides
        self.crop_size = crop_size
        self.use_augs = use_augs
        self.is_training = is_training
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
        
        with open(pkl_file, 'rb') as f: 
            self.all_data = pickle.load(f)
                        
        # print(f"found {len(self.all_video_names)} samples in {dataset_location}")
            

    # def getitem_helper(self, index):
    def __getitem__(self, index):
        # print(index)
        # print('index', index)
        data = self.all_data[index]
        video_name = data[0]
        full_idx = data[1]
        trajs = data[2]
        

        video = self.annotation[video_name]
        samples = [video[idx] for idx in full_idx]
        
        rgbs = []
        for sample in samples:
            img_path = sample['img_path']
            img_path = self.dataset_location + img_path
            rgbs.append(cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB))

        rgbs = np.stack(rgbs, axis=0)

        S,H,W,C = rgbs.shape
        S,N,D = trajs.shape


        S = len(trajs)
        
        rgbs = rgbs.astype(np.uint8)
        rgbs_cr = []
        for i in range(len(rgbs)):
            crop_resize = A.Compose([
                    A.Resize(*self.crop_size)])

            data = crop_resize(image=rgbs[i])
            rgbs_cr.append(data['image'])
        
        rgbs = np.stack(rgbs_cr)
        rgbs = np.moveaxis(rgbs, -1, 1).astype(np.uint8)
        # print(rgbs.shape)
        d = {
            'rgbs': rgbs.astype(np.uint8), # S, H, W, C
            'trajs': trajs.numpy().astype(np.int64), # S, 2
            'visibs': trajs.numpy().astype(np.float32), # S
        }

        # print(d['rgbs'].shape)
        # print(d['trajs'].shape)
        # print(d['visibs'].shape)

        # print("index", index)
        return d

    def __len__(self):
        return len(self.all_data)