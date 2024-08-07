import time
from numpy import random
from numpy.core.numeric import full
import torch
import numpy as np
import os
import scipy.ndimage
import torchvision.transforms as transforms
import torch.nn.functional as F
from PIL import Image
import random
from torch._C import dtype, set_flush_denormal
import utils.basic
import utils.improc
import glob
import json
import imageio
import cv2
import re
import sys
from torchvision.transforms import ColorJitter, GaussianBlur
import random
np.random.seed(125)
torch.multiprocessing.set_sharing_strategy('file_system')
SEED = 125

def readFlow(fn):
    """ Read .flo file in Middlebury format"""
    # Code adapted from:
    # http://stackoverflow.com/questions/28013200/reading-middlebury-flow-files-with-python-bytes-array-numpy
    
    # WARNING: this will work on little-endian architectures (eg Intel x86) only!
    # print 'fn = %s'%(fn)
    with open(fn, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)
        if 202021.25 != magic:
            print('Magic number incorrect. Invalid .flo file')
            return None
        else:
            w = np.fromfile(f, np.int32, count=1)
            h = np.fromfile(f, np.int32, count=1)
            # print 'Reading %d x %d flo file\n' % (w, h)
            data = np.fromfile(f, np.float32, count=2*int(w)*int(h))
            # Reshape data into 3D array (columns, rows, bands)
            # The reshape here is for visualization, the original code is (w,h,2)
            return np.resize(data, (int(h), int(w), 2))

def readImage(name):
    if name.endswith('.pfm') or name.endswith('.PFM'):
        data = readPFM(name)
        if len(data.shape)==3:
            return data[:,:,0:3]
        else:
            return data
    return imageio.imread(name)

class SintelDataset(torch.utils.data.Dataset):
    def __init__(self,
                 dataset_location='/data/sintel',
                 dset='t',
                 use_augs=False,
                 test_mode=False,
                 crop_size=(368, 496),
                 scene_name=None,
                 scene_num_frames=None,
                 split='clean'
    ):
        self.set_seed(SEED)
        rgb_root = os.path.join(dataset_location, f'training/{split}')
        flow_root = os.path.join(dataset_location, f'training/flow')

        self.use_augs = use_augs
        self.crop_size = crop_size
        self.test_mode = test_mode
        
        folder_names = [folder.split('/')[-1] for folder in glob.glob(os.path.join(rgb_root, "*"))]
        folder_names = sorted(folder_names)
        # print(len(folder_names))
        # exit()
        # print('folder_names', folder_names)
        if scene_name is not None:
            assert scene_name in folder_names
            print(f"Using scene {scene_name}")
            folder_names = [scene_name]
        self.rgb0_paths = []
        self.rgb1_paths = []
        self.flow_paths = []

        for ii, folder_name in enumerate(folder_names):
            cur_rgb_path = os.path.join(rgb_root, folder_name)
            cur_flow_path = os.path.join(flow_root, folder_name)
            
            img_names = [folder.split('/')[-1].split('.')[0] for folder in glob.glob(os.path.join(cur_rgb_path, "*"))]
            img_names = sorted(img_names)
            # if scene_num_frames is not None:
            #     assert scene_num_frames < len(img_names)
            #     img_names = img_names[:scene_num_frames]
            
            # print('img_names', img_names)
            S_here = len(img_names)

            for si in range(0, S_here-1):
                rgb0_path = os.path.join(cur_rgb_path, '%s.png' % img_names[si])
                rgb1_path = os.path.join(cur_rgb_path, '%s.png' % img_names[si+1])
                flow_path = os.path.join(cur_flow_path, '%s.flo' % img_names[si])

                self.rgb0_paths.append(rgb0_path)
                self.rgb1_paths.append(rgb1_path)
                self.flow_paths.append(flow_path)
        # print(self.rgb0_paths)
        # print(self.rgb1_paths)
        # print(self.flow_paths)
        # exit()
        print('found %d samples in %s' % (len(self.rgb0_paths), dataset_location))
    
    def set_seed(self, seed = 10):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    def __getitem__(self, index):
        cur_rgb0_path = self.rgb0_paths[index]
        cur_rgb1_path = self.rgb1_paths[index]
        cur_flow_path = self.flow_paths[index]

        rgb0 = readImage(cur_rgb0_path)
        rgb1 = readImage(cur_rgb1_path)
        flow = readFlow(cur_flow_path)


        if not self.test_mode:
            rgbs = [rgb0,rgb1]

            if self.use_augs:
                rgbs, trajs, visibles = self.add_photometric_augs(rgbs, trajs, visibles)
                rgbs, occs, masks, trajs = self.add_spatial_augs(rgbs, occs, masks, trajs, visibles)
            else:
                rgbs, flow, xrange, yrange = self.just_crop(rgbs, flow)

            rgb0, rgb1 = rgbs
        
        rgb0 = torch.from_numpy(rgb0).permute(2,0,1) # 3,H,W
        rgb1 = torch.from_numpy(rgb1).permute(2,0,1) # 3,H,W
        flow = torch.from_numpy(flow).permute(2,0,1) # 2,H,W

        # add_spatial_augs(rgbs, flow)
        prep_rgb0 = utils.improc.preprocess_color(rgb0)
        prep_rgb1 = utils.improc.preprocess_color(rgb1)
        sample = {
            # 'rgb0': rgb0,
            # 'rgb1': rgb1,
            'rgb0': prep_rgb0,
            'rgb1': prep_rgb1,
            'flow': flow,
            'xrange': xrange,
            'yrange': yrange
        }
        
        return sample, True

    def just_crop(self, rgbs, flow):
        self.set_seed(SEED)
        S = len(rgbs)
        H, W = rgbs[0].shape[:2]
        ############ spatial transform ############

        H_new = H
        W_new = W

        if H == self.crop_size[0]:
            y0 = 0
        else:
            y0 = np.random.randint(0, H_new - self.crop_size[0])
            
        if W == self.crop_size[1]:
            x0 = 0
        else:
            x0 = np.random.randint(0, W_new - self.crop_size[1])

        rgbs = [rgb[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]] for rgb in rgbs]
        flow = flow[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
        xrange = (x0, x0+self.crop_size[1])
        yrange = (y0, y0+self.crop_size[0])
        return rgbs, flow, xrange, yrange
    
    def __len__(self):
        return len(self.rgb0_paths)


