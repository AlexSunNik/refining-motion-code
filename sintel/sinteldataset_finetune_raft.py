import time
from numpy import random
from numpy.core.numeric import full
import torch
import numpy as np
import os
import torchvision.transforms as transforms
import torch.nn.functional as F
from PIL import Image
import random
from torch._C import dtype, set_flush_denormal
import utils.basic
import utils.improc
import glob
import imageio
import matplotlib.pyplot as plt
from torchvision.transforms import ColorJitter, GaussianBlur
import random
import pickle
from augmentor import *
torch.multiprocessing.set_sharing_strategy('file_system')
SEED = 125
COLOR_THRESH = 0.005

def backwarp_using_2d_flow(im1, flow0, binary_feat=False):
    B, C, Y, X = list(im1.shape)
    cloud0 = utils.basic.gridcloud2d(B, Y, X).cpu()
    cloud0_displacement = flow0.reshape(B, 2, Y*X).permute(0, 2, 1)
    resampling_coords = cloud0 + cloud0_displacement
    return resample2d(im1, resampling_coords, binary_feat=binary_feat)

def resample2d(im, xy, binary_feat=False):
    B, C, Y, X = list(im.shape)
    xy = utils.basic.normalize_gridcloud2d(xy, Y, X)
    xy = torch.reshape(xy, [B, Y, X, 2])
    im = F.grid_sample(im, xy)
    if binary_feat:
        im = im.round()
    return im

class InputPadder:
    """ Pads images such that dimensions are divisible by 8 """

    def __init__(self, dims, mode='sintel', padding_factor=8):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // padding_factor) + 1) * padding_factor - self.ht) % padding_factor
        pad_wd = (((self.wd // padding_factor) + 1) * padding_factor - self.wd) % padding_factor
        if mode == 'sintel':
            self._pad = [pad_wd // 2, pad_wd - pad_wd // 2, pad_ht // 2, pad_ht - pad_ht // 2]
        else:
            self._pad = [pad_wd // 2, pad_wd - pad_wd // 2, 0, pad_ht]

    def pad(self, *inputs):
        return [F.pad(x, self._pad, mode='replicate') for x in inputs]

    def unpad(self, x):
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht - self._pad[3], self._pad[0], wd - self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]


def coords_grid(b, h, w, homogeneous=False, device=None):
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w))  # [H, W]

    stacks = [x, y]

    if homogeneous:
        ones = torch.ones_like(x)  # [H, W]
        stacks.append(ones)

    grid = torch.stack(stacks, dim=0).float()  # [2, H, W] or [3, H, W]

    grid = grid[None].repeat(b, 1, 1, 1)  # [B, 2, H, W] or [B, 3, H, W]

    if device is not None:
        grid = grid.to(device)

    return grid

def bilinear_sample(img, sample_coords, mode='bilinear', padding_mode='zeros', return_mask=False):
    # img: [B, C, H, W]
    # sample_coords: [B, 2, H, W] in image scale
    if sample_coords.size(1) != 2:  # [B, H, W, 2]
        sample_coords = sample_coords.permute(0, 3, 1, 2)

    b, _, h, w = sample_coords.shape

    # Normalize to [-1, 1]
    x_grid = 2 * sample_coords[:, 0] / (w - 1) - 1
    y_grid = 2 * sample_coords[:, 1] / (h - 1) - 1

    grid = torch.stack([x_grid, y_grid], dim=-1)  # [B, H, W, 2]

    img = F.grid_sample(img, grid, mode=mode, padding_mode=padding_mode, align_corners=True)

    if return_mask:
        mask = (x_grid >= -1) & (y_grid >= -1) & (x_grid <= 1) & (y_grid <= 1)  # [B, H, W]

        return img, mask

    return img

# flow: B, 1, H, W
def flow_warp(feature, flow, mask=False, padding_mode='zeros'):
    b, c, h, w = feature.size()
    assert flow.size(1) == 2

    grid = coords_grid(b, h, w).to(flow.device) + flow  # [B, 2, H, W]

    return bilinear_sample(feature, grid, padding_mode=padding_mode,
                           return_mask=mask)

def forward_backward_consistency_check(fwd_flow, bwd_flow,
                                    #    alpha=0.01 * 0.25,
                                    #    beta=0.5 * 0.25,
                                        alpha=0.01 * 0.2,
                                       beta=0.5 * 0.2,
                                       ):
    # fwd_flow, bwd_flow: [B, 2, H, W]
    # alpha=0.01 and beta=0.5 is from UnFlow (https://arxiv.org/abs/1711.07837)
    assert fwd_flow.dim() == 4 and bwd_flow.dim() == 4
    assert fwd_flow.size(1) == 2 and bwd_flow.size(1) == 2
    flow_mag = torch.norm(fwd_flow, dim=1) + torch.norm(bwd_flow, dim=1)  # [B, H, W]

    warped_bwd_flow = flow_warp(bwd_flow, fwd_flow)  # [B, 2, H, W]
    warped_fwd_flow = flow_warp(fwd_flow, bwd_flow)  # [B, 2, H, W]

    diff_fwd = torch.norm(fwd_flow + warped_bwd_flow, dim=1)  # [B, H, W]
    diff_bwd = torch.norm(bwd_flow + warped_fwd_flow, dim=1)

    threshold = alpha * flow_mag + beta

    fwd_occ = (diff_fwd > threshold).float()  # [B, H, W]
    bwd_occ = (diff_bwd > threshold).float()

    return fwd_occ.unsqueeze(1), bwd_occ.unsqueeze(1)

def run_prep(raft_model, d, device, sw=None, name_ext="", return_feat=False):
    total_loss = torch.tensor(0.0, requires_grad=True).to(device)

    metrics = {}
    metrics['l1'] = None
    metrics['l2'] = None
    # metrics['ate_all'] = 0.0
    
    rgb0 = d['rgb0'].to(device).float() # B, C, H, W
    rgb1 = d['rgb1'].to(device).float() # B, C, H, W
    flow_g = d['flow'].to(device).float() # B, 2, H, W

    C, H, W = rgb0.shape
    assert(C==3)
    D, H, W = flow_g.shape
    assert(D==2)
    rgb0 = rgb0.unsqueeze(0)
    rgb1 = rgb1.unsqueeze(0)
    prep_rgb0 = utils.improc.preprocess_color(rgb0)
    prep_rgb1 = utils.improc.preprocess_color(rgb1)
    rgbs = torch.stack([rgb0, rgb1], dim=1) # B,S,C,H,W
    S = 2

    clip = max(min(torch.max(torch.abs(flow_g[0:1])),50.0),2.0)
        
    if sw is not None and sw.save_this:
        sw.summ_rgbs(f'0_inputs{name_ext}/rgbs', utils.improc.preprocess_color(rgbs[0:1]).unbind(1))
        sw.summ_flow(f'0_inputs{name_ext}/flow_g', flow_g[0:1], clip=clip)

    with torch.no_grad():
        flow_fw, _ = raft_model(prep_rgb0, prep_rgb1, iters=32)
        flow_bw, _ = raft_model(prep_rgb1, prep_rgb0, iters=32)
    
    flow_fw = flow_fw.cpu()
    flow_bw = flow_bw.cpu()
    occ_fw, occ_bw = forward_backward_consistency_check(flow_fw, flow_bw)
    recon_rgb0 = backwarp_using_2d_flow(prep_rgb1, flow_fw)
    diff_map = torch.abs(recon_rgb0 - prep_rgb0)
    diff_map = diff_map.mean(dim=1)
    color_consty_map = diff_map < COLOR_THRESH
    print("Color Consistent Ratio:", torch.sum(color_consty_map) / color_consty_map.numel())

    flow_err = torch.norm(flow_fw - flow_g, dim=1, keepdim=True)

    flow_err_clamp = flow_err.clamp(max=5.0)/5.0

    flow_err_unexpected = flow_err_clamp * (1.0 - occ_fw)
    mean_flow_err_unexpected = utils.basic.reduce_masked_mean(flow_err, 1.0-occ_fw)

    if sw is not None and sw.save_this:
        sw.summ_flow('0_inputs/flow_fw', flow_fw, clip=clip)
        sw.summ_oned('0_inputs/flow_err', flow_err_clamp, norm=False, frame_id=flow_err.mean().item())
        sw.summ_oned('0_inputs/flow_err_unexpected', flow_err_unexpected, norm=False, frame_id=mean_flow_err_unexpected.item())
        sw.summ_oned('0_inputs/occ_fw', occ_fw, norm=False)
        
    dat = {
        'flow_fw': flow_fw.squeeze(0),
        'occ_fw': occ_fw.squeeze(0),
        'flow_g': flow_g.squeeze(0),
        'rgbs': rgbs.squeeze(0),
        'color_consty_map': color_consty_map,
        'rgb0': rgb0,
        'rgb1': rgb1,
    } 
    return dat

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

class SintelDatasetTrain(torch.utils.data.Dataset):
    def __init__(self,
                 dataset_location='/data/sintel',
                 dset='t',
                 use_augs=False,
                 test_mode=False,
                 crop_size=(368, 496),
                 scene_name=None,
                 scene_num_frames=None,
                #  return_feat=False
                 return_feat=True,
                 split='clean'
    ):
        # self.set_seed(SEED)
        self.split = split
        self.return_feat = return_feat
        self.seed = 0
        rgb_root = os.path.join(dataset_location, f'training/{split}')
        print('rgbroot', rgb_root)
        # rgb_root = os.path.join(dataset_location, 'training/final')
        flow_root = os.path.join(dataset_location, 'training/flow')
        
        self.h_flip_prob = 0.5
        self.v_flip_prob = 0.1
        self.use_augs = use_augs
        self.crop_size = crop_size
        self.test_mode = test_mode
        # photometric augmentation params
        self.photo_aug = ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.5/3.14)
        self.asymmetric_color_aug_prob = 0.2

        folder_names = [folder.split('/')[-1] for folder in glob.glob(os.path.join(rgb_root, "*"))]
        folder_names = sorted(folder_names)
        print(folder_names)
        if scene_name is not None:
            assert scene_name in folder_names
            print(f"Using scene {scene_name}")
            folder_names = [scene_name]
        self.rgb0_paths = []
        self.rgb1_paths = []
        self.flow_paths = []
        self.dat_list = []
        for ii, folder_name in enumerate(folder_names):
            cur_rgb_path = os.path.join(rgb_root, folder_name)
            cur_flow_path = os.path.join(flow_root, folder_name)
            
            img_names = [folder.split('/')[-1].split('.')[0] for folder in glob.glob(os.path.join(cur_rgb_path, "*"))]
            img_names = sorted(img_names)
            S_here = len(img_names)

            for si in range(0, S_here-1):
                rgb0_path = os.path.join(cur_rgb_path, '%s.png' % img_names[si])
                rgb1_path = os.path.join(cur_rgb_path, '%s.png' % img_names[si+1])
                flow_path = os.path.join(cur_flow_path, '%s.flo' % img_names[si])

                self.rgb0_paths.append(rgb0_path)
                self.rgb1_paths.append(rgb1_path)
                self.flow_paths.append(flow_path)
        print('found %d samples in %s' % (len(self.rgb0_paths), dataset_location))
    
    def set_seed(self, seed = 10):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    def prep_data(self, raft_model, device="cuda", sw=None, scenename=None, ext_name=""):
        pseudo_label_folder = "./raft_pseudo_labels"
        if not os.path.exists(pseudo_label_folder):
            os.makedirs(pseudo_label_folder)
        savename = f"{scenename}_colorthresh{COLOR_THRESH}{ext_name}.pkl"
        # append folder name to savename
        savename = os.path.join(pseudo_label_folder, savename)
        if self.split == "final":
            savename = f"final_{scenename}_colorthresh{COLOR_THRESH}{ext_name}.pkl"
        if os.path.isfile(savename):
            with open(savename, 'rb') as handle:
                self.dat_list = pickle.load(handle)
            return
        for index in range(len(self.rgb0_paths)):
            cur_rgb0_path = self.rgb0_paths[index]
            cur_rgb1_path = self.rgb1_paths[index]
            cur_flow_path = self.flow_paths[index]
            rgb0 = readImage(cur_rgb0_path)
            rgb1 = readImage(cur_rgb1_path)
            flow = readFlow(cur_flow_path)
            rgb0 = torch.from_numpy(rgb0).permute(2,0,1) # 3,H,W
            rgb1 = torch.from_numpy(rgb1).permute(2,0,1) # 3,H,W
            flow = torch.from_numpy(flow).permute(2,0,1) # 2,H,W
            sample = {
            'rgb0': rgb0,
            'rgb1': rgb1,
            'flow': flow
            }
            dat = run_prep(raft_model, sample, device, sw=sw, name_ext=index, return_feat=self.return_feat)
            dat['t'] = index
            for k in dat:
                if hasattr(dat[k], 'device'):
                    dat[k] = dat[k].cpu()
            vis = 1.0 - dat["occ_fw"].float()
            consistent_data = torch.sum(vis)
            total = vis.numel()
            consistent_ratio = consistent_data / total
            print("Consistent Ratio", consistent_ratio)
            # sw.summ_scalar('consistent_ratio', consistent_ratio)
            self.dat_list.append(dat)

        with open(savename, 'wb') as handle:
            pickle.dump(self.dat_list, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def __getitem__(self, index):
        dat = self.dat_list[index]
        dat = self.crop_aug(dat)
        return dat, True

    def just_crop(self, dat):
        self.set_seed(self.seed)
        self.seed += 1
        rgbs = dat['rgbs'].permute(0, 2, 3, 1)
        flow = dat['flow_g'].permute(1, 2, 0)
        dat['flow_fw'] = dat['flow_fw'].permute(1, 2, 0)
        dat['occ_fw'] = dat['occ_fw'].permute(1, 2, 0)
        dat['color_consty_map'] = dat['color_consty_map'].permute(1, 2, 0)
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

        rgbs = rgbs[:, y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
        # rgbs = [rgb[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]] for rgb in rgbs]
        flow = flow[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
        flow_fw = dat['flow_fw'][y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
        occ_fw = dat['occ_fw'][y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
        color_consty_map = dat['color_consty_map'][y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
        dat = {
            'flow_fw': flow_fw.permute(2, 0, 1),
            'occ_fw': occ_fw.permute(2, 0, 1),
            'flow_g': flow.permute(2, 0, 1),
            'rgbs': rgbs.permute(0, 3, 1, 2),
            'color_consty_map': color_consty_map.permute(2, 0, 1),
            'yrange': (y0, y0+self.crop_size[0]),
            'xrange': (x0, x0+self.crop_size[1]),
            't': dat['t']
        }
    
        dat['rgb0'] = dat['rgbs'][0]
        dat['rgb1'] = dat['rgbs'][1]
        del dat['rgbs']
        return dat
    

    def color_transform(self, img1, img2):
        """ Photometric augmentation """

        # asymmetric
        if np.random.rand() < self.asymmetric_color_aug_prob:
            img1 = np.array(self.photo_aug(Image.fromarray(img1)), dtype=np.uint8)
            img2 = np.array(self.photo_aug(Image.fromarray(img2)), dtype=np.uint8)

        # symmetric
        else:
            image_stack = np.concatenate([img1, img2], axis=0)
            image_stack = np.array(self.photo_aug(Image.fromarray(image_stack)), dtype=np.uint8)
            img1, img2 = np.split(image_stack, 2, axis=0)

        return img1, img2
    
    def crop_aug(self, dat):
        self.set_seed(self.seed)
        self.seed += 1
        rgb0 = dat['rgb0']
        rgb1 = dat['rgb1']
        # ******************************************************************************************* #
        # Color Jitter
        rgb0 = rgb0.squeeze(0).permute(1,2,0).numpy().astype(np.uint8)
        rgb1 = rgb1.squeeze(0).permute(1,2,0).numpy().astype(np.uint8)
        # print(np.max(rgb0), np.max(rgb1))
        rgb0, rgb1 = self.color_transform(rgb0, rgb1)
        rgb0 = torch.tensor(np.array(rgb0)).float().permute(2, 0, 1).unsqueeze(0)
        rgb1 = torch.tensor(np.array(rgb1)).float().permute(2, 0, 1).unsqueeze(0)
        rgb0 = utils.improc.preprocess_color(rgb0)
        rgb1 = utils.improc.preprocess_color(rgb1)      
        rgbs = torch.stack([rgb0, rgb1], dim=1).squeeze() # B,S,C,H,W
        rgbs = rgbs.permute(0, 2, 3, 1)

        flow = dat['flow_g'].permute(1, 2, 0)
        dat['flow_fw'] = dat['flow_fw'].permute(1, 2, 0)
        dat['occ_fw'] = dat['occ_fw'].permute(1, 2, 0)
        dat['color_consty_map'] = dat['color_consty_map'].permute(1, 2, 0)
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

        rgbs = rgbs[:, y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
        flow = flow[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
        flow_fw = dat['flow_fw'][y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
        occ_fw = dat['occ_fw'][y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
        color_consty_map = dat['color_consty_map'][y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
        
        # ******************************************************************************************* #
        # Flip
        rgbs = rgbs.numpy()
        flow = flow.numpy()
        flow_fw = flow_fw.numpy()
        occ_fw = occ_fw.numpy()
        color_consty_map = color_consty_map.numpy()

        if np.random.rand() < self.h_flip_prob: # h-flip
            rgbs = rgbs[:, :, ::-1]
            flow = flow[:, ::-1] * [-1.0, 1.0]
            flow_fw = flow_fw[:, ::-1] * [-1.0, 1.0]
            occ_fw = occ_fw[:, ::-1]
            color_consty_map = color_consty_map[:, ::-1]
                
        if np.random.rand() < self.v_flip_prob: # v-flip
            rgbs = rgbs[:, ::-1, :]
            flow = flow[::-1, :] * [1.0, -1.0]
            flow_fw = flow_fw[::-1, :] * [1.0, -1.0]
            occ_fw = occ_fw[::-1, :]
            color_consty_map = color_consty_map[::-1, :]
        
        rgbs = torch.tensor(rgbs.copy())
        flow = torch.tensor(flow.copy())
        flow_fw = torch.tensor(flow_fw.copy())
        occ_fw = torch.tensor(occ_fw.copy())
        color_consty_map = torch.tensor(color_consty_map.copy())

        dat = {
            'flow_fw': flow_fw.permute(2, 0, 1),
            'occ_fw': occ_fw.permute(2, 0, 1),
            'flow_g': flow.permute(2, 0, 1),
            'rgbs': rgbs.permute(0, 3, 1, 2),
            'color_consty_map': color_consty_map.permute(2, 0, 1),
            'yrange': (y0, y0+self.crop_size[0]),
            'xrange': (x0, x0+self.crop_size[1]),
            't': dat['t']
        }
    
        dat['rgb0'] = dat['rgbs'][0]
        dat['rgb1'] = dat['rgbs'][1]
        del dat['rgbs']
        return dat
    
    def __len__(self):
        return len(self.rgb0_paths)


