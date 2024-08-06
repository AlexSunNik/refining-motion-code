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
import pickle 
import io
from tensorboardX import SummaryWriter
import utils.improc

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)

class CrohdDataset_Train(torch.utils.data.Dataset):
    def __init__(self, seqlen=8, dset='t', pickle_path=None):
        dataset_location = "/orion/u/aharley/head_tracking/HT21"
        label_location = "/orion/u/aharley/head_tracking/HT21Labels"
        subfolders = []
        with open(pickle_path, 'rb') as handle:
            self.final_data = CPU_Unpickler(handle).load()

        if dset == 't':
            dataset_location = os.path.join(dataset_location, "train")
            label_location = os.path.join(label_location, "train")
            subfolders = ['HT21-01', 'HT21-02', 'HT21-03', 'HT21-04']
            # Testing
            # subfolders = ['HT21-01']
            # if video_name is not None:
                # subfolders = [video_name]
        elif dset == 'v':
            dataset_location = os.path.join(dataset_location, "val")
            label_location = os.path.join(label_location, "val")
            subfolders = ['HT21-11', 'HT21-12', 'HT21-13', 'HT21-14', 'HT21-15']
        else:
            raise Exception("unexpceted dset. Choose between t and v.")

        self.dataset_location = dataset_location
        self.label_location = label_location
        self.seqlen = seqlen
        self.subfolders = subfolders
        self.load_fails = [0] * len(self.final_data)

    def __getitem__(self, index):
        gotit = False
        sample, gotit = self.getitem_helper(index)
        return sample
        while not gotit:
            sample, gotit = self.getitem_helper(index)

            if not gotit:
                # save time by not trying this index again
                load_fail = 1
                self.load_fails[index] = load_fail
                # print('warning: updated load_fails (on this worker): %d/%d...' % (np.sum(self.load_fails), len(self.load_fails)))

                while load_fail:
                    index = np.random.randint(0, len(self.load_fails))
                    load_fail = self.load_fails[index]

        return sample
    
    def getitem_helper(self, index):
        gotit = True
        # identify which sample and which starting frame it is
        data = self.final_data[index]
        # print(data[0])
        # print(data[1])
        # print(data[2].shape)
        # exit()
        subfolder = data[0]
        start_frame = data[1]
        # print("start_frame", start_frame)
        tracks = data[2]
        
        # Just for debugging
        # tracks = tracks[:, :2]
        visibles = torch.ones(tracks.shape[0], tracks.shape[1])

        rgbs = []
        S = 8 * 3
        for i in range(S):
            # read image
            image_name = os.path.join(self.dataset_location, subfolder, 'img1', str(start_frame+i+1).zfill(6)+'.jpg')
            rgb = np.array(Image.open(image_name))
            rgbs.append(rgb)


        rgbs = np.stack(rgbs, axis=0)
        # print(rgbs.shape)
        rgbs = rgbs[::3]
        # print(rgbs.shape)
        
        rgbs = torch.tensor(rgbs).permute(0, 3, 1, 2).float()

        # print(rgbs.shape)
        S, C, H, W = rgbs.shape
        S1, N, D = tracks.shape
        # print(rgbs.shape)
        # print(tracks.shape)
        # print(tracks.shape)
        rgbs_ = rgbs.reshape(S, C, H, W)
        # H_, W_ = 768, 1280
        # H_, W_ = 768, 1280
        # H_, W_ = 320, 512
        H_, W_ = 512, 852
        sy = H_/H
        sx = W_/W
        # print(rgbs.shape)
        rgbs_ = F.interpolate(rgbs_, (H_, W_), mode='bilinear')
        H, W = H_, W_
        rgbs = rgbs_.reshape(S, C, H, W)
        tracks *= torch.tensor([W/1280, H/768])
        
        # rgbs = torch.tensor(rgbs).unsqueeze(0)
        # # print(rgbs.shape)
        # trajs_vis = tracks.unsqueeze(0)
        # # print(trajs_vis.shape)
        # # trajs_vis = trajs_vis.unsqueeze(0)
        # # print(trajs_vis.shape)
        # # self.seed * all_length + start
        # writer_t = SummaryWriter("vis_sample_gen" + '/' + "debug" + '/t', max_queue=10, flush_secs=60)
        # sw = utils.improc.Summ_writer(
        # writer=writer_t,
        # global_step=0,
        # log_freq=1,
        # fps=5,
        # scalar_freq=1,
        # just_gif=True)

        # # print(cur_rgbs.shape)
        # prep_rgbs = utils.improc.preprocess_color(rgbs)
        # # print(torch.mean(prep_rgbs, dim=2, keepdim=True).shape)
        # gray_rgbs = torch.mean(prep_rgbs, dim=2, keepdim=True).repeat(1, 1, 3, 1, 1)
        # linewidth = 2
        # name = "dataloader"
        # seed = start = 0
        # kp_vis = sw.summ_traj2ds_on_rgbs(f'{subfolder}_{start_frame}_seed{seed}_start{start}_trajs_e_on_rgbs', trajs_vis, gray_rgbs, cmap='spring', linewidth=linewidth)
        # # write to disk, in case that's more convenient
        # kp_list = list(kp_vis.unbind(1))
        # kp_list = [kp[0].permute(1,2,0).cpu().numpy() for kp in kp_list]
        # kp_list = [Image.fromarray(kp) for kp in kp_list]
        # idx = 0
        # out_fn = f'./sample_gifs_crohd/{subfolder}_{start_frame}_{name}_seed{idx}_start{start_frame}.gif'
        # kp_list[0].save(out_fn, save_all=True, append_images=kp_list[1:])
        # print('saved %s' % out_fn)
        # exit()

        # print(rgbs.shape)
        # exit()
        # tracks[:,:,:,0] *= sx
        # tracks[:,:,:,1] *= sy

        # print(rgbs.shape)
        # print(tracks.shape)
        # print(rgbs.shape)
        # rgbs = torch.tensor(rgbs).permute(0, 3, 1, 2)
        sample = {
            'rgbs': rgbs, # (S, H, W, 3) in 0-255
            'trajs_g': tracks,
            "subfolder": subfolder,
            "start_frame": start_frame,
            'visible': visibles
        }
        return sample, gotit

    def __len__(self):
        return len(self.final_data)
        return sum(self.subfolder_lens)

if __name__ == "__main__":
    B = 1
    S = 8
    shuffle=False
    dataset = HeadTrackingDataset(seqlen=S)

    from torch.utils.data import Dataset, DataLoader
    train_dataloader = DataLoader(
        dataset,
        batch_size=B,
        shuffle=shuffle,
        num_workers=0)

    train_iterloader = iter(train_dataloader)

    sample = next(train_iterloader)

