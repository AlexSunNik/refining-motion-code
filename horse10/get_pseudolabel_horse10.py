import time
import argparse
import numpy as np
import timeit
import utils.improc
import utils.geom
import random
from utils.basic import print_, print_stats
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
import torch.nn.functional as F
from fire import Fire
import sys

from nets.pips import Pips
import saverloader
from datasets.horsedataset import HorseDataset
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from cycle_consistency import *
import pickle
import os
# *************************************************************************** #

TRAIN_N = 128
# Should already generate enough data for finetuning
EARLY_STOP = 1500
# Define the window size and threshold for the cycle consistency
WINDOW_SIZE = 40
THRESH = 3
# *************************************************************************** #

def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


random.seed(125)
np.random.seed(125)
torch.manual_seed(125)
# *************************************************************************** #

def run_model_8(model, xy, rgbs):
    outs = model(xy, rgbs, iters=6)
    # S, N, 2
    preds = outs[0][-1].squeeze()
    vis = outs[2] # B, S, 1
    vis = torch.sigmoid(vis) # visibility confidence
    occluded = (1-vis)>.5

    return preds, occluded


def run_model(model, rgbs, points, sw, video_name="test", CycleConsty=None, save_video_name=None, all_idx=None):
    rgbs = rgbs.cuda().float() # B, S, C, H, W
    N = len(points)
    B, S, C, H, W = rgbs.shape
    
    x = torch.ones((1, N), device=torch.device('cuda')) * 320.0
    y = torch.ones((1, N), device=torch.device('cuda')) * 180.0

    xy0 = torch.stack([torch.tensor(points[:, 0, 0]), torch.tensor(points[:, 0, 1])], dim = -1)
    xy0 = xy0.view(1, -1, 2)
    _, S, C, H, W = rgbs.shape

    query_points = xy0
    sampled_queries = []
    window_size = WINDOW_SIZE
    thresh = THRESH

    save_name = f'data{thresh}/{video_name}_{int(all_idx[0])}.pkl'
    if os.path.isfile(save_name):
        return
    
    for i in range(query_points.shape[1]):
        point = query_points[0, i]
        # get windowed queries
        x = point[0]
        y = point[1]
        y = int(y)
        x = int(x)
        ys = torch.tensor(range(y-window_size//2, y+window_size//2))
        xs = torch.tensor(range(x-window_size//2, x+window_size//2))
        grid_x, grid_y = torch.meshgrid(xs, ys, indexing='ij')
        grid_x = grid_x.flatten()
        grid_y = grid_y.flatten()
        xy = torch.stack([grid_x, grid_y], dim=-1)
        xys = torch.split(xy, split_size_or_sections=1, dim=0)
        sampled_queries += xys
    sampled_queries = list(set(sampled_queries))
    all_coords = torch.cat(sampled_queries, dim = 0).unsqueeze(0)
    
    all_data = []
    loops = 1

    for idx in range(loops):
        xy = all_coords
        # get consistency
        xy = xy.cuda()
        preds, occluded = run_model_8(model, xy, rgbs)
        new_query = preds[-1].unsqueeze(0)
        bkw_preds, bkw_occluded = run_model_8(model, new_query, torch.flip(rgbs, dims=[1]))
        inconsistency = CycleConsty.get_inconsty2(preds, bkw_preds, threshs=[thresh])
        good_idx = CycleConsty.decide_good_tracks2(None, inconsistency, None, None)
        print(f"Number of good tracks: {torch.sum(torch.tensor(good_idx))} out of {len(good_idx)}")

        good_traj = preds[:, good_idx]
        all_data.append(good_traj)
        # ************************************************************************************ #
        if video_name % 100 == 0:
            trajs_vis = good_traj
            trajs_vis = trajs_vis.unsqueeze(0)
            writer_t = SummaryWriter("vis_sample_gen" + '/' + "debug" + '/t', max_queue=10, flush_secs=60)
            sw = utils.improc.Summ_writer(
            writer=writer_t,
            global_step=0,
            log_freq=1,
            fps=5,
            scalar_freq=1,
            just_gif=True)

            prep_rgbs = utils.improc.preprocess_color(rgbs)
            gray_rgbs = torch.mean(prep_rgbs, dim=2, keepdim=True).repeat(1, 1, 3, 1, 1)
            linewidth = 2
            name = str(thresh)
            seed = start = idx
            kp_vis = sw.summ_traj2ds_on_rgbs(f'data{thresh}/{video_name}_{int(all_idx[0])}', trajs_vis, gray_rgbs, cmap='spring', linewidth=linewidth)
            # write to disk, in case that's more convenient
            kp_list = list(kp_vis.unbind(1))
            kp_list = [kp[0].permute(1,2,0).cpu().numpy() for kp in kp_list]
            kp_list = [Image.fromarray(kp) for kp in kp_list]
            out_fn = f'./sample_gifs_horse_{thresh}/{video_name}_{int(all_idx[0])}.gif'
            kp_list[0].save(out_fn, save_all=True, append_images=kp_list[1:])
            print('saved %s' % out_fn)
    

    chunked_data = []
    all_data = torch.cat(all_data, dim = 1)
    # ************************************************************************************ #
    cur_N = all_data.shape[1]
    start = 0
    while True:
        end = start + TRAIN_N
        chunked_data.append((video_name, all_idx, all_data[:, start:end]))
        start = end
        if end > cur_N:
            break
    print("number of saved tracks", len(chunked_data))
    print(len(chunked_data))
    save_name = f'data{thresh}/{video_name}_{int(all_idx[0])}.pkl'
    with open(save_name, 'wb') as handle:
        pickle.dump(chunked_data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def main(
        exp_name='debug',
        B=1, # batchsize
        S=32, # seqlen
        rand_frames=False,
        crop_size=(256,448), 
        use_augs=False, # resizing/jittering/color/blur augs
        shuffle=False, # dataset shuffling
        log_dir='./logs_just_vis',
        max_iters=2,
        log_freq=1,
        device_ids=[0],
        dname=None,
):

    # add argparse to read the path of the horse10 dataset as a command line string argument

    parser = argparse.ArgumentParser()
    parser.add_argument('--horse10_path', type=str, default='/orion/group/horse10/horse10/')
    args = parser.parse_args()
      
    exp_name = "get_pseudolabel_horse10"

    assert(crop_size[0] % 64 == 0)
    assert(crop_size[1] % 64 == 0)
    
    ## autogen a descriptive name
    model_name = "%d_%d" % (B, S)
    if rand_frames:
        model_name += "r"
    if use_augs:
        model_name += "_A"
    model_name += "_%s" % exp_name
    import datetime
    model_date = datetime.datetime.now().strftime('%H:%M:%S')
    model_name = model_name + '_' + model_date
    print('model_name', model_name)
    
    writer_t = SummaryWriter(log_dir + '/' + model_name + '/t', max_queue=10, flush_secs=60)
    global_step = 0

    # pointer to the horse dataset
    horse_location = args.horse10_path
    subset = 't'
    num_workers=8
    crop_size=(256,448)

    horse_dataset = HorseDataset(
        dataset_location=horse_location,
        S=32,
        # rand_frames=rand_frames,
        crop_size=crop_size,
        use_augs=False,
        is_training=subset=='t'
    )
    horse_dataloader = DataLoader(
        horse_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        worker_init_fn=worker_init_fn,
        drop_last=True)
    horse_iterloader = iter(horse_dataloader)
    

    # *************************************************************************** #
    model = Pips(stride=4).cuda()
    init_dir = '../reference_model'
    if init_dir:
        _ = saverloader.load(init_dir, model)
    model.eval()

    n_pool = 100
    ate_all_pool_t = utils.misc.SimplePool(n_pool, version='np')
    ate_vis_pool_t = utils.misc.SimplePool(n_pool, version='np')
    CycleConsty = CycleConsistency(True, True)
    
    idx2name_map = {}

    for idx, d in enumerate(horse_dataloader):
        if idx == 1500:
            break
        global_step += 1
        
        # read sample
        sw_t = utils.improc.Summ_writer(
            writer=writer_t,
            global_step=global_step,
            log_freq=log_freq,
            fps=8,
            scalar_freq=int(log_freq/4),
            just_gif=True)
        
        sw_t.save_this = False

        if idx % 100 == 0:
            print(idx, len(horse_dataloader))
            print("ate_all", ate_all_pool_t.mean())
            print("ate_vis", ate_vis_pool_t.mean())
            sw_t.save_this = True

        rgbs = d['rgbs'].float() # B, S, C, H, W
        xys_g = d['xys_g'].float() # B, S, 2
        video_name = d['video_name'][0]
        full_idx = d['full_idx']
        if video_name == "false":
            continue
        idx2name_map[idx] = video_name
        with torch.no_grad():
            run_model(model, rgbs, xys_g, sw_t, video_name=idx, CycleConsty=CycleConsty, save_video_name=video_name, all_idx=full_idx)

    with open("idx2name_map.pkl", 'wb') as f: 
        pickle.dump(idx2name_map, f)
    
    writer_t.close()
            

if __name__ == '__main__':
    Fire(main)