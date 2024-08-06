import time
import numpy as np
import timeit
import saverloader
from nets.raftnet import Raftnet
from nets.pips import Pips
import random
from utils.basic import print_, print_stats
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tensorboardX import SummaryWriter
import torch.nn.functional as F
from crohddataset import CrohdDataset
import utils.basic
import utils.improc
import utils.test
from fire import Fire
from cycle_consistency import *
from PIL import Image
import imageio.v2 as imageio
import pickle 
import argparse
import os 

device = 'cuda'
random.seed(125)
np.random.seed(125)

def run_model_8(model, xy, rgbs):
    outs = model(xy, rgbs, iters=6)
    # S, N, 2
    preds = outs[0][-1].squeeze()
    vis = outs[2] # B, S, 1
    vis = torch.sigmoid(vis) # visibility confidence
    occluded = (1-vis)>.5

    return preds, occluded

def prep_sample(sample, N_max, S_stride=3, req_occlusion=False):
    rgbs = sample['rgbs'].permute(0,1,4,2,S_stride).float()[:,::S_stride] # (1, S, C, H, W) in 0-255
    boxlist = sample['boxlist'][0].float()[::S_stride] # (S, N, 4), N = n heads
    xylist = sample['xylist'][0].float()[::S_stride] # (S, N, 2)
    scorelist = sample['scorelist'][0].float()[::S_stride] # (S, N)
    vislist = sample['vislist'][0].float()[::S_stride] # (S, N)
    
    S, N, _ = xylist.shape

    # collect valid heads
    scorelist_sum = scorelist.sum(0) # (N)
    seq_present = scorelist_sum == S
    motion = torch.sqrt(torch.sum((xylist[1:] - xylist[:1])**2, dim=2)).sum(0) # (N)
    seq_moving = motion > 150
    seq_vis_init = vislist[:2].sum(0) == 2
    seq_occlusion = vislist.sum(0) < 8
    seq_visible = vislist.sum(0) == 8
    if req_occlusion:
        seq_valid = seq_present * seq_vis_init * seq_moving * seq_occlusion
    else:
        seq_valid = seq_present * seq_vis_init * seq_moving * seq_visible
    if seq_valid.sum() == 0:
        return None, True
    
    kp_xys = xylist[:, seq_valid> 0].unsqueeze(0)
    vis = vislist[:, seq_valid > 0].unsqueeze(0)

    N = kp_xys.shape[2]
    if N > N_max:
        kp_xys = kp_xys[:,:,:N_max]
        vis = vis[:,:,:N_max]
        
    d = {
        'rgbs': rgbs, # B, S, C, H, W
        'trajs_g': kp_xys, # B, S, 2
        'vis_g': vis, # B, S,
        "subfolder": sample['subfolder'],
        "start_frame": sample['start_frame']
    }
    return d, False


def run_pips(model, d, sw, CycleConsty=None, name_ext="", frame_start=None, frame_end=None):
    rgbs = d['rgbs'].cuda()
    trajs_g = d['trajs_g'].cuda() # B,S,N,2
    vis_g = d['vis_g'].cuda() # B,S,N
    subfolder = d['subfolder'][0]
    start_frame = int(d['start_frame'][0])
    if frame_start is not None and frame_end is not None:
        if start_frame < frame_start or start_frame >= frame_end:
            return
    print(subfolder, start_frame)
    save_name = f'data/{subfolder}_{start_frame}.pkl'
    if os.path.isfile(save_name):
        return
    valids = torch.ones_like(vis_g) # B,S,N

    B, S, C, H, W = rgbs.shape
    B, S1, N, D = trajs_g.shape
    rgbs_ = rgbs.reshape(B*S, C, H, W)
    H_, W_ = 768, 1280
    sy = H_/H
    sx = W_/W
    rgbs_ = F.interpolate(rgbs_, (H_, W_), mode='bilinear')
    H, W = H_, W_
    rgbs = rgbs_.reshape(B, S, C, H, W)
    trajs_g[:,:,:,0] *= sx
    trajs_g[:,:,:,1] *= sy

    _, S, C, H, W = rgbs.shape

    query_points = trajs_g[:,0]
    sampled_queries = []
    window_size = 24
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
    xys = torch.cat(sampled_queries, dim = 0).unsqueeze(0)

    CHUNK_N = 512
    loops = int(xys.shape[1]) // CHUNK_N
    print("Loop over", loops)
    all_data = []
    for idx in range(loops):
        draw = (idx % 10 == 0)
        xy = xys[:, idx*CHUNK_N:(idx+1)*CHUNK_N]
        # get consistency
        xy = xy.cuda()
        preds, occluded = run_model_8(model, xy, rgbs)
        # print(preds.shape)
        new_query = preds[-1].unsqueeze(0)
        bkw_preds, bkw_occluded = run_model_8(model, new_query, torch.flip(rgbs, dims=[1]))
        # print(bkw_preds.shape)
        inconsistency = CycleConsty.get_inconsty2(preds, bkw_preds, threshs=[1.5])
        good_idx = CycleConsty.decide_good_tracks2(None, inconsistency, None, None)
        # print(good_idx.shape)
        print(f"Number of saved tracks: {torch.sum(torch.tensor(good_idx))} out of {len(good_idx)}")

        good_traj = preds[:, good_idx]
        all_data.append(good_traj)
        # Vis
        if draw:
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
            name = name_ext
            seed = start = idx
            kp_vis = sw.summ_traj2ds_on_rgbs(f'{subfolder}_{start_frame}_seed{seed}_start{start}_trajs_e_on_rgbs', trajs_vis, gray_rgbs, cmap='spring', linewidth=linewidth)
            # write to disk, in case that's more convenient
            kp_list = list(kp_vis.unbind(1))
            kp_list = [kp[0].permute(1,2,0).cpu().numpy() for kp in kp_list]
            kp_list = [Image.fromarray(kp) for kp in kp_list]
            out_fn = f'./sample_gifs_crohd/{subfolder}_{start_frame}_{name}_seed{idx}_start{idx}.gif'
            kp_list[0].save(out_fn, save_all=True, append_images=kp_list[1:])
            print('saved %s' % out_fn)
    

    chunked_data = []
    all_data = torch.cat(all_data, dim = 1)
    TRAIN_N = 128

    cur_N = all_data.shape[1]
    start = 0
    while True:
        end = start + TRAIN_N
        if end > cur_N:
            break
        chunked_data.append((subfolder, start_frame, all_data[:, start:end]))
        start = end
    print("number of good tracks", len(chunked_data))
    print(len(chunked_data))

    folder_name = f'data'
    # Check if the folder exists
    if not os.path.exists(folder_name):
        # Create the folder if it doesn't exist
        os.makedirs(folder_name)
    save_name = f'data/{subfolder}_{start_frame}.pkl'
    with open(save_name, 'wb') as handle:
        pickle.dump(chunked_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return


def main(
        exp_name='crohd', 
        B=1,
        S=8,
        N=16,
        modeltype='pips',
        init_dir='../reference_model',
        # req_occlusion=True,
        req_occlusion=False,
        stride=4,
        log_dir='logs_test_on_crohd',
        max_iters=0, # auto-select based on dataset
        log_freq=100,
        shuffle=False,
        subset='all',
        use_augs=False,
):
    # the idea in this file is to evaluate on head tracking in croHD
    parser = argparse.ArgumentParser("TrainSingleVideo")
    # ['HT21-01', 'HT21-02', 'HT21-03', 'HT21-04']
    parser.add_argument("--dset_split", help="video_name to train on", default=0, type=int)
    # add dataset path
    parser.add_argument("--dataset_path", help="path to the crohd dataset", default='/orion/group/head_tracking')
    args = parser.parse_args()
    dset_split = args.dset_split
    dataset_path = args.dataset_path
    
    frame_start = None
    frame_end = None
    
    # for parallelization
    if dset_split == 0:
        video_name = "HT21-01"
        
    elif dset_split == 1:
        video_name = "HT21-02"
        frame_start = 0
        frame_end = 1000

    elif dset_split == 2:
        video_name = "HT21-02"
        frame_start = 1000
        frame_end = 2000

    elif dset_split == 3:
        video_name = "HT21-02"
        frame_start = 2000
        frame_end = 3315

    elif dset_split == 4:
        video_name = "HT21-03"

    elif dset_split == 5:
        video_name = "HT21-04"


    assert(modeltype=='pips' or modeltype=='raft' or modeltype=='dino')
    
    S_stride = 3 # subsample the frames this much

    model_name = "%d_%d_%d_%s" % (B, S, N, modeltype)
    if req_occlusion:
        model_name += "_occ"
    else:
        model_name += "_vis"
    model_name += "_%s" % exp_name
    import datetime
    model_date = datetime.datetime.now().strftime('%H:%M:%S')
    model_name = model_name + '_' + model_date
    print('model_name', model_name)

    writer_t = SummaryWriter(log_dir + '/' + model_name + '/t', max_queue=10, flush_secs=60)

    dataset = CrohdDataset(dataset_path=dataset_path, seqlen=S*S_stride, video_name=video_name, stop=frame_end, start=frame_start)
    test_dataloader = DataLoader(
        dataset,
        batch_size=B,
        shuffle=False,
        num_workers=12)
    test_iterloader = iter(test_dataloader)

    global_step = 0
    
    if modeltype=='pips':
        model = Pips(S=S, stride=stride).cuda()
        _ = saverloader.load(init_dir, model)
        model.eval()
    elif modeltype=='raft':
        model = Raftnet(ckpt_name='../RAFT/models/raft-things.pth').cuda()
        model.eval()
    elif modeltype=='dino':
        patch_size = 8
        model = torch.hub.load('facebookresearch/dino:main', 'dino_vits%d' % patch_size).cuda()
        model.eval()
    else:
        assert(False) # need to choose a valid modeltype
    
    if max_iters==0:
        max_iters = len(test_dataloader)
    print('setting max_iters', max_iters)
    CycleConsty = CycleConsistency(True, True)
    while global_step < max_iters:

        global_step += 1

        sw_t = utils.improc.Summ_writer(
            writer=writer_t,
            global_step=global_step,
            log_freq=log_freq,
            fps=5,
            scalar_freq=int(log_freq/2),
            just_gif=True)

        returned_early = True
        while returned_early:
            try:
                sample = next(test_iterloader)
            except StopIteration:
                test_iterloader = iter(test_dataloader)
                sample = next(test_iterloader)
            sample, returned_early = prep_sample(sample, N, S_stride, req_occlusion)

            
        with torch.no_grad():
            run_pips(model, sample, sw_t, CycleConsty, frame_start=frame_start, frame_end=frame_end)

    writer_t.close()

if __name__ == '__main__':
    Fire(main)

