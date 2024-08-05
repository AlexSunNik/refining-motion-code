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
# *************************************************************************** #
from nets.pips import Pips
from nets.raftnet import Raftnet
import saverloader
from datasets.horsedataset import HorseDataset
from torch.utils.data import Dataset, DataLoader
from PIL import Image


def run_raft(raft, rgbs, points, sw):
    rgbs = rgbs.cuda().float() # B, S, C, H, W
    N = len(points)
    B, S, C, H, W = rgbs.shape
    xy0 = torch.stack([torch.tensor(points[:, 0, 0]), torch.tensor(points[:, 0, 1])], dim = -1)
    xy0 = xy0.view(1, -1, 2)

    prep_rgbs = utils.improc.preprocess_color(rgbs)

    flows_e = []
    for s in range(S-1):
        rgb0 = prep_rgbs[:,s]
        rgb1 = prep_rgbs[:,s+1]
        flow, _ = raft(rgb0, rgb1, iters=32)
        flows_e.append(flow)
    flows_e = torch.stack(flows_e, dim=1) # B, S-1, 2, H, W

    coords = []
    # coord0 = trajs_g[:,0] # B, N, 2
    coord0 = xy0.cuda()
    coords.append(coord0)
    coord = coord0.clone()
    for s in range(S-1):
        delta = utils.samp.bilinear_sample2d(
            flows_e[:,s], coord[:,:,0], coord[:,:,1]).permute(0,2,1) # B, N, 2, forward flow at the discrete points
        coord = coord + delta
        coords.append(coord)
    trajs_e = torch.stack(coords, dim=1) # B, S, N, 2
    
    return trajs_e


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


random.seed(125)
np.random.seed(125)
torch.manual_seed(125)



def run_vis(d, device, dname, sw=None):
    total_loss = torch.tensor(0.0, requires_grad=True).to(device)
    metrics = {}
    
    rgbs = d['rgbs'].float() # B, S, C, H, W
    # print(torch.max(rgbs), torch.min(rgbs))
    masks_g = d['masks_g'].float() # B, S, 1, H, W
    xys_g = d['xys_g'].float() # B, S, 2
    whs_g = d['whs_g'].float() # B, S, 2
    vis_g = d['vis_g'].float() # B, S

    B, S, C, H, W = rgbs.shape
    assert(C==3)

    boxlists_g = utils.geom.get_boxlist_from_centroid_and_size(
        xys_g[:,:,1], xys_g[:,:,0], whs_g[:,:,1], whs_g[:,:,0]) # B,S,4

    if sw is not None and sw.save_this:
        prep_rgbs = utils.improc.preprocess_color(rgbs)
        prep_grays = prep_rgbs.mean(dim=2, keepdim=True).repeat(1,1,3,1,1)
        
        sw.summ_traj2ds_on_rgb('%s_0_inputs/trajs_on_rgbs' % dname, xys_g[0:1].unsqueeze(2), utils.improc.preprocess_color(rgbs[0:1].mean(dim=1)), cmap='winter', linewidth=2)

        rgb_g_vis = []
        mask_g_vis = []
        for si in range(0,S,1):
            rgb_g_vis.append(sw.summ_boxlist2d('', utils.improc.preprocess_color(rgbs[0:1,si]), boxlists_g[0:1,si:si+1], frame_id=si, only_return=True))
            mask_vis = sw.summ_oned('', masks_g[0:1,si], norm=False, only_return=True)
            mask_g_vis.append(sw.summ_boxlist2d('', utils.improc.preprocess_color(mask_vis), boxlists_g[0:1,si:si+1], frame_id=vis_g[0,si], only_return=True))
        joint_vis_g = [torch.cat([rgb, mask], dim=-1) for (rgb,mask) in zip(rgb_g_vis, mask_g_vis)]
        sw.summ_rgbs('%s_0_inputs/boxes_on_rgbs_and_masks_g' % dname, joint_vis_g)

    return None



def run_model(model, rgbs, points, sw, video_name="test"):
    rgbs = rgbs.cuda().float() # B, S, C, H, W
    N = len(points)
    B, S, C, H, W = rgbs.shape
    
    x = torch.ones((1, N), device=torch.device('cuda')) * 320.0
    y = torch.ones((1, N), device=torch.device('cuda')) * 180.0

    xy0 = torch.stack([torch.tensor(points[:, 0, 0]), torch.tensor(points[:, 0, 1])], dim = -1)
    xy0 = xy0.view(1, -1, 2)
    _, S, C, H, W = rgbs.shape

    trajs_e = torch.zeros((B, S, N, 2), dtype=torch.float32, device='cuda')
    for n in range(N):
        cur_frame = 0
        done = False
        traj_e = torch.zeros((B, S, 2), dtype=torch.float32, device='cuda')
        traj_e[:,0] = xy0[:,n] # B, 1, 2  # set first position 
        feat_init = None
        while not done:
            end_frame = cur_frame + 8

            rgb_seq = rgbs[:,cur_frame:end_frame]
            # 1, 8, 3, H, W
            S_local = rgb_seq.shape[1]
            # 1, 8-S_local, 3, H, W
            rgb_seq = torch.cat([rgb_seq, rgb_seq[:,-1].unsqueeze(1).repeat(1,8-S_local,1,1,1)], dim=1)

            outs = model(traj_e[:,cur_frame].reshape(1, -1, 2), rgb_seq, iters=6, feat_init=feat_init, return_feat=True)
            preds = outs[0]
            vis = outs[2] # B, S, 1
            feat_init = outs[3]
            
            vis = torch.sigmoid(vis) # visibility confidence
            xys = preds[-1].reshape(1, 8, 2)
            traj_e[:,cur_frame:end_frame] = xys[:,:S_local]

            found_skip = False
            thr = 0.9
            si_last = 8-1 # last frame we are willing to take
            si_earliest = 1 # earliest frame we are willing to take
            si = si_last
            while not found_skip:
                if vis[0,si] > thr:
                    found_skip = True
                else:
                    si -= 1
                if si == si_earliest:
                    # print('decreasing thresh')
                    thr -= 0.02
                    si = si_last
            # print('found skip at frame %d, where we have' % si, vis[0,si].detach().item())

            cur_frame = cur_frame + si

            if cur_frame >= S:
                done = True
        trajs_e[:,:,n] = traj_e
    
    # print(trajs_e.shape, points.shape)
    pad = 50
    rgbs = F.pad(rgbs.reshape(B*S, 3, H, W), (pad, pad, pad, pad), 'constant', 0).reshape(B, S, 3, H+pad*2, W+pad*2)
    trajs_e = trajs_e + pad
    points = points.unsqueeze(2)
    # points += pad
    # if sw is not None and sw.save_this:
    #     linewidth = 2

    #     kp_vis = sw.summ_traj2ds_on_rgbs('video_%d/kp_%d_trajs_e_on_rgbs' % (sw.global_step, n), points[0:1,:,:], gray_rgbs[0:1,:S], cmap='spring', linewidth=linewidth)

    #     # write to disk, in case that's more convenient
    #     kp_list = list(kp_vis.unbind(1))
    #     kp_list = [kp[0].permute(1,2,0).cpu().numpy() for kp in kp_list]
    #     kp_list = [Image.fromarray(kp) for kp in kp_list]
    #     # out_fn = './chain_out_%d.gif' % sw.global_step
    #     # out_fn = './chain_out_%d.gif' % 0
    #     # out_fn = f'baseline_viz/pips_demo_baseline_{video_name}.gif'
    #     # out_fn = f'{video_name}_tri.gif'
    #     out_fn = f'horse10_pips_iter{video_name}_gt.gif'
    #     kp_list[0].save(out_fn, save_all=True, append_images=kp_list[1:])
    #     print('saved %s' % out_fn)
            
    #     sw.summ_traj2ds_on_rgb('outputs/trajs_e_on_rgb', points[0:1], prep_rgbs[0:1,0], cmap='spring')
    #     sw.summ_traj2ds_on_rgb('outputs/trajs_e_on_rgb2', points[0:1], torch.mean(prep_rgbs[0:1], dim=1), cmap='spring')
        
    # points -= pad
    return trajs_e-pad



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

    parser = argparse.ArgumentParser()
    parser.add_argument('--horse10_path', type=str, default='/orion/group/horse10/horse10/')
    # checkpoint
    parser.add_argument('--checkpoint_path', type=str, default='/orion/u/xs15/raft-sintel.pth')
    args = parser.parse_args()

    exp_name = "horse10_raft_eval"

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
    # use_augs = False

    global_step = 0

    # Horse dataset
    # *************************************************************************** #
    
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
    checkpoint_path = args.checkpoint_path
    raft = Raftnet(ckpt_name=checkpoint_path).cuda()
    raft.eval()
    # exit()

    n_pool = 10000
    ate_all_pool_t = utils.misc.SimplePool(n_pool, version='np')
    ate_vis_pool_t = utils.misc.SimplePool(n_pool, version='np')
    all_errors = []

    for idx, d in enumerate(horse_dataloader):
        if idx == 1500:
            break
        global_step += 1

        iter_start_time = time.time()
        iter_read_time = 0.0
        
        # read sample
        read_start_time = time.time()
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
        # print(rgbs.shape)
        # print(torch.max(rgbs), torch.min(rgbs))
        xys_g = d['xys_g'].float() # B, S, 2
        # xys_g[:,:,0], xys_g[:,:,1] = xys_g[:,:,1], xys_g[:,:,0]

        vis_g = d['vis_g'].float() # B, S
        xys_valid = d['xys_valid'].float() # B, S
        vis_valid = d['vis_valid'].float() # B, S
        with torch.no_grad():
            # trajs_e = run_model(model, rgbs, xys_g, sw_t, video_name=idx)
            trajs_e = run_raft(raft, rgbs, xys_g, sw_t)
        trajs_e = trajs_e.cpu()
        xys_g = xys_g.unsqueeze(2)
        # print(trajs_e)
        # print(xys_g)
        # print(xys_g.shape, trajs_e.shape)
        # exit()
        ate = torch.norm(trajs_e - xys_g, dim=-1) # B, S, N
        ate_all = torch.mean(ate)
        # print(ate_all)
        # ate_vis = torch.mean(ate[:, vis_g])
        ate_vis = ate_all
        all_errors.append(ate_all)
        # ate_all = utils.basic.reduce_masked_mean(ate, xys_valid)
        # ate_vis = utils.basic.reduce_masked_mean(ate, xys_valid*vis_g)
        # print(ate_all, ate_vis)
        if ate_all > 0:
            ate_all_pool_t.update([ate_all])
        if ate_vis > 0:
            ate_vis_pool_t.update([ate_vis])

        # exit()
    
    print("final metrics")
    print("ate_all", ate_all_pool_t.mean())
    print("ate_vis", ate_vis_pool_t.mean())
    mid_point = len(all_errors) // 2
    print("median", sorted(all_errors)[mid_point])
    writer_t.close()
            

if __name__ == '__main__':
    Fire(main)