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
import saverloader
from datasets.horsedataset import HorseDataset
from datasets.horsedataset_train import HorseDataset as HorseDatasetTrain

from torch.utils.data import Dataset, DataLoader
from PIL import Image
from cycle_consistency import *
import pickle
import os

def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

def requires_grad(parameters, flag=True):
    for p in parameters:
        p.requires_grad = flag


def validate(horse_dataloader, writer_t, model, train_global_step):
    n_pool = 10000
    ate_all_pool_t = utils.misc.SimplePool(n_pool, version='np')
    ate_vis_pool_t = utils.misc.SimplePool(n_pool, version='np')
    global_step = 0
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
            log_freq=1,
            fps=8,
            scalar_freq=int(1/4),
            just_gif=True)
        
        sw_t.save_this = False

        if idx % 100 == 0:
            print(idx, len(horse_dataloader))
            print("ate_all", ate_all_pool_t.mean())
            print("ate_vis", ate_vis_pool_t.mean())
            sw_t.save_this = True

        rgbs = d['rgbs'].float() # B, S, C, H, W
        xys_g = d['xys_g'].float() # B, S, 2

        vis_g = d['vis_g'].float() # B, S
        xys_valid = d['xys_valid'].float() # B, S
        vis_valid = d['vis_valid'].float() # B, S
        with torch.no_grad():
            trajs_e = run_model(model, rgbs, xys_g, sw_t, video_name=f"{idx}_train{train_global_step}")
        trajs_e = trajs_e.cpu()
        xys_g = xys_g.unsqueeze(2)
        ate = torch.norm(trajs_e - xys_g, dim=-1) # B, S, N
        ate_all = utils.basic.reduce_masked_mean(ate, xys_valid)
        ate_vis = utils.basic.reduce_masked_mean(ate, xys_valid*vis_g)
        # print(ate_all, ate_vis)
        if ate_all > 0:
            ate_all_pool_t.update([ate_all])
        if ate_vis > 0:
            ate_vis_pool_t.update([ate_vis])
    
    print("final metrics")
    print("ate_all", ate_all_pool_t.mean())
    print("ate_vis", ate_vis_pool_t.mean())
    writer_t.close()

def run_pips(model, d):
    rgbs = d['rgbs'].float().cuda()
    trajs_g = d['trajs'].cuda() # B,S,N,2
    visible = torch.ones(trajs_g.shape[0], trajs_g.shape[1], trajs_g.shape[2]).cuda()
    vis_g = visible
    valids = torch.ones_like(visible) # B,S,N

    B, S, C, H, W = rgbs.shape
    B, S1, N, D = trajs_g.shape
    total_loss = 0.0
    preds, preds_anim, vis_e, stats = model(trajs_g[:,0], rgbs, coords_init=None, iters=6, trajs_g=trajs_g, vis_g=visible, valids=valids)
                

    seq_loss, vis_loss, ce_loss = stats
    total_loss += seq_loss.mean()
    total_loss += ce_loss.mean()

    pred_occluded = torch.sigmoid(vis_e) # visibility confidence
    pred_occluded = (1-pred_occluded)>.5
    query_points = np.zeros((trajs_g[:,0].shape[0], trajs_g[:,0].shape[1], trajs_g[:,0].shape[2]+1))
    query_points[:, :, 1:] = trajs_g[:,0].cpu().numpy()

    ate = torch.norm(preds[-1] - trajs_g, dim=-1) # B, S, N
    ate_all = utils.basic.reduce_masked_mean(ate, valids)
    ate_vis = utils.basic.reduce_masked_mean(ate, valids*vis_g)
    
    return total_loss, (ate_all, ate_vis)

def fetch_optimizer(lr, wdecay, epsilon, num_steps, params):
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=wdecay, eps=epsilon)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_steps+200)

    return optimizer, scheduler


random.seed(125)
np.random.seed(125)
torch.manual_seed(125)

def run_model_8(model, xy, rgbs):
    outs = model(xy, rgbs, iters=6)
    # S, N, 2
    preds = outs[0][-1].squeeze()
    vis = outs[2] # B, S, 1
    vis = torch.sigmoid(vis) # visibility confidence
    occluded = (1-vis)>.5

    return preds, occluded

def run_vis(d, device, dname, sw=None):
    total_loss = torch.tensor(0.0, requires_grad=True).to(device)
    metrics = {}
    
    rgbs = d['rgbs'].float() # B, S, C, H, W
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
    rgbs_ = rgbs.reshape(B*S, C, H, W)


    xy0 = torch.stack([torch.tensor(points[:, 0, 0]), torch.tensor(points[:, 0, 1])], dim = -1)
    # print(xy0.shape)
    xy0 = xy0.view(1, -1, 2)
    # print(xy0.shape)
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
                    thr -= 0.02
                    si = si_last

            cur_frame = cur_frame + si

            if cur_frame >= S:
                done = True
        trajs_e[:,:,n] = traj_e
    
    pad = 50
    rgbs = F.pad(rgbs.reshape(B*S, 3, H, W), (pad, pad, pad, pad), 'constant', 0).reshape(B, S, 3, H+pad*2, W+pad*2)
    trajs_e = trajs_e + pad


    prep_rgbs = utils.improc.preprocess_color(rgbs)
    gray_rgbs = torch.mean(prep_rgbs, dim=2, keepdim=True).repeat(1, 1, 3, 1, 1)

    if sw is not None and sw.save_this:
        linewidth = 2

        kp_vis = sw.summ_traj2ds_on_rgbs('video_%d/kp_%d_trajs_e_on_rgbs' % (sw.global_step, n), trajs_e[0:1,:,:], gray_rgbs[0:1,:S], cmap='spring', linewidth=linewidth)

        # write to disk, in case that's more convenient
        kp_list = list(kp_vis.unbind(1))
        kp_list = [kp[0].permute(1,2,0).cpu().numpy() for kp in kp_list]
        kp_list = [Image.fromarray(kp) for kp in kp_list]
        out_fn = f'horse10_pips_iter{video_name}.gif'
        kp_list[0].save(out_fn, save_all=True, append_images=kp_list[1:])
        print('saved %s' % out_fn)
            
        sw.summ_traj2ds_on_rgb('outputs/trajs_e_on_rgb', trajs_e[0:1], prep_rgbs[0:1,0], cmap='spring')
        sw.summ_traj2ds_on_rgb('outputs/trajs_e_on_rgb2', trajs_e[0:1], torch.mean(prep_rgbs[0:1], dim=1), cmap='spring')
    
    # print(points.shape)
    points = points.unsqueeze(2)
    return trajs_e-pad


def main(
        exp_name='debug',
        B=1, # batchsize
        # S=32, # seqlen
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
    # generated pseudo labels path from get_pseudolabel_horse10.py
    parser.add_argument('--pseudo_labels_path', type=str, default='all_horse_data.pkl')
    args = parser.parse_args()


    # the idea in this file is:
    # just load the data and vis
    # to help debug dataloaders
    
    exp_name = "horse10_pips_finetune"

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
    
    pseudo_labels_path = args.pseudo_labels_path
    horse_dataset_train = HorseDatasetTrain(
        dataset_location=horse_location,
        S=32,
        # rand_frames=rand_frames,
        crop_size=crop_size,
        use_augs=False,
        is_training=subset=='t',
        pkl_file = pseudo_labels_path
    )
    train_dataloader = DataLoader(
        horse_dataset_train,
        batch_size=1,
        shuffle=False,
        num_workers=12)
    train_iterloader = iter(train_dataloader)
    # print(horse_dataset_train[0])
    # x = horse_dataset_train.getitem_helper(0)
    grad_acc = 8
    lr = 1e-5
    use_scheduler = True
    # *************************************************************************** #
    # exit()

    n_pool = 100
    ate_all_pool_t = utils.misc.SimplePool(n_pool, version='np')
    ate_vis_pool_t = utils.misc.SimplePool(n_pool, version='np')
    CycleConsty = CycleConsistency(True, True)
    
    model = Pips(stride=4).cuda()
    init_dir = 'reference_model'
    if init_dir:
        _ = saverloader.load(init_dir, model)
    parameters = list(model.parameters())
    requires_grad(parameters, True)
    max_iters = 6000
    if use_scheduler:
        print(f"Learning Rate used: {lr}")
        optimizer, scheduler = fetch_optimizer(lr, 0.0001, 1e-8, max_iters // grad_acc, model.parameters())
        # optimizer, scheduler = fetch_optimizer(lr, max_iters * max_iters_2)
    else:
        optimizer = torch.optim.Adam(parameters, lr=lr, weight_decay=1e-7)
    optimizer.zero_grad()


    VAL_FREQ = 500

    while global_step < max_iters:
        model = model.train()
        read_start_time = time.time()

        sw_t = utils.improc.Summ_writer(
            writer=writer_t,
            global_step=global_step,
            log_freq=log_freq,
            fps=5,
            scalar_freq=int(log_freq/2),
            just_gif=True)

        returned_early = True
        # while returned_early:
        try:
            sample = next(train_iterloader)
        except StopIteration:
            train_iterloader = iter(train_dataloader)
            sample = next(train_iterloader)
            # sample, returned_early = prep_sample(sample, N, S_stride, req_occlusion)

        # if global_step % VAL_FREQ == 0 and global_step != 0:
        if global_step % VAL_FREQ == 0 or global_step == 0:
            print("save model")
            save_name = f"horsemodel_iter{global_step}.pt"
            torch.save(model.state_dict(), save_name)
            model.eval()
            validate(horse_dataloader, writer_t, model, global_step)
            model.train()
            # exit()
        read_time = time.time()-read_start_time
        iter_start_time = time.time()
            
        total_loss, (ate_all, ate_vis) = run_pips(model, sample)
        total_loss.backward()

        sw_t.summ_scalar('total_loss', total_loss)
        sw_t.summ_scalar('ate_all', float(ate_all.item()))
        sw_t.summ_scalar('ate_vis', float(ate_vis.item()))
        
        if global_step % grad_acc == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            if use_scheduler:
                scheduler.step()
            optimizer.zero_grad()
            print(f"Training Loss at {global_step}th iteration: {total_loss}")
        total_loss = None
        sample = None
        global_step += 1


    print("save model")
    save_name = f"horsemodel_final.pt"
    torch.save(model.state_dict(), save_name)
    model.eval()
    validate(horse_dataloader, writer_t, model, global_step)
    model.train()
    # save_name = f"exp{exp}_reqocc{REQ_OCC}_iter{global_step}_ablation2p5.pt"
    # torch.save(model.state_dict(), save_name)
    # model.eval()
    # n_pool = 10000
    # ate_all_pool_v = utils.misc.SimplePool(n_pool, version='np')
    # ate_vis_pool_v = utils.misc.SimplePool(n_pool, version='np')
    # ate_occ_pool_v = utils.misc.SimplePool(n_pool, version='np')
    # print("REQ_OCC: True.")
    # validate(model, test_dataloader, N=16, S_stride=3, req_occlusion=True, ate_all_pool_t=ate_all_pool_v, ate_vis_pool_t=ate_vis_pool_v, ate_occ_pool_t=ate_occ_pool_v)
    # ate_all_pool_v = utils.misc.SimplePool(n_pool, version='np')
    # ate_vis_pool_v = utils.misc.SimplePool(n_pool, version='np')
    # ate_occ_pool_v = utils.misc.SimplePool(n_pool, version='np')
    # print("REQ_OCC: False.")
    # validate(model, test_dataloader, N=16, S_stride=3, req_occlusion=False, ate_all_pool_t=ate_all_pool_v, ate_vis_pool_t=ate_vis_pool_v, ate_occ_pool_t=ate_occ_pool_v)
    # model.train()
    
    writer_t.close()
            

if __name__ == '__main__':
    Fire(main)