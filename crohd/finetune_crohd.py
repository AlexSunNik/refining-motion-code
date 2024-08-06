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
from crohddataset_train import CrohdDataset_Train
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
import jax
import jax.numpy as jnp

REQ_OCC = True
quick = False
if quick:
    VAL_FREQ = 5
    STOP = 5
else:
    VAL_FREQ = 500
    STOP = None
device = 'cuda'
random.seed(125)
np.random.seed(125)

def run_model_8(model, xy, rgbs):
    outs = model(xy, rgbs, iters=6)
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
    # print('N', N)
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


def validate(model, test_dataloader, N, S_stride, req_occlusion, ate_all_pool_t=None, ate_vis_pool_t=None, ate_occ_pool_t=None, sw_v=None, name_ext=""):
    max_iters = len(test_dataloader)
    print('setting max_iters', max_iters)
    global_step = 0
    test_iterloader = iter(test_dataloader)
    
    if STOP is not None:
        max_iters = STOP
    # max_iters = 20
    summed_scalars=None
    num_samples=0
    while global_step < max_iters:
        
        read_start_time = time.time()
        global_step += 1
        
        returned_early = True
        while returned_early:
            try:
                sample = next(test_iterloader)
            except StopIteration:
                test_iterloader = iter(test_dataloader)
                sample = next(test_iterloader)
            sample, returned_early = prep_sample(sample, N, S_stride, req_occlusion)

        read_time = time.time()-read_start_time
        iter_start_time = time.time()
            
        with torch.no_grad():
            metrics, scalars = run_pips_validate(model, sample, sw_v)
            scalars = jax.tree_map(lambda x: jnp.sum(x, axis=0), scalars)
            if summed_scalars is None:
                summed_scalars = scalars
            else:
                summed_scalars = jax.tree_map(jnp.add, summed_scalars, scalars)
            num_samples += 1
            mean_scalars = jax.tree_map(lambda x: x / num_samples, summed_scalars)
            
        if metrics['ate_all'] > 0:
            ate_all_pool_t.update([metrics['ate_all']])
        if metrics['ate_vis'] > 0:
            ate_vis_pool_t.update([metrics['ate_vis']])
        if metrics['ate_occ'] > 0:
            ate_occ_pool_t.update([metrics['ate_occ']])

        iter_time = time.time()-iter_start_time
    
    # print(mean_scalars)
    avg_jac, pts_acc, occ_acc = mean_scalars["average_jaccard"], mean_scalars["average_pts_within_thresh"], mean_scalars["occlusion_accuracy"]
    avg_jac, pts_acc, occ_acc =  float(avg_jac), float(pts_acc), float(occ_acc)
    print("avg_jac, pts_acc, occ_acc", avg_jac, pts_acc, occ_acc)
    print('%s; step %06d/%d; rtime %.2f; itime %.2f; ate = %.2f; ate_pooled = %.2f' % (
        "test", global_step, max_iters, read_time, iter_time,
        metrics['ate_all'], ate_all_pool_t.mean()))
        
def run_pips_validate(model, d, sw):

    rgbs = d['rgbs'].cuda()
    trajs_g = d['trajs_g'].cuda() # B,S,N,2
    vis_g = d['vis_g'].cuda() # B,S,N
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

    preds, preds_anim, vis_e, stats = model(trajs_g[:,0], rgbs, iters=6, trajs_g=trajs_g, vis_g=vis_g, valids=valids, sw=sw)
    
    ate = torch.norm(preds[-1] - trajs_g, dim=-1) # B, S, N
    ate_all = utils.basic.reduce_masked_mean(ate, valids)
    ate_vis = utils.basic.reduce_masked_mean(ate, valids*vis_g)
    ate_occ = utils.basic.reduce_masked_mean(ate, valids*(1.0-vis_g))
    
    pred_occluded = torch.sigmoid(vis_e) # visibility confidence
    pred_occluded = (1-pred_occluded)>.5

    query_points = np.zeros((trajs_g[:,0].shape[0], trajs_g[:,0].shape[1], trajs_g[:,0].shape[2]+1))
    query_points[:, :, 1:] = trajs_g[:,0].cpu().numpy()

    scalars = evaluation_datasets.compute_tapvid_metrics(
        query_points=query_points,
        gt_occluded=(1 - vis_g).cpu().permute(0, 2, 1).numpy(),
        gt_tracks=trajs_g.permute(0, 2, 1, 3).cpu().numpy(),
        pred_occluded=pred_occluded.detach().cpu().permute(0, 2, 1).numpy(),     # TODO
        pred_tracks=preds[-1].detach().cpu().permute(0, 2, 1, 3).numpy(),  # TODO
        query_mode='strided',
    )
    metrics = {
        'ate_all': ate_all.item(),
        'ate_vis': ate_vis.item(),
        'ate_occ': ate_occ.item(),
    }

    trajs_e = preds[-1]

    if sw is not None and sw.save_this:
        sw.summ_traj2ds_on_rgbs('inputs_0/orig_trajs_on_rgbs', trajs_g, utils.improc.preprocess_color(rgbs), cmap='winter', linewidth=2)
        
        sw.summ_traj2ds_on_rgbs('outputs/trajs_on_rgbs', trajs_e[0:1], utils.improc.preprocess_color(rgbs[0:1]), cmap='spring', linewidth=2)
        gt_rgb = utils.improc.preprocess_color(sw.summ_traj2ds_on_rgb('inputs_0_all/single_trajs_on_rgb', trajs_g[0:1], torch.mean(utils.improc.preprocess_color(rgbs[0:1]), dim=1), cmap='winter', frame_id=metrics['ate_all'], only_return=True, linewidth=2))
        gt_black = utils.improc.preprocess_color(sw.summ_traj2ds_on_rgb('inputs_0_all/single_trajs_on_rgb', trajs_g[0:1], torch.ones_like(rgbs[0:1,0])*-0.5, cmap='winter', frame_id=metrics['ate_all'], only_return=True, linewidth=2))
        sw.summ_traj2ds_on_rgb('outputs/single_trajs_on_gt_rgb', trajs_e[0:1], gt_rgb[0:1], cmap='spring', linewidth=2)
        sw.summ_traj2ds_on_rgb('outputs/single_trajs_on_gt_black', trajs_e[0:1], gt_black[0:1], cmap='spring', linewidth=2)

    return metrics, scalars

def run_pips(model, d, sw, CycleConsty=None, name_ext=""):
    rgbs = d['rgbs'].float().cuda()
    trajs_g = d['trajs_g'].cuda() # B,S,N,2
    subfolder = d['subfolder'][0]
    start_frame = int(d['start_frame'][0])
    visible = d['visible'].cuda() # B,S,N,2
    vis_g = visible
    # print("sdfd")
    
    # print(subfolder, start_frame)
    # print(visible.shape)
    valids = torch.ones_like(visible) # B,S,N

    B, S, C, H, W = rgbs.shape
    B, S1, N, D = trajs_g.shape
    
    total_loss = 0.0
    preds, preds_anim, vis_e, stats = model(trajs_g[:,0], rgbs, coords_init=None, iters=6, trajs_g=trajs_g, vis_g=visible, valids=valids)

    seq_loss, vis_loss, ce_loss = stats
    total_loss += seq_loss.mean()
    # We do not train the visibility loss
    # total_loss += vis_loss.mean()
    total_loss += ce_loss.mean()

    pred_occluded = torch.sigmoid(vis_e) # visibility confidence
    pred_occluded = (1-pred_occluded)>.5
    query_points = np.zeros((trajs_g[:,0].shape[0], trajs_g[:,0].shape[1], trajs_g[:,0].shape[2]+1))
    query_points[:, :, 1:] = trajs_g[:,0].cpu().numpy()

    ate = torch.norm(preds[-1] - trajs_g, dim=-1) # B, S, N
    ate_all = utils.basic.reduce_masked_mean(ate, valids)
    ate_vis = utils.basic.reduce_masked_mean(ate, valids*vis_g)
    
    scalars = evaluation_datasets.compute_tapvid_metrics(
            query_points=query_points,
            gt_occluded=(1 - vis_g).cpu().permute(0, 2, 1).numpy(),
            gt_tracks=trajs_g.permute(0, 2, 1, 3).cpu().numpy(),
            pred_occluded=pred_occluded.detach().cpu().permute(0, 2, 1).numpy(),     # TODO
            pred_tracks=preds[-1].detach().cpu().permute(0, 2, 1, 3).numpy(),  # TODO
            query_mode='strided',
    )
    avg_jaccard, avg_pts, occ_acc = scalars["average_jaccard"], scalars["average_pts_within_thresh"], scalars["occlusion_accuracy"]
    return total_loss, (avg_jaccard, avg_pts, occ_acc, ate_all, ate_vis)



def requires_grad(parameters, flag=True):
    for p in parameters:
        p.requires_grad = flag

def fetch_optimizer(lr, wdecay, epsilon, num_steps, params):
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=wdecay, eps=epsilon)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_steps+200)
    
    return optimizer, scheduler


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
    parser.add_argument("--dataset_path", help="path to the crohd dataset", default='/orion/group/head_tracking')
    parser.add_argument('--pseudo_labels_path', type=str, default='all_crohd_data.pkl')
    args = parser.parse_args()
    dataset_path = args.dataset_path
    pseudo_labels_path = args.pseudo_labels_path

    n_pool = 10000
    ate_all_pool_v = utils.misc.SimplePool(n_pool, version='np')
    ate_vis_pool_v = utils.misc.SimplePool(n_pool, version='np')
    ate_occ_pool_v = utils.misc.SimplePool(n_pool, version='np')

    exp = 1
    if exp == 0:
        grad_acc = 4
        lr = 1e-4
        use_scheduler = False

    elif exp == 1:
        grad_acc = 8
        lr = 1e-5
        use_scheduler = True
    global_step = 0


    assert(modeltype=='pips' or modeltype=='raft' or modeltype=='dino')
    
    S_stride = 3 # subsample the frames this much

    ## autogen a name
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

    dataset = CrohdDataset_Train(seqlen=S*S_stride, pickle_path=pseudo_labels_path)
    print("Total Number of data", len(dataset))
    train_dataloader = DataLoader(
        dataset,
        batch_size=B,
        shuffle=False,
        num_workers=12)
    
    val_dataset = CrohdDataset(dataset_path=dataset_path, seqlen=S*S_stride)
    test_dataloader = DataLoader(
        val_dataset,
        batch_size=B,
        shuffle=False,
        num_workers=12)
    
    train_iterloader = iter(train_dataloader)

    global_step = 0
    
    if modeltype=='pips':
        model = Pips(S=S, stride=stride).cuda()
        _ = saverloader.load(init_dir, model)
        parameters = list(model.parameters())
        requires_grad(parameters, True)
    else:
        assert(False) # need to choose a valid modeltype

    max_iters = 5000
    if use_scheduler:
        print(f"Learning Rate used: {lr}")
        optimizer, scheduler = fetch_optimizer(lr, 0.0001, 1e-8, max_iters // grad_acc, model.parameters())
    else:
        optimizer = torch.optim.Adam(parameters, lr=lr, weight_decay=1e-7)
    optimizer.zero_grad()

    n_pool = 10000
    ate_all_pool_t = utils.misc.SimplePool(n_pool, version='np')
    ate_vis_pool_t = utils.misc.SimplePool(n_pool, version='np')
    ate_occ_pool_t = utils.misc.SimplePool(n_pool, version='np')
    
    print('setting max_iters', max_iters)
    CycleConsty = CycleConsistency(True, True)

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

        try:
            sample = next(train_iterloader)
        except StopIteration:
            train_iterloader = iter(train_dataloader)
            sample = next(train_iterloader)
            # sample, returned_early = prep_sample(sample, N, S_stride, req_occlusion)

        if global_step % VAL_FREQ == 0:
            print("save model")
            save_name = f"exp{exp}_reqocc{REQ_OCC}_iter{global_step}.pt"
            torch.save(model.state_dict(), save_name)
            model.eval()
            n_pool = 10000
            ate_all_pool_v = utils.misc.SimplePool(n_pool, version='np')
            ate_vis_pool_v = utils.misc.SimplePool(n_pool, version='np')
            ate_occ_pool_v = utils.misc.SimplePool(n_pool, version='np')
            validate(model, test_dataloader, N=16, S_stride=3, req_occlusion=REQ_OCC, ate_all_pool_t=ate_all_pool_v, ate_vis_pool_t=ate_vis_pool_v, ate_occ_pool_t=ate_occ_pool_v)
            model.train()
            # exit()
        read_time = time.time()-read_start_time
        iter_start_time = time.time()
            
        total_loss, (avg_jaccard, avg_pts, occ_acc, ate_all, ate_vis) = run_pips(model, sample, sw_t, CycleConsty)
        total_loss.backward()

        sw_t.summ_scalar('total_loss', total_loss)
        sw_t.summ_scalar('jac', np.average(avg_jaccard))
        sw_t.summ_scalar('d_avg', np.average(avg_pts))
        sw_t.summ_scalar('occ_acc', np.average(occ_acc))
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
    save_name = f"exp{exp}_reqocc{REQ_OCC}_iter{global_step}.pt"
    torch.save(model.state_dict(), save_name)
    model.eval()
    n_pool = 10000
    ate_all_pool_v = utils.misc.SimplePool(n_pool, version='np')
    ate_vis_pool_v = utils.misc.SimplePool(n_pool, version='np')
    ate_occ_pool_v = utils.misc.SimplePool(n_pool, version='np')
    validate(model, test_dataloader, N=16, S_stride=3, req_occlusion=REQ_OCC, ate_all_pool_t=ate_all_pool_v, ate_vis_pool_t=ate_vis_pool_v, ate_occ_pool_t=ate_occ_pool_v)
    model.train()
    
    writer_t.close()

if __name__ == '__main__':
    Fire(main)

