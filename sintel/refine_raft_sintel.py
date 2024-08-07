
# ['alley_1', 'alley_2', 'ambush_2', 'ambush_4', 
#  'ambush_5', 'ambush_6', 'ambush_7', 'bamboo_1', 
#  'bamboo_2', 'bandage_1', 'bandage_2', 'cave_2', 
#  'cave_4', 'market_2', 'market_5', 'market_6', 
#  'mountain_1', 'shaman_2', 'shaman_3', 'sleeping_1', 
#  'sleeping_2', 'temple_2', 'temple_3']
MAXITERS = 1000
VAL_FREQ = 100
ALPHA = 0.01*0.25
BETA = 0.5*0.25
MAX_FLOW = 400
import time
import numpy as np
import os
from copy import deepcopy
import utils.improc
import utils.geom
import random
from utils.basic import print_, print_stats
from sinteldataset import SintelDataset
from sinteldataset_finetune_raft import SintelDatasetTrain, COLOR_THRESH
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tensorboardX import SummaryWriter
import torch.nn.functional as F
from fire import Fire
from nets.raftnet import *

random.seed(125)
np.random.seed(125)
torch.manual_seed(125)


def sequence_loss(flow_preds, flow_gt, valid, gamma=0.8, max_flow=MAX_FLOW):
    """ Loss function defined over sequence of flow predictions """

    n_predictions = len(flow_preds)    
    flow_loss = 0.0

    # exlude invalid pixels and extremely large diplacements
    mag = torch.sum(flow_gt**2, dim=1).sqrt()
    mag = mag.unsqueeze(1)
    valid = (valid >= 0.5) & (mag < max_flow)

    for i in range(n_predictions):
        i_weight = gamma**(n_predictions - i - 1)
        i_loss = (flow_preds[i] - flow_gt).abs()
        flow_loss += i_weight * (valid[:, None] * i_loss).mean()

    epe = torch.sum((flow_preds[-1] - flow_gt)**2, dim=1).sqrt()
    epe = epe.view(-1)[valid.view(-1)]

    metrics = {
        'epe': epe.mean().item(),
        '1px': (epe < 1).float().mean().item(),
        '3px': (epe < 3).float().mean().item(),
        '5px': (epe < 5).float().mean().item(),
    }

    return flow_loss, metrics

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


def generate_window_grid(h_min, h_max, w_min, w_max, len_h, len_w, device=None):
    assert device is not None

    x, y = torch.meshgrid([torch.linspace(w_min, w_max, len_w, device=device),
                           torch.linspace(h_min, h_max, len_h, device=device)],
                          )
    grid = torch.stack((x, y), -1).transpose(0, 1).float()  # [H, W, 2]

    return grid


def normalize_coords(coords, h, w):
    # coords: [B, H, W, 2]
    c = torch.Tensor([(w - 1) / 2., (h - 1) / 2.]).float().to(coords.device)
    return (coords - c) / c  # [-1, 1]


def get_truly_random_seed_through_os():
    """
    Usually the best random sample you could get in any programming language is generated through the operating system. 
    In Python, you can use the os module.

    source: https://stackoverflow.com/questions/57416925/best-practices-for-generating-a-random-seeds-to-seed-pytorch/57416967#57416967
    """
    RAND_SIZE = 4
    random_data = os.urandom(
        RAND_SIZE
    )  # Return a string of size random bytes suitable for cryptographic use.
    random_seed = int.from_bytes(random_data, byteorder="big")
    return random_seed

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
                                       alpha=ALPHA,
                                       beta=BETA,
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

def requires_grad(parameters, flag=True):
    for p in parameters:
        p.requires_grad = flag

def fetch_optimizer(lr, wdecay, epsilon, num_steps, params):
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=wdecay, eps=epsilon)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, lr, num_steps+100,
        pct_start=0.1, cycle_momentum=False, anneal_strategy='linear')
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_steps, eta_min=0)
    return optimizer, scheduler


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

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


def validate(val_loader, model, device, sw, save_img=False):
    model.eval()
    model.model = torch.nn.DataParallel(model.model.module, device_ids=[0])
    total_loss = torch.tensor(0.0, requires_grad=True).to(device)
    metrics = {}
    metrics['l1'] = None
    metrics['l2'] = None
    total_err = []
    epe_list = []
    # metrics['ate_all'] = 0.0
    for idx, sample in enumerate(val_loader):
        d = sample[0]
        rgb0 = d['rgb0'].to(device).float() # B, C, H, W
        rgb1 = d['rgb1'].to(device).float() # B, C, H, W
        flow_g = d['flow'].to(device).float() # B, 2, H, W

        B, C, H, W = rgb0.shape
        assert(C==3)
        B, D, H, W = flow_g.shape
        assert(D==2)

        rgbs = torch.stack([rgb0, rgb1], dim=1) # B,S,C,H,W
        S = 2

        clip = max(min(torch.max(torch.abs(flow_g[0:1])),50.0),2.0)
            
        if sw is not None and sw.save_this:
            sw.summ_rgbs(f'0_inputs/{idx}/rgbs', utils.improc.preprocess_color(rgbs[0:1]).unbind(1))
            sw.summ_flow(f'0_inputs/{idx}/flow_g', flow_g[0:1], clip=clip)

        with torch.no_grad():
            flow_fw, _ = model(rgb0, rgb1, iters=32)

        flow_err = torch.norm(flow_fw - flow_g, dim=1, keepdim=True)

        flow_err_clamp = flow_err.clamp(max=5.0)/5.0
        total_err.append(flow_err.mean().item())
        epe = torch.sum((flow_fw.squeeze() - flow_g.squeeze())**2, dim=0).sqrt()
        epe_list.append(epe.view(-1).detach().cpu().numpy())
        
        if sw is not None and sw.save_this and save_img:
            sw.summ_flow(f'0_inputs/{idx}/flow_fw', flow_fw, clip=clip)
            sw.summ_oned(f'0_inputs/{idx}/flow_err', flow_err_clamp, norm=False, frame_id=flow_err.mean().item())

        total_err.append(flow_err.mean().item())

    epe_all = np.concatenate(epe_list)
    epe = np.mean(epe_all)
    px1 = np.mean(epe_all<1)
    px3 = np.mean(epe_all<3)
    px5 = np.mean(epe_all<5)

    print("EPE", epe, px1, px3, px5)
    model.model = torch.nn.DataParallel(model.model)
    model.train()
    return total_err, sum(total_err), (epe_all, epe, px1, px3, px5)



def run_model(raft, d, device, horz_flip=False, vert_flip=False, scale_doub=False, time_flip=False, sw=None, is_train=True, just_recon=False):

    metrics = {}
    metrics['l1'] = None
    metrics['l2'] = None
    rgb0 = d['rgb0'].to(device).float() # B, C, H, W
    rgb1 = d['rgb1'].to(device).float() # B, C, H, W
    flow_fw = d['flow_fw'].to(device).float() # B, C, H, W
    occ_fw = d['occ_fw'].to(device).float() # B, C, H, W
    flow_g = d['flow_g'].to(device).float() # B, 2, H, W
    color_consty_map = d['color_consty_map'].to(device).float() # B, C, H, W

    B, D, H, W = flow_g.shape
    assert(D==2)

    flow_predictions = raft(rgb0, rgb1, 32, False)

    valid = color_consty_map.int() & (1-occ_fw).int()

    loss, metrics = sequence_loss(flow_predictions, flow_fw, valid, 0.85)
    
    l1 = torch.abs(flow_predictions[-1] - flow_g).sum(dim=1, keepdim=True)
    l2 = torch.norm(flow_predictions[-1] - flow_g, dim=1) # B,H,W
    metrics['l2'] = l2.mean().item()
    metrics['l1'] = l1.mean().item()
    return loss, metrics

def main(
        exp_name='2imgs',
        B=1, # batchsize 
        S_train=2, # seqlen 
        S_val=2, # seqlen 
        S_test=2, # seqlen 
        horz_flip=True, # this causes B*=2
        vert_flip=True, # this causes B*=2
        scale_doub=True, # this causes B*=2
        time_flip=False, # this causes B*=2
        stride=8, # spatial stride of the model 
        crop_size=(256,512), # raw chairs data is 384,512
        use_augs=True, # resizing/jittering/color/blur augs
        cache_len=0, # how many samples to cache into ram (usually for debug)
        cache_freq=1000, # how often to add a new sample to cache
        flt_location='../flyingthings',
        badja_location='../badja2',
        subset='all', # dataset subset
        shuffle=True, # dataset shuffling
        lr=1e-4,
        grad_acc=1,
        use_scheduler=True,
        log_dir='./debug',
        max_iters=1000000,
        train2_freq=0,
        log_freq=10000,
        ckpt_dir='./checkpoints',
        save_freq=1000,
        keep_latest=1,
        init_dir='',
        load_optimizer=True,
        load_step=True,
        ignore_load=None,
        # cuda
        device_ids=[0],
):
    NUM_FRAMES = 49
    
    parser = argparse.ArgumentParser("TrainSingleVideo")
    parser.add_argument("--scene_name", help="video_name to train on", default='alley_1', type=str)
    parser.add_argument("--split", help="split to train on", default='final', type=str)
    parser.add_argument("--gen_data", help="only gen data", default=False, action='store_true')
    # ckpt_dir
    parser.add_argument("--ckpt_dir", help="checkpoint directory", default='./', type=str)
    # sintel_path
    parser.add_argument("--sintel_path", help="sintel path", default='/orion/group/sintel', type=str)
    args = parser.parse_args()

    scene_name = args.scene_name
    print(f"For Scene {scene_name}")
    gen_data = args.gen_data
    split = args.split
    ckpt_dir = args.ckpt_dir
    sintel_location = args.sintel_path
    
    device = 'cuda:%d' % device_ids[0]

    exp_name = f"{scene_name}_allimgs_gradacc1_12bs_maxiters{MAXITERS}_lr{lr}_colorthresh{COLOR_THRESH}"
    

    B = 1
    horz_flip = False
    vert_flip = False
    scale_doub = False
    time_flip = False
    
    max_iters = MAXITERS
    log_freq = 200
    shuffle = False
    
    use_augs = False
    cache_len = 101


    assert(crop_size[0] % 64 == 0)
    assert(crop_size[1] % 64 == 0)

    S = S_train
    H, W = crop_size
    
    ## autogen a descriptive name
    B_ = B
    if horz_flip:
        B_ *= 2
    if vert_flip:
        B_ *= 2
    if scale_doub:
        B_ *= 8
    if time_flip:
        B_ *= 2
    model_name = "%d" % B_
    if horz_flip:
        model_name += "h"
    if vert_flip:
        model_name += "v"
    if scale_doub:
        model_name += "s"
    if time_flip:
        model_name += "t"
    if grad_acc > 1:
        model_name += "x%d" % grad_acc

    model_name += "_%dx%dx%d" % (S_train, H, W)

    NN = B_*grad_acc
    
    lrn = "%.1e" % lr # e.g., 5.0e-04
    lrn = lrn[0] + lrn[3:5] + lrn[-1] # e.g., 5e-4
    model_name += "_%s" % lrn
    if use_scheduler:
        model_name += "s"
    if cache_len:
        model_name += "_cache%d" % cache_len
    if use_augs:
        model_name += "_A"
    model_name += "_%s" % exp_name
    import datetime
    model_date = datetime.datetime.now().strftime('%H:%M:%S')
    model_name = model_name + '_' + model_date
    print('model_name', model_name)
    
    ckpt_dir = '%s/%s' % (ckpt_dir, model_name)
    writer_t = SummaryWriter(log_dir + '/' + model_name + '/t', max_queue=10, flush_secs=60)

    def worker_init_fn(worker_id):
        np.random.seed(np.random.get_state()[1][0] + worker_id)

    train_dataset = SintelDatasetTrain(
        dataset_location=sintel_location,
        use_augs=False,
        test_mode=False,
        crop_size=crop_size,
        dset='t', 
        scene_name=scene_name,
        scene_num_frames=NUM_FRAMES,
        split=split
    )
    raft = Raftnet(ckpt_name="../reference_model/raft-things.pth").cuda()
    raft.eval()
    train_dataset.prep_data(raft, None, scenename=scene_name, ext_name=f"_aug_{NUM_FRAMES}")
    print("Finished prep data")
    
    if gen_data:
        exit()
    
    val_dataset = SintelDataset(
        dataset_location=sintel_location,
        use_augs=False,
        test_mode=False,
        crop_size=(436, 1024),
        dset='t', 
        scene_name=scene_name,
        scene_num_frames=NUM_FRAMES,
        split=split
    )
    NUM_FRAMES = len(val_dataset)
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=shuffle,
        num_workers=4,
        worker_init_fn=worker_init_fn,
        drop_last=True)
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=12,
        # batch_size=1,
        shuffle=shuffle,
        num_workers=4,
        worker_init_fn=worker_init_fn,
        drop_last=True)
    
    train_iterloader = iter(train_dataloader)
    
    raft.train()
    # raft = nn.DataParallel(raft)
    parameters = list(raft.parameters())
    if use_scheduler:
        optimizer, scheduler = fetch_optimizer(lr, 0.0001, 1e-8, max_iters, parameters)
    else:
        optimizer = torch.optim.Adam(parameters, lr=lr)
    
    global_step = 0
    n_pool = 100
    NN_pool = utils.misc.SimplePool(n_pool*10, version='np')
    loss_pool_t = utils.misc.SimplePool(n_pool*10, version='np')
    l1_pool_t = utils.misc.SimplePool(n_pool*10, version='np')
    l2_pool_t = utils.misc.SimplePool(n_pool*10, version='np')
    epe_pool_t = utils.misc.SimplePool(n_pool*10, version='np')
    
    sw_t = utils.improc.Summ_writer(
        writer=writer_t,
        global_step=global_step,
        log_freq=log_freq,
        fps=8,
        scalar_freq=int(log_freq/2),
        just_gif=True)
    
    data_counter = 0
    losses = []
    raft.train()
    while global_step < max_iters:

        iter_start_time = time.time()
        iter_read_time = 0.0
        
        for internal_step in range(grad_acc):
            global_step += 1

            # read sample
            read_start_time = time.time()
            sw_t = utils.improc.Summ_writer(
                    writer=writer_t,
                    global_step=global_step,
                    log_freq=log_freq,
                    fps=8,
                    scalar_freq=int(log_freq/2),
                    just_gif=True)
                
                
            if global_step == 1 or global_step % VAL_FREQ == 0:
                sw_t.save_this = True
                per_sample_err, total_err, epes = validate(val_dataloader, deepcopy(raft), device, sw_t, save_img=True)
                epe_all, epe, px1, px3, px5 = epes
                sw_t.summ_scalar('epe', float(epe))
                sw_t.summ_scalar('px1', float(px1))
                sw_t.summ_scalar('px3', float(px3))
                sw_t.summ_scalar('px5', float(px5))
                torch.save(raft.state_dict(), f"{scene_name}_{global_step}.pth")

            data_counter += 1
            gotit = (False,False)
            while not all(gotit):
                try:
                    dat, gotit = next(train_iterloader)
                except StopIteration:
                    seed = global_step * 10
                    train_dataset.seed = seed
                    train_iterloader = iter(train_dataloader)
                    dat, gotit = next(train_iterloader)

            read_time = time.time()-read_start_time
            iter_read_time += read_time

            total_loss, metrics = run_model(raft, dat, device, horz_flip=horz_flip, vert_flip=vert_flip, scale_doub=scale_doub, time_flip=time_flip, sw=sw_t, is_train=True)
            total_loss /= grad_acc
            total_loss.backward()
            losses.append(metrics['l2'])
        

        iter_time = time.time()-iter_start_time

        sw_t.summ_scalar('total_loss', total_loss)
        loss_pool_t.update([total_loss.detach().cpu().numpy()])
        sw_t.summ_scalar('pooled/total_loss', loss_pool_t.mean())

        if metrics['l1'] is not None:
            l1_pool_t.update([metrics['l1']])
        if metrics['l2'] is not None:
            l2_pool_t.update([metrics['l2']])
        if metrics['epe'] is not None:
            epe_pool_t.update([metrics['epe']])

        sw_t.summ_scalar('pooled/l1', l1_pool_t.mean())
        sw_t.summ_scalar('pooled/l2', l2_pool_t.mean())
        sw_t.summ_scalar('pooled/epe', epe_pool_t.mean())
        

        torch.nn.utils.clip_grad_norm_(raft.parameters(), 5.0)
        optimizer.step()
        if use_scheduler:
            scheduler.step()
        optimizer.zero_grad()

        current_lr = optimizer.param_groups[0]['lr']
        sw_t.summ_scalar('_/current_lr', current_lr)

        NN_pool.update([NN/iter_time])
        
        print('%s; step %06d/%d; rtime %.2f; itime %.2f (%d/s); loss %.3f; loss_t %.2f; l2_t %.2f; epe %.2f' % (
            model_name, global_step, max_iters, read_time, iter_time, int(NN_pool.mean()),
            total_loss.item(), loss_pool_t.mean(), l2_pool_t.mean(), epe_pool_t.mean()))

            
    writer_t.close()
            

if __name__ == '__main__':
    Fire(main)
