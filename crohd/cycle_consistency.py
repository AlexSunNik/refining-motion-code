import numpy as np 
import torch 
import evaluation_datasets
import torch.nn.functional as F

class CycleConsistency:
    def __init__(self, no_speed=False, all_time=False):
        # 0: 30%, 1: 25%, 2: 15%, 3: 10%, 4: 5%
        # 0: 30%, 1: 25%, 2, 20%, 3: 15%, 4: 10%, 5: 5%
        # self.mode = 2

        # Use no speed info
        self.no_speed = no_speed
        if no_speed:
            # 0: 95%, 1: 90%, 2, 85%, 3: 80%, 4: 75%
            # d_0.5, d_1, d_1.5
            self.mode = 1
            # self.tau_jac = [0.2734605377276667, 0.44692107545533344, 1.7103166479354606, 5.429761715502367, None]
            self.tau_jac = [0.3224340641881164, 0.668691255558749, 1.7713643759508317, 5.248898947309022, None]
        else:
            # 0: 30%, 1: 25%, 2, 20%, 3: 15%, 4: 10%, 5: 5%
            self.mode = 1
            # self.tau_jac = {(0, 1): [0.012321752657506138,
            #                     0.28732300293881946,
            #                     1.4370687371118924,
            #                     2.9049567910160476,
            #                     5.475756898173625],
            #                     (1, 2): [0.39756953703990816,
            #                     0.4843939346956494,
            #                     3.3102628404079555,
            #                     6.2074610248362845,
            #                     15.299415916501111],
            #                     (2, 4.5): [0.5603302364973562,
            #                     0.7499844611139248,
            #                     4.07666080436105,
            #                     8.310720107478343,
            #                     29.54579878032021],
            #                     (4.5, None): [2.6918125066234895,
            #                     6.157756091600325,
            #                     21.472421970612903,
            #                     35.14220898731958,
            #                     48.81199600402625]}
            self.tau_jac = {(0, 1): [0.012321752657506138,
                                0.28732300293881946,
                                0.86219587, # maybe 20%
                                1.4370687371118924,
                                2.9049567910160476,
                                5.475756898173625],
                                (1, 2): [0.39756953703990816,
                                0.4843939346956494,
                                1.897328388,# maybe 20%
                                3.3102628404079555,
                                6.2074610248362845,
                                15.299415916501111],
                                (2, 4.5): [0.5603302364973562,
                                0.7499844611139248,
                                2.413322633,# maybe 20%
                                4.07666080436105,
                                8.310720107478343,
                                29.54579878032021],
                                (4.5, None): [2.6918125066234895,
                                6.157756091600325,
                                13.81508903,# maybe 20%
                                21.472421970612903,
                                35.14220898731958,
                                48.81199600402625]}
        self.all_time = all_time
        if all_time:
            self.mode = 0
        
        print("Mode Selected", self.mode)
    # speed: speed of the track
    # inconsty: L2 inconsistency
    # we see that high speed tracks are not robust in transferrability
    # only keep the first three regions
    # return true for good tracks
    
    # def check_inconsty_regions(self, speed, inconsty):
    #     if self.no_speed:
    #         return bool(inconsty < self.tau_jac[self.mode])
    #     else:
    #         for region, taus in self.tau_jac.items():
    #             if region[1] is None and speed > region[0]:
    #                 return False
    #             elif speed >= region[0] and speed < region[1]:
    #                 return bool(inconsty < taus[self.mode])
    #         return False

    def check_inconsty_regions(self, speed, inconsty):
        if self.no_speed:
            return bool(inconsty < self.tau_jac[self.mode])
        else:
            for region, taus in self.tau_jac.items():
                if region[1] is None and speed > region[0]:
                    return bool(inconsty < taus[self.mode])
                elif speed >= region[0] and speed < region[1]:
                    return bool(inconsty < taus[self.mode])
            return False
        
    def decide_good_tracks(self, speed, inconsty, backward_start_occ, occ_consistency):
        # Check occlusion
        res = []
        # Get each point
        backward_start_occ = backward_start_occ.view(-1)
        occ_consistency = occ_consistency.view(-1)
        for i in range(len(inconsty)):
            if (backward_start_occ[i] + occ_consistency[i]) >= 1:
                res.append(False)
                continue
            # Check cycle inconsistency for speed regions
            if self.no_speed:
                res.append(self.check_inconsty_regions(None, inconsty[i]))
            else:
                res.append(self.check_inconsty_regions(speed[i], inconsty[i]))
        
        return res
    
    def decide_good_tracks2(self, speed, d_thresh, backward_start_occ, occ_consistency, metric=1):
        # Get each point
        # backward_start_occ = backward_start_occ.view(-1)
        # occ_consistency = occ_consistency.view(-1)
        sel_d = d_thresh[self.mode]
        res = (sel_d >= metric)
        return res
    
    def decide_good_tracks2_color(self, speed, d_thresh, color_inconsty, backward_start_occ, occ_consistency, color_thresh=5):
        # Get each point
        # backward_start_occ = backward_start_occ.view(-1)
        # occ_consistency = occ_consistency.view(-1)
        sel_d = d_thresh[self.mode]
        res = (sel_d == 1)
        color_res = (color_inconsty < color_thresh).cpu().unsqueeze(-1)
        return res, color_res
    
    def decide_good_tracks3(self, speed, d_thresh):
        # Get each point
        sel_d = d_thresh[self.mode]
        res = (sel_d == 1)
        return res
    
    def get_color_fwd_inconsty(self, preds, rgbs, Y, X):
        # print(preds.shape)
        # print(rgbs.shape)
        sampled_rgbs = []
        # print(preds.shape)
        # print(rgbs.shape)
        for s in range(preds.shape[0]):
            coords = preds[s:s+1].unsqueeze(0)
            cur_rgbs = rgbs[0, s:s+1]
            # print(coords.shape)
            # print(cur_rgbs.shape)
            # print(cur_rgbs[0, 0, 0, 0])
            # print(coords)

            # xy
            # print(coords.shape)
            # print(Y, X)
            y = 2.0*(coords[..., 1:2] / float(Y-1)) - 1.0
            x = 2.0*(coords[..., 0:1] / float(X-1)) - 1.0
            coords = torch.cat([y, x], dim = -1)
            # coords = torch.cat([x, y], dim = -1)
            sampled_rgb = F.grid_sample(cur_rgbs, coords, mode='bilinear', padding_mode='zeros')
            sampled_rgb = sampled_rgb.squeeze(2)
            # print(sampled_rgb.flatten()[-10:])
            if s == 0:
                query_rgb = sampled_rgb
            else: 
                sampled_rgbs.append((sampled_rgb - query_rgb) ** 2)

        sampled_rgbs = torch.cat(sampled_rgbs, dim=0)
        # print(sampled_rgbs.shape)
        rgb_difference = sampled_rgbs
        # rgb_difference = (sampled_rgbs - query_rgb) ** 2
        # print(rgb_difference.shape)
        rgb_difference = torch.sum(rgb_difference, dim=1).sqrt().mean(dim=0)
        # print(rgb_difference.shape)
        # print(rgb_difference)
        # print(torch.max(rgb_difference))
        return rgb_difference

    def get_inconsty(self, points, pred_points):
        points = points.view(-1, 2).cpu()
        pred_points = pred_points.view(-1, 2).cpu()
        diff = torch.sqrt(torch.sum((points - pred_points) ** 2, axis = -1))
        return diff

    # Here we enforce cycle consistency on all of the temporal frames
    def get_inconsty2(self, preds, bkw_preds, threshs=[1]):
        # 8, N, 2
        bkw_preds = torch.flip(bkw_preds, dims=[0])
        scalars = evaluation_datasets.compute_thresh_consty(
                    gt_tracks=preds.detach().cpu().numpy(),
                    pred_tracks=bkw_preds.detach().cpu().numpy(),
                    threshs=threshs
            )
        # print(scalars)
        return scalars
    
    def get_inconsty3(self, preds, bkw_preds):
        bkw_preds = torch.flip(bkw_preds, dims=[0])
        # print(bkw_preds.shape)
        # print(preds.shape)
        # exit()
        diff = torch.sqrt(torch.sum((preds - bkw_preds) ** 2, axis = [0, 2]))
        # print(scalars)
        return diff
    
    def get_occ_inconsty(self, occ, bkw_occ):
        occ = occ.cpu().to(torch.int)
        bkw_occ = bkw_occ.cpu().to(torch.int)
        diff = torch.abs(occ - bkw_occ)
        return diff.squeeze(0)

    def get_pred_speed(self, traj, num_frames):
        S = traj.shape[0]
        N = traj.shape[1]
        total_dist = torch.tensor([0.0] * N)
        for i in range(S - 1):
            cur_dist = torch.sqrt(torch.sum((traj[i+1] - traj[i]) ** 2, axis=-1)).cpu()
            total_dist += cur_dist
        pred_speed = total_dist / num_frames
        return pred_speed