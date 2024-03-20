# -*- coding: utf-8 -*-
import time

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

import torch
import torch.nn as nn

from lib.utils.geometry import batch_rodrigues


class VIBELoss(nn.Module):
    def __init__(
            self,
            e_loss_weight=60.,
            e_3d_loss_weight=30.,
            e_pose_loss_weight=1.,
            e_shape_loss_weight=0.001,
            d_motion_loss_weight=1.,
            optical_flow_loss_weight=0.001,
            device='cuda',
    ):
        super(VIBELoss, self).__init__()
        self.e_loss_weight = e_loss_weight
        self.e_3d_loss_weight = e_3d_loss_weight
        self.e_pose_loss_weight = e_pose_loss_weight
        self.e_shape_loss_weight = e_shape_loss_weight
        self.d_motion_loss_weight = d_motion_loss_weight
        self.optical_flow_loss_weight = optical_flow_loss_weight

        self.device = device
        self.criterion_shape = nn.L1Loss().to(self.device)
        self.criterion_keypoints = nn.MSELoss(reduction='none').to(self.device)
        self.criterion_regr = nn.MSELoss().to(self.device)

        self.enc_loss = batch_encoder_disc_l2_loss
        self.dec_loss = batch_adv_disc_l2_loss

    def forward(
            self,
            generator_outputs,
            data_2d,
            data_3d,
            data_body_mosh=None,
            data_motion_mosh=None,
            body_discriminator=None,
            motion_discriminator=None,
    ):
        # to reduce time dimension
        reduce = lambda x: x.reshape((x.shape[0] * x.shape[1],) + x.shape[2:])
        # flatten for weight vectors
        flatten = lambda x: x.reshape(-1)
        # accumulate all predicted thetas from IEF

        if data_2d:
            sample_2d_count = data_2d['kp_2d'].shape[0]
            real_2d = torch.cat((data_2d['kp_2d'], data_3d['kp_2d']), 0)
            data_flow = torch.cat((data_2d['flow'], data_3d['flow']), 0)
        else:
            sample_2d_count = 0
            real_2d = data_3d['kp_2d']
            data_flow = data_3d['flow']

        real_2d = reduce(real_2d)

        real_3d = reduce(data_3d['kp_3d'])
        data_3d_theta = reduce(data_3d['theta'])

        w_3d = data_3d['w_3d'].type(torch.bool)
        w_smpl = data_3d['w_smpl'].type(torch.bool)

        preds = generator_outputs[-1]

        '''
        for k, v in preds.items():
            print(k, v.shape)

        print('flow', data_flow.shape)
        # '''

        pred_j3d = preds['kp_3d'][sample_2d_count:]
        pred_theta = preds['theta'][sample_2d_count:]

        pred_theta = reduce(pred_theta)
        pred_j2d = reduce(preds['kp_2d'])
        pred_j3d = reduce(pred_j3d)

        w_3d = flatten(w_3d)
        w_smpl = flatten(w_smpl)

        pred_theta = pred_theta[w_smpl]
        pred_j3d = pred_j3d[w_3d]
        data_3d_theta = data_3d_theta[w_smpl]
        real_3d = real_3d[w_3d]

        pred_cam = preds['theta'][:, :, :3]

        # <======== Generator Loss
        loss_kp_2d = self.keypoint_loss(pred_j2d, real_2d, openpose_weight=1., gt_weight=1.) * self.e_loss_weight

        loss_kp_3d = self.keypoint_3d_loss(pred_j3d, real_3d)
        loss_kp_3d = loss_kp_3d * self.e_3d_loss_weight

        real_shape, pred_shape = data_3d_theta[:, 75:], pred_theta[:, 75:]
        real_pose, pred_pose = data_3d_theta[:, 3:75], pred_theta[:, 3:75]

        loss_dict = {
            'loss_kp_2d': loss_kp_2d,
            'loss_kp_3d': loss_kp_3d,
        }
        if pred_theta.shape[0] > 0:
            loss_pose, loss_shape = self.smpl_losses(pred_pose, pred_shape, real_pose, real_shape)
            loss_shape = loss_shape * self.e_shape_loss_weight
            loss_pose = loss_pose * self.e_pose_loss_weight
            loss_dict['loss_shape'] = loss_shape
            loss_dict['loss_pose'] = loss_pose

        optical_flow_loss = self.batch_optical_flow_loss(preds['verts'], data_flow, pred_cam) * self.optical_flow_loss_weight

        loss_dict['optical_flow'] = optical_flow_loss

        gen_loss = torch.stack(list(loss_dict.values())).sum()

        '''
        # <======== Motion Discriminator Loss
        end_idx = 75
        start_idx = 6
        pred_motion = total_predict_thetas
        e_motion_disc_loss = self.enc_loss(motion_discriminator(pred_motion[:, :, start_idx:end_idx]))
        e_motion_disc_loss = e_motion_disc_loss * self.d_motion_loss_weight

        fake_motion = pred_motion.detach()
        real_motion = data_motion_mosh['theta']
        fake_disc_value = motion_discriminator(fake_motion[:, :, start_idx:end_idx])
        real_disc_value = motion_discriminator(real_motion[:, :, start_idx:end_idx])
        d_motion_disc_real, d_motion_disc_fake, d_motion_disc_loss = self.dec_loss(real_disc_value, fake_disc_value)

        d_motion_disc_real = d_motion_disc_real * self.d_motion_loss_weight
        d_motion_disc_fake = d_motion_disc_fake * self.d_motion_loss_weight
        d_motion_disc_loss = d_motion_disc_loss * self.d_motion_loss_weight

        loss_dict['e_m_disc_loss'] = e_motion_disc_loss
        loss_dict['d_m_disc_real'] = d_motion_disc_real
        loss_dict['d_m_disc_fake'] = d_motion_disc_fake
        loss_dict['d_m_disc_loss'] = d_motion_disc_loss

        gen_loss = gen_loss + e_motion_disc_loss
        motion_dis_loss = d_motion_disc_loss
        '''

        return gen_loss, loss_dict

    def keypoint_loss(self, pred_keypoints_2d, gt_keypoints_2d, openpose_weight, gt_weight):
        """
        Compute 2D reprojection loss on the keypoints.
        The loss is weighted by the confidence.
        The available keypoints are different for each dataset.
        """
        conf = gt_keypoints_2d[:, :, -1].unsqueeze(-1).clone()
        conf[:, :25] *= openpose_weight
        conf[:, 25:] *= gt_weight
        loss = (conf * self.criterion_keypoints(pred_keypoints_2d, gt_keypoints_2d[:, :, :-1])).mean()
        return loss

    def keypoint_3d_loss(self, pred_keypoints_3d, gt_keypoints_3d):
        """
        Compute 3D keypoint loss for the examples that 3D keypoint annotations are available.
        The loss is weighted by the confidence.
        """
        pred_keypoints_3d = pred_keypoints_3d[:, 25:39, :]
        gt_keypoints_3d = gt_keypoints_3d[:, 25:39, :]

        # conf = gt_keypoints_3d[:, :, -1].unsqueeze(-1).clone()
        # gt_keypoints_3d = gt_keypoints_3d[:, :, :-1].clone()
        # gt_keypoints_3d = gt_keypoints_3d
        # conf = conf
        pred_keypoints_3d = pred_keypoints_3d
        if len(gt_keypoints_3d) > 0:
            gt_pelvis = (gt_keypoints_3d[:, 2, :] + gt_keypoints_3d[:, 3, :]) / 2
            gt_keypoints_3d = gt_keypoints_3d - gt_pelvis[:, None, :]
            pred_pelvis = (pred_keypoints_3d[:, 2, :] + pred_keypoints_3d[:, 3, :]) / 2
            pred_keypoints_3d = pred_keypoints_3d - pred_pelvis[:, None, :]
            # print(conf.shape, pred_keypoints_3d.shape, gt_keypoints_3d.shape)
            # return (conf * self.criterion_keypoints(pred_keypoints_3d, gt_keypoints_3d)).mean()
            return self.criterion_keypoints(pred_keypoints_3d, gt_keypoints_3d).mean()
        else:
            return torch.FloatTensor(1).fill_(0.).to(self.device)

    def smpl_losses(self, pred_rotmat, pred_betas, gt_pose, gt_betas):
        pred_rotmat_valid = batch_rodrigues(pred_rotmat.reshape(-1, 3)).reshape(-1, 24, 3, 3)
        gt_rotmat_valid = batch_rodrigues(gt_pose.reshape(-1, 3)).reshape(-1, 24, 3, 3)
        pred_betas_valid = pred_betas
        gt_betas_valid = gt_betas
        if len(pred_rotmat_valid) > 0:
            loss_regr_pose = self.criterion_regr(pred_rotmat_valid, gt_rotmat_valid)
            loss_regr_betas = self.criterion_regr(pred_betas_valid, gt_betas_valid)
        else:
            loss_regr_pose = torch.FloatTensor(1).fill_(0.).to(self.device)
            loss_regr_betas = torch.FloatTensor(1).fill_(0.).to(self.device)
        return loss_regr_pose, loss_regr_betas

    def batch_camera_scale_regularization(self, pred_theta):
        camera_scale = pred_theta[:, :, 0]
        camera_scale_regularization = (1 / camera_scale - 1) ** 2
        return torch.sum(camera_scale_regularization)


    def batch_optical_flow_loss(self, pred_verts, gt_flow, pred_cam):
        '''
        # test code, if this not work, use for-loop instead
        # work for pytorch 2.0.0, but not for 1.4.0 in providing environment
        print(torch.__version__)
        a = torch.arange(18).view(6, 3)
        x = torch.arange(3).to(torch.int)
        print(a[x, x], flush=True)
        #'''

        batch_size, seq_len = gt_flow.shape[:2]
        batch_loss = []

        timer = {
            'project': 0,
            'mask': 0,
            'flow': 0
        }

        for b in range(batch_size):
            prefix_mask = torch.empty((6890), dtype=torch.uint8)
            prefix_verts_2d = torch.empty((6890, 2), dtype=torch.float32)

            seq_loss = []

            for i in range(seq_len):

                start = time.time()

                verts = pred_verts[b, i, :, :]
                s = pred_cam[b, i, 0].detach()
                tx = pred_cam[b, i, 1].detach()
                ty = pred_cam[b, i, 2].detach()

                proj_cam = torch.eye(4, device=self.device)
                proj_cam[0, 0] = 112 * s
                proj_cam[1, 1] = 112 * s
                proj_cam[0, 3] = 112 * (s * tx + 1)
                proj_cam[1, 3] = 112 * (s * ty + 1)

                verts = torch.cat((verts.permute(1, 0), torch.ones(1, 6890).to(self.device)))
                verts = torch.matmul(proj_cam, verts)

                depth_map = torch.ones(224, 224).to(self.device, dtype=torch.float)

                verts_int = verts[:2].to(torch.int)

                valid_mask = (verts_int[0] >= 0) & (verts_int[0] < 224) & (verts_int[1] >= 0) & (verts_int[1] < 224)

                depth_map[verts_int[1, valid_mask], verts_int[0, valid_mask]] = torch.min(depth_map[verts_int[1, valid_mask], verts_int[0, valid_mask]], verts[2, valid_mask])

                '''
                for j in range(6890):
                    v = verts[:, j]
                    x = int(v[0])
                    y = int(v[1])
                    if x < 0 or x >= 224 or y < 0 or y >= 224:
                        continue
                    depth_map[x, y] = min(depth_map[x, y], v[2])
                #'''

                verts_2d = verts[:2, :]

                x_coords = verts_2d[0].to(torch.int)
                y_coords = verts_2d[1].to(torch.int)

                x1_coords = x_coords + 1
                y1_coords = y_coords + 1

                z_coords = verts[2]

                in_image_mask = (x_coords >= 0) & (x_coords < 223) & (y_coords >= 0) & (y_coords < 223)

                x_in_image = torch.where(in_image_mask, x_coords, torch.zeros_like(x_coords))
                y_in_image = torch.where(in_image_mask, y_coords, torch.zeros_like(y_coords))

                x1_in_image = torch.where(in_image_mask, x1_coords, torch.zeros_like(x_coords))
                y1_in_image = torch.where(in_image_mask, y1_coords, torch.zeros_like(y_coords))

                mask1 = torch.zeros((6890), dtype=torch.uint8, requires_grad=False, device=self.device)
                depth_values = torch.where(in_image_mask, depth_map[y_in_image, x_in_image], torch.ones_like(z_coords))
                mask1[in_image_mask & (depth_values + 0.02 >= z_coords)] = 1

                mask2 = torch.zeros((6890), dtype=torch.uint8, requires_grad=False, device=self.device)
                depth_values = torch.where(in_image_mask, depth_map[y_in_image, x1_in_image], torch.ones_like(z_coords))
                mask2[in_image_mask & (depth_values + 0.02 >= z_coords)] = 1

                mask3 = torch.zeros((6890), dtype=torch.uint8, requires_grad=False, device=self.device)
                depth_values = torch.where(in_image_mask, depth_map[y1_in_image, x_in_image], torch.ones_like(z_coords))
                mask3[in_image_mask & (depth_values + 0.02 >= z_coords)] = 1

                mask4 = torch.zeros((6890), dtype=torch.uint8, requires_grad=False, device=self.device)
                depth_values = torch.where(in_image_mask, depth_map[y1_in_image, x1_in_image],
                                           torch.ones_like(z_coords))
                mask4[in_image_mask & (depth_values + 0.02 >= z_coords)] = 1

                mask = mask1 & mask2 & mask3 & mask4
                mask.requires_grad_(False)

                '''
                for j in range(6890):
                    x = int(verts_2d[0, j])
                    y = int(verts_2d[1, j])
                    if x < 0 or x >= 224 or y < 0 or y >= 224:
                        continue
                    d = depth_map[int(y), int(x)]
                    if d != 0 and d * 1.02 > verts[2, j]:
                        mask[j] = 1

                timer['mask'] += time.time() - start
                start = time.time()
                #'''

                if i > 0:
                    combined_mask = mask & prefix_mask

                    flow = gt_flow[b, i - 1].clone()
                    # print(flow.shape)

                    flow_difs = []

                    #'''
                    valid_mask = (prefix_verts_2d[0] >= 0) & (prefix_verts_2d[0] < 224) & (prefix_verts_2d[1] >= 0) & (
                                prefix_verts_2d[1] < 224) & combined_mask

                    valid_indices = torch.where(valid_mask)[0]

                    #print(valid_mask)
                    #print(valid_indices)

                    pred_flow_valid = verts_2d[:, valid_indices] - prefix_verts_2d[:, valid_indices]

                    tmp = pred_flow_valid[0].clone()
                    pred_flow_valid[0] = pred_flow_valid[1].clone()
                    pred_flow_valid[1] = tmp

                    flow_dif_valid = (pred_flow_valid - flow[:, prefix_verts_2d_int[1, valid_indices],
                                                        prefix_verts_2d_int[0, valid_indices]]) ** 2
                    flow_difs.append(flow_dif_valid)
                    #'''

                    '''
                    for k in range(6890):
                        x = int(prefix_verts_2d[0, k])
                        y = int(prefix_verts_2d[1, k])
                        if x < 0 or x >= 224 or y < 0 or y >= 224 or combined_mask[k] == 0:
                            continue
                        else:
                            pred_flow = verts_2d[:, k] - prefix_verts_2d[:, k]

                            tmp = pred_flow[0].clone()
                            pred_flow[0] = pred_flow[1].clone()
                            pred_flow[1] = tmp

                            flow_dif = (pred_flow - flow[:, y, x]) ** 2
                            flow_difs.append(flow_dif)       
                    #'''

                    if len(flow_difs) != 0:
                        pair_loss = torch.mean(torch.pow(torch.cat(flow_difs), 2))
                    else:
                        pair_loss = torch.tensor(0, device=self.device)

                    seq_loss.append(pair_loss)

                timer['flow'] += time.time() - start


                prefix_verts_2d = verts_2d.clone()
                prefix_verts_2d_int = prefix_verts_2d.to(torch.int)
                prefix_mask = mask.clone()

            seq_loss = torch.mean(torch.stack(seq_loss))

        #print(timer, flush=True)

        batch_loss.append(seq_loss)

        return torch.mean(torch.stack(batch_loss))


def batch_encoder_disc_l2_loss(disc_value):
    '''
        Inputs:
            disc_value: N x 25
    '''
    k = disc_value.shape[0]
    return torch.sum((disc_value - 1.0) ** 2) * 1.0 / k


def batch_adv_disc_l2_loss(real_disc_value, fake_disc_value):
    '''
        Inputs:
            disc_value: N x 25
    '''
    ka = real_disc_value.shape[0]
    kb = fake_disc_value.shape[0]
    lb, la = torch.sum(fake_disc_value ** 2) / kb, torch.sum((real_disc_value - 1) ** 2) / ka
    return la, lb, la + lb


def batch_encoder_disc_wasserstein_loss(disc_value):
    '''
        Inputs:
            disc_value: N x 25
    '''
    k = disc_value.shape[0]
    return -1 * disc_value.sum() / k


def batch_adv_disc_wasserstein_loss(real_disc_value, fake_disc_value):
    '''
        Inputs:
            disc_value: N x 25
    '''

    ka = real_disc_value.shape[0]
    kb = fake_disc_value.shape[0]

    la = -1 * real_disc_value.sum() / ka
    lb = fake_disc_value.sum() / kb
    return la, lb, la + lb


def batch_smooth_pose_loss(pred_theta):
    pose = pred_theta[:, :, 3:75]
    pose_diff = pose[:, 1:, :] - pose[:, :-1, :]
    return torch.mean(pose_diff).abs()


def batch_smooth_shape_loss(pred_theta):
    shape = pred_theta[:, :, 75:]
    shape_diff = shape[:, 1:, :] - shape[:, :-1, :]
    return torch.mean(shape_diff).abs()
