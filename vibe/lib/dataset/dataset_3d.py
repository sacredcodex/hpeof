# -*- coding: utf-8 -*-

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

import os
import torch
import random
import logging
import numpy as np
import os.path as osp
import joblib
import cv2

from torch.utils.data import Dataset
from lib.core.config import VIBE_DB_DIR
from lib.data_utils.kp_utils import convert_kps
from lib.data_utils.img_utils import normalize_2d_kp, transfrom_keypoints, split_into_chunks, generate_patch_image_cv, convert_cvimg_to_tensor

logger = logging.getLogger(__name__)

class Dataset3D(Dataset):
    def __init__(self, set, seqlen, overlap=0., folder=None, dataset_name=None, debug=False):

        self.folder = folder
        self.set = set
        self.dataset_name = dataset_name
        self.seqlen = seqlen
        self.stride = int(seqlen * (1-overlap))
        self.debug = debug
        self.db = self.load_db()
        self.vid_indices = split_into_chunks(self.db['vid_name'], self.seqlen, self.stride)

    def __len__(self):
        return len(self.vid_indices)

    def __getitem__(self, index):
        return self.get_single_item(index)

    def load_db(self):
        db_file = osp.join(VIBE_DB_DIR, f'{self.dataset_name}_{self.set}_db.pt')

        if osp.isfile(db_file):
            db = joblib.load(db_file)
        else:
            raise ValueError(f'{db_file} do not exists')

        print(f'Loaded {self.dataset_name} dataset from {db_file}')
        return db

    def get_single_item(self, index):
        start_index, end_index = self.vid_indices[index]

        is_train = self.set == 'train'

        if self.dataset_name == '3dpw':
            kp_2d = convert_kps(self.db['joints2D'][start_index:end_index + 1], src='common', dst='spin')
            kp_3d = self.db['joints3D'][start_index:end_index + 1]
        elif self.dataset_name == 'mpii3d':
            kp_2d = self.db['joints2D'][start_index:end_index + 1]
            if is_train:
                kp_3d = self.db['joints3D'][start_index:end_index + 1]
            else:
                kp_3d = convert_kps(self.db['joints3D'][start_index:end_index + 1], src='spin', dst='common')
        elif self.dataset_name == 'h36m':
            kp_2d = self.db['joints2D'][start_index:end_index + 1]
            if is_train:
                kp_3d = self.db['joints3D'][start_index:end_index + 1]
            else:
                kp_3d = convert_kps(self.db['joints3D'][start_index:end_index + 1], src='spin', dst='common')

        kp_2d_tensor = np.ones((self.seqlen, 49, 3), dtype=np.float16)
        nj = 14 if not is_train else 49
        kp_3d_tensor = np.zeros((self.seqlen, nj, 3), dtype=np.float16)


        if self.dataset_name == '3dpw':
            pose  = self.db['pose'][start_index:end_index+1]
            shape = self.db['shape'][start_index:end_index+1]
            w_smpl = torch.ones(self.seqlen).float()
            w_3d = torch.ones(self.seqlen).float()
        elif self.dataset_name == 'h36m':
            if not is_train:
                pose = np.zeros((kp_2d.shape[0], 72))
                shape = np.zeros((kp_2d.shape[0], 10))
                w_smpl = torch.zeros(self.seqlen).float()
                w_3d = torch.ones(self.seqlen).float()
            else:
                pose = self.db['pose'][start_index:end_index + 1]
                shape = self.db['shape'][start_index:end_index + 1]
                w_smpl = torch.ones(self.seqlen).float()
                w_3d = torch.ones(self.seqlen).float()
        elif self.dataset_name == 'mpii3d':
            pose = np.zeros((kp_2d.shape[0], 72))
            shape = np.zeros((kp_2d.shape[0], 10))
            w_smpl = torch.zeros(self.seqlen).float()
            w_3d = torch.ones(self.seqlen).float()

        bbox = self.db['bbox'][start_index:end_index + 1]
        #input = torch.from_numpy(self.db['features'][start_index:end_index+1]).float()

        img_names = self.db['img_name'][start_index:end_index + 1]
        flows = self.db['flow'][start_index:end_index + 1]

        img_tensor = []
        flow_tensor = []
        input_tensor = []

        scale = 1.3

        for i in range(len(img_names)):
            image_name = img_names[i]
            image_name = '/data/anu_hpeof/vibe-flow/VIBE' + image_name[1:]
            # print(image_name, flush=True)
            box = bbox[i]
            # try:
            image = cv2.imread(image_name)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # except AttributeError:
            #    print(f'Not find \'{image_name}\'', flush=True)

            crop_image, trans = generate_patch_image_cv(
                cvimg=image.copy(),
                c_x=box[0],
                c_y=box[1],
                bb_width=box[2],
                bb_height=box[3],
                patch_width=224,
                patch_height=224,
                do_flip=False,
                scale=scale,
                rot=0,
            )

            sx = trans[1, 1]
            sy = trans[0, 0]

            flow_name = flows[i]
            if flow_name != 'null':
                flow_name = '/data/anu_hpeof/vibe-flow/VIBE' + flow_name[1:]
                flow = np.load(flow_name)[0].transpose(1, 2, 0)
                flow_patch = cv2.warpAffine(flow, trans, (224, 224), flags=cv2.INTER_LINEAR,
                                            borderMode=cv2.BORDER_CONSTANT)
                flow_patch = torch.from_numpy(flow_patch)
                flow_patch = flow_patch.permute(2, 0, 1)
                flow_patch = flow_patch * torch.tensor([sx, sy], device=flow_patch.device).view(2, 1, 1)
            else:
                flow_patch = torch.zeros(2, 224, 224)

            crop_image = convert_cvimg_to_tensor(crop_image)

            img_tensor.append(crop_image)
            flow_tensor.append(flow_patch)
            input_tensor.append(torch.cat((crop_image, flow_patch), dim=0))

        theta_tensor = np.zeros((self.seqlen, 85), dtype=np.float16)

        for idx in range(self.seqlen):
            # crop image and transform 2d keypoints
            kp_2d[idx,:,:2], trans = transfrom_keypoints(
                kp_2d=kp_2d[idx,:,:2],
                center_x=bbox[idx,0],
                center_y=bbox[idx,1],
                width=bbox[idx,2],
                height=bbox[idx,3],
                patch_width=224,
                patch_height=224,
                do_augment=False,
            )

            kp_2d[idx,:,:2] = normalize_2d_kp(kp_2d[idx,:,:2], 224)

            # theta shape (85,)
            theta = np.concatenate((np.array([1., 0., 0.]), pose[idx], shape[idx]), axis=0)

            kp_2d_tensor[idx] = kp_2d[idx]
            theta_tensor[idx] = theta
            kp_3d_tensor[idx] = kp_3d[idx]

        target = {
            'feature': torch.stack(input_tensor, dim=0),
            'flow': torch.stack(flow_tensor, dim=0),
            'theta': torch.from_numpy(theta_tensor).float(), # camera, pose and shape
            'kp_2d': torch.from_numpy(kp_2d_tensor).float(), # 2D keypoints transformed according to bbox cropping
            'kp_3d': torch.from_numpy(kp_3d_tensor).float(), # 3D keypoints
            'w_smpl': w_smpl,
            'w_3d': w_3d,
        }

        if self.dataset_name == 'mpii3d' and not is_train:
            target['valid'] = self.db['valid_i'][start_index:end_index+1]

        if self.dataset_name == '3dpw' and not is_train:
            vn = self.db['vid_name'][start_index:end_index + 1]
            fi = self.db['frame_id'][start_index:end_index + 1]
            target['instance_id'] = [f'{v}/{f}'for v,f in zip(vn,fi)]



        # if self.dataset_name == '3dpw' and not self.is_train:
            # target['imgname'] = self.db['img_name'][start_index:end_index+1].tolist()
            # target['imgname'] = np.array(target['imgname'])
            # print(target['imgname'].dtype)
            # target['center'] = self.db['bbox'][start_index:end_index+1, :2]
            # target['valid'] = torch.from_numpy(self.db['valid'][start_index:end_index+1])

        if self.debug:
            from lib.data_utils.img_utils import get_single_image_crop

            if self.dataset_name == 'mpii3d':
                video = self.db['img_name'][start_index:end_index+1]
                # print(video)
            elif self.dataset_name == 'h36m':
                video = self.db['img_name'][start_index:end_index + 1]
            else:
                vid_name = self.db['vid_name'][start_index]
                vid_name = '_'.join(vid_name.split('_')[:-1])
                f = osp.join(self.folder, 'imageFiles', vid_name)
                video_file_list = [osp.join(f, x) for x in sorted(os.listdir(f)) if x.endswith('.jpg')]
                frame_idxs = self.db['frame_id'][start_index:end_index + 1]
                # print(f, frame_idxs)
                video = [video_file_list[i] for i in frame_idxs]

            video = torch.cat(
                [get_single_image_crop(image, bbox).unsqueeze(0) for image, bbox in zip(video, bbox)], dim=0
            )

            target['video'] = video

        return target





