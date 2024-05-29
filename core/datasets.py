# Data loading based on https://github.com/NVIDIA/flownet2-pytorch

import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F
from loguru import logger as logging
import os
import copy
import random
from glob import glob
import os.path as osp
import cv2
from PIL import Image

from core.utils import frame_utils
from core.utils.augmentor import FlowAugmentor, SparseFlowAugmentor, Augmentor

class StereoDataset(data.Dataset):
    def __init__(self, aug_params=None, sparse=False, reader=None, args=None, kitti_completion=False,
                 is_vkitti2=False, is_MS2=False):
        self.augmentor = None
        self.sparse = sparse
        self.img_pad = aug_params.pop("img_pad", None) if aug_params is not None else None
        self.kitti_completion = kitti_completion
        self.is_vkitti2 = is_vkitti2
        self.is_MS2 = is_MS2

        if aug_params is not None and "crop_size" in aug_params:
            if sparse:
                self.augmentor = SparseFlowAugmentor(args, **aug_params)
            else:
                self.augmentor = FlowAugmentor(args, **aug_params)

        self.disparity_reader = reader

        self.is_test = False
        self.init_seed = False
        self.flow_list = []
        self.disparity_list = []
        self.sparse_hint_list = []
        self.image_list = []
        self.extra_info = []
        self.obj_list = []
        self.dxy_list = []

        self.args = args

    def __getitem__(self, index):

        if self.is_test:
            img1 = frame_utils.read_gen(self.image_list[index][0])
            img2 = frame_utils.read_gen(self.image_list[index][1])
            img1 = np.array(img1).astype(np.uint8)[..., :3]
            img2 = np.array(img2).astype(np.uint8)[..., :3]
            img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
            img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
            return img1, img2, self.extra_info[index]

        if not self.init_seed:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                torch.manual_seed(1000)
                np.random.seed(1000)
                random.seed(1000)
                self.init_seed = True

        index = index % len(self.image_list)

        if not self.kitti_completion:
            disp = self.disparity_reader(self.disparity_list[index])  # positive
            if isinstance(disp, tuple):
                disp, valid = disp
            else:
                valid = disp < 512
        elif self.is_vkitti2:
            disp, valid, conversion_rate = self.disp_load_vkitti2(self.disparity_list[index])
        elif self.is_MS2:
            disp, valid, conversion_rate = self.disp_load_ms2(self.disparity_list[index], self.focal_length,
                                                              self.baseline)

        else:
            disp, valid, conversion_rate = self.disp_load_kitti_completion(self.disparity_list[index])  # sparse

        img1 = frame_utils.read_gen(self.image_list[index][0])
        img2 = frame_utils.read_gen(self.image_list[index][1])

        img1 = np.array(img1).astype(np.uint8)
        img2 = np.array(img2).astype(np.uint8)

        disp = np.array(disp).astype(np.float32)  #
        flow = np.stack([disp, np.zeros_like(disp)], axis=-1)

        '''sparse hint'''
        if self.args.guided_flag and len(self.sparse_hint_list) == 0:
            disp_hints, valid_hints = self._generate_hints_sparse(disp, self.args.hints_density, valid)
            flow_hints = np.stack([disp_hints, np.zeros_like(disp_hints)], axis=-1)
        elif self.args.guided_flag and len(
                self.sparse_hint_list) > 0 and self.kitti_completion and not self.is_MS2:
            '''kitti completion'''
            disp_hints, valid_hints, conversion_rate = self.disp_load_kitti_completion(self.sparse_hint_list[index])
            flow_hints = np.stack([disp_hints, np.zeros_like(disp_hints)], axis=-1)

        elif self.args.guided_flag and len(self.sparse_hint_list) > 0 and self.is_MS2:
            '''ms2'''
            disp_hints, valid_hints, conversion_rate = self.disp_load_ms2(self.sparse_hint_list[index],
                                                                          self.focal_length,
                                                                          self.baseline)

            flow_hints = np.stack([disp_hints, np.zeros_like(disp_hints)], axis=-1)

        # grayscale images
        if len(img1.shape) == 2:
            img1 = np.tile(img1[..., None], (1, 1, 3))
            img2 = np.tile(img2[..., None], (1, 1, 3))
        else:
            img1 = img1[..., :3]
            img2 = img2[..., :3]

        if self.augmentor is not None:
            if self.sparse and not self.args.guided_flag:
                img1, img2, flow, valid, flow_hints = self.augmentor(img1, img2, flow, valid,
                                                                     more_bottom=self.args.more_bottom)
            elif self.sparse and self.args.guided_flag:
                img1, img2, flow, valid, flow_hints = self.augmentor(img1, img2, flow, valid, flow_hints,
                                                                     more_bottom=self.args.more_bottom)
            elif not self.sparse and self.args.guided_flag:
                img1, img2, flow, flow_hints = self.augmentor(img1, img2, flow, flow_hints)
            else:
                img1, img2, flow = self.augmentor(img1, img2, flow)

        img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
        flow = torch.from_numpy(flow).permute(2, 0, 1).float()
        if self.args.guided_flag:
            flow_hints = torch.from_numpy(flow_hints).permute(2, 0, 1).float()

        if self.sparse:
            valid = torch.from_numpy(valid)
        else:
            valid = (flow[0].abs() < 512) & (flow[1].abs() < 512)

        if self.img_pad is not None:
            padH, padW = self.img_pad
            img1 = F.pad(img1, [padW] * 2 + [padH] * 2)
            img2 = F.pad(img2, [padW] * 2 + [padH] * 2)

        flow = flow[:1]


        if self.args.guided_flag and not self.kitti_completion:
            flow_hints = flow_hints[:1]
            return self.image_list[index] + [
                self.disparity_list[index]], img1, img2, flow, valid.float(), flow_hints
        elif self.args.guided_flag and self.kitti_completion:
            flow_hints = flow_hints[:1]
            conversion_rate = torch.tensor(conversion_rate).float()

            return self.image_list[index] + [
                self.disparity_list[index]], img1, img2, flow, valid.float(), flow_hints, conversion_rate
        elif not self.args.guided_flag and self.kitti_completion:
            conversion_rate = torch.tensor(conversion_rate).float()
            return self.image_list[index] + [
                self.disparity_list[index]], img1, img2, flow, valid.float(), conversion_rate

        return self.image_list[index] + [self.disparity_list[index]], img1, img2, flow, valid.float()

    def __mul__(self, v):
        copy_of_self = copy.deepcopy(self)
        copy_of_self.flow_list = v * copy_of_self.flow_list
        copy_of_self.image_list = v * copy_of_self.image_list
        copy_of_self.disparity_list = v * copy_of_self.disparity_list
        copy_of_self.extra_info = v * copy_of_self.extra_info
        return copy_of_self

    def __len__(self):
        return len(self.image_list)

    def disp_load_kitti_completion(self, filename):

        data = Image.open(filename)
        w, h = data.size
        baseline = 0.54
        width_to_focal = dict()
        width_to_focal[1242] = 721.5377
        width_to_focal[1241] = 718.856
        width_to_focal[1224] = 707.0493
        width_to_focal[1226] = 708.2046
        width_to_focal[1238] = 718.3351

        data = np.array(data, dtype=np.float32) / 256.  # depth,m

        conversion_rate = width_to_focal[w] * baseline

        data[data > 0.01] = conversion_rate / (data[data > 0.01])  # disp
        data[data < 0.01] = 0
        valid_hint = (data > 0.1)
        data = data * valid_hint
        return data, valid_hint, conversion_rate

    def disp_load_ms2(self, filename, focal, baseline):
        data = Image.open(filename)
        w, h = data.size

        data = np.array(data, dtype=np.float32) / 256.  # depth,m
        conversion_rate = focal * baseline

        data[data > 0.01] = conversion_rate[0] / (data[data > 0.01])  # disp
        data[data < 0.01] = 0
        valid_hint = (data > 0.1)
        data = data * valid_hint
        return data, valid_hint, conversion_rate[0]

    def disp_load_vkitti2(self, filename):
        depth = cv2.imread(filename, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        depth = (depth / 100).astype(np.float32)

        valid = (depth > 0) & (depth < 655)
        focal_length = 725.0087
        baseline = 0.532725

        disp = baseline * focal_length / depth

        disp[~valid] = 0.

        return disp, valid, baseline * focal_length

    def _generate_hints_sparse(self, disp_gt, hints_density, valid_gt):
        np.random.seed(1000)
        random.seed(1000)
        mask = np.zeros((valid_gt.shape[-2], valid_gt.shape[-1]))
        nonzero_indeces = np.nonzero(valid_gt)
        num = int(valid_gt.shape[-2] * valid_gt.shape[-1] * hints_density)
        selected_indices = np.random.choice(nonzero_indeces[0].size, size=num, replace=False)
        mask[nonzero_indeces[0][selected_indices], nonzero_indeces[1][selected_indices]] = 1
        valid_hints = valid_gt & (mask > 0)
        hints = np.where(valid_hints > 0, disp_gt, np.zeros_like(disp_gt))
        return hints, valid_hints


class KITTI_completion(StereoDataset):
    def __init__(self, aug_params=None, root='datasets/kitti_mono', image_set='training',
                 args=None):
        super(KITTI_completion, self).__init__(aug_params, sparse=True, reader=frame_utils.readDispKITTI, args=args,
                                               kitti_completion=True)
        assert os.path.exists(root)

        train_seq = sorted(glob(os.path.join(root, 'train/*')))
        val_seq = sorted(glob(os.path.join(root, 'val/*')))

        split = 'train' if image_set == 'training' else 'val'

        use_seqs = train_seq if image_set == 'training' else val_seq
        disp_list = []
        for seq in sorted(use_seqs):
            seq_name = seq.split('/')[-1]
            disp_list_tmp = sorted(
                glob(os.path.join(root, f'{split}/{seq_name}/*/groundtruth/image_02/*.png')))  # gt
            disp_list += disp_list_tmp

        image1_list, image2_list = [], []
        sparse_hint_list = []

        velodyne_raw_dir = 'velodyne_raw'

        for samp in disp_list:
            sample_name = samp.split('/')[3]
            img_num = samp.split('/')[-1].split('.')[0]
            image1_list.append(os.path.join(root, f'{sample_name[:10]}/{sample_name}/image_02/data/{img_num}.png'))
            image2_list.append(os.path.join(root, f'{sample_name[:10]}/{sample_name}/image_03/data/{img_num}.png'))
            sparse_hint_list.append(samp.replace('groundtruth', velodyne_raw_dir))

        if image_set == 'val':
            state = np.random.get_state()
            np.random.seed(1000)
            val_num = 300
            val_idxs = set(np.random.permutation(len(disp_list))[:val_num])
            np.random.set_state(state)
            for idx, (img1, img2, disp, sparse_hint) in enumerate(
                    zip(image1_list, image2_list, disp_list, sparse_hint_list)):
                if idx in val_idxs:
                    self.image_list += [[img1, img2]]
                    self.disparity_list += [disp]
                    self.sparse_hint_list += [sparse_hint]
        elif image_set == 'test':
            state = np.random.get_state()
            np.random.seed(1000)
            val_idxs = set(np.random.permutation(len(disp_list))[:])
            np.random.set_state(state)
            for idx, (img1, img2, disp, sparse_hint) in enumerate(
                    zip(image1_list, image2_list, disp_list, sparse_hint_list)):
                if idx in val_idxs:
                    self.image_list += [[img1, img2]]
                    self.disparity_list += [disp]
                    self.sparse_hint_list += [sparse_hint]

        else:
            for idx, (img1, img2, disp, sparse_hint) in enumerate(
                    zip(image1_list, image2_list, disp_list, sparse_hint_list)):
                self.image_list += [[img1, img2]]
                self.disparity_list += [disp]
                self.sparse_hint_list += [sparse_hint]


class VKITTI2(StereoDataset):
    def __init__(self, aug_params=None, root='datasets/vkitti2', image_set='training',
                 args=None, single_scena=True):
        super(VKITTI2, self).__init__(aug_params, sparse=True, reader=frame_utils.readDispKITTI, args=args,
                                      kitti_completion=True, is_vkitti2=True)

        assert os.path.exists(root)

        train_seq_list = ['Scene01', 'Scene02']
        test_seq_list = ['Scene06', 'Scene18', 'Scene20']

        scena = '15-deg-left' if single_scena else '*'

        use_seqs = train_seq_list if image_set == 'training' else test_seq_list

        img_left_list = []
        img_right_list = []
        disp_gt_list = []

        for seq in use_seqs:
            img_left_list += sorted(glob(root + f'/{seq}/{scena}/frames/rgb/Camera_0/rgb*.jpg'))

        for samp in img_left_list:
            img_right_list.append(samp.replace('Camera_0', 'Camera_1'))
            disp_gt_list.append(samp.replace('rgb', 'depth').replace('jpg', 'png'))

        if image_set == 'val':
            state = np.random.get_state()
            np.random.seed(1000)
            val_idxs = set(np.random.permutation(len(img_left_list))[:300])
            np.random.set_state(state)

            for idx, (img1, img2, disp) in enumerate(
                    zip(img_left_list, img_right_list, disp_gt_list)):
                if idx in val_idxs:
                    self.image_list += [[img1, img2]]
                    self.disparity_list += [disp]
        else:
            for idx, (img1, img2, disp) in enumerate(
                    zip(img_left_list, img_right_list, disp_gt_list)):
                self.image_list += [[img1, img2]]
                self.disparity_list += [disp]


class MS2(StereoDataset):
    def __init__(self, aug_params=None, root='datasets/MS2', image_set='training',
                 args=None):
        super(MS2, self).__init__(aug_params, sparse=True, reader=frame_utils.readDepthMS2, args=args,
                                  kitti_completion=True, is_vkitti2=False, is_MS2=True)

        assert os.path.exists(root)

        train_seq_list = ['_2021-08-06-11-23-45',  # urban
                          '_2021-08-13-16-14-48',  # Residential
                          '_2021-08-13-16-31-10',  # road1
                          '_2021-08-13-17-06-04',  # campus
                          ]
        test_seq_list = ['_2021-08-13-16-08-46',  # road3
                         ]

        use_seqs = train_seq_list if image_set == 'training' else test_seq_list

        img_left_list = []
        img_right_list = []
        disp_gt_list = []
        sparse_hint_list = []

        calib_path_list = []
        calib_list = []

        for seq in use_seqs:

            img_left_list += sorted(glob(root + f'/sync_data/{seq}/rgb/img_left/*.png'))
            calib_path_list += osp.join(root, 'sync_data', seq, 'calib.npy')
            calib_path = osp.join(root, 'sync_data', seq, 'calib.npy')
            calib_list += [np.load(calib_path, allow_pickle=True).item()]

        if image_set == 'training':
            img_left_list = img_left_list[::3]
        else:
            img_left_list = img_left_list[::2]

        intrinsics = calib_list[0]['K_rgbL'].astype(np.float32)
        baseline = abs(calib_list[0]['T_rgbR'][0].astype(np.float32)) * 0.001  # convert to the meter scale
        self.focal_length = intrinsics[0, 0]
        self.baseline = baseline

        for samp in img_left_list:
            img_right_list.append(samp.replace('img_left', 'img_right'))
            disp_gt_list.append(
                samp.replace('sync_data', 'proj_depth').replace('img_left', 'depth_filtered'))  # filtered
            sparse_hint_list.append(samp.replace('sync_data', 'proj_depth').replace('img_left', 'depth'))  # filtered

        if image_set == 'val':
            state = np.random.get_state()
            np.random.seed(1000)
            val_idxs = set(np.random.permutation(len(img_left_list))[:300])
            np.random.set_state(state)

            for idx, (img1, img2, disp, sparse_hint) in enumerate(
                    zip(img_left_list, img_right_list, disp_gt_list, sparse_hint_list)):
                if idx in val_idxs:
                    self.image_list += [[img1, img2]]
                    self.disparity_list += [disp]
                    self.sparse_hint_list += [sparse_hint]
        else:
            for idx, (img1, img2, disp, sparse_hint) in enumerate(
                    zip(img_left_list, img_right_list, disp_gt_list, sparse_hint_list)):
                self.image_list += [[img1, img2]]
                self.disparity_list += [disp]
                self.sparse_hint_list += [sparse_hint]




def fetch_dataloader(args):
    """ Create the data loader for the corresponding trainign set """

    aug_params = {'crop_size': args.image_size, 'min_scale': args.spatial_scale[0], 'max_scale': args.spatial_scale[1],
                  'do_flip': False, 'yjitter': not args.noyjitter}
    if hasattr(args, "do_flip") and args.do_flip is not None:
        aug_params["do_flip"] = args.do_flip

    train_dataset = None
    for dataset_name in args.train_datasets:

        if 'kitti_completion' in dataset_name:

            new_dataset = KITTI_completion(aug_params, image_set='training', args=args)
            logging.info(f"Adding {len(new_dataset)} samples from KITTI_completion")
        elif 'vkitti2' in dataset_name:
            new_dataset = VKITTI2(aug_params, image_set='training', args=args, single_scena=True)
            logging.info(f"Adding {len(new_dataset)} samples from vkitti2")
        elif 'ms2' in dataset_name:
            new_dataset = MS2(aug_params, image_set='training', args=args)
            logging.info(f"Adding {len(new_dataset)} samples from MS2")

        train_dataset = new_dataset if train_dataset is None else train_dataset + new_dataset

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset,
            num_replicas=torch.cuda.device_count(),
            rank=args.local_rank)
    else:
        train_sampler = None

    shuffle = False if args.distributed else True

    train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size,
                                   pin_memory=True, shuffle=shuffle,
                                   num_workers=args.num_workers, drop_last=True,
                                   sampler=train_sampler)  # int(os.environ.get('SLURM_CPUS_PER_TASK', 6)) - 2


    return train_loader, train_sampler
