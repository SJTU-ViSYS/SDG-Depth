from __future__ import print_function, division
import sys
import time

import torch
import torch.nn.functional as F
from tqdm import tqdm

import core.datasets as datasets
from core.utils.utils import InputPadder

sys.path.append('core')

try:
    autocast = torch.cuda.amp.autocast
except:
    # dummy autocast for PyTorch < 1.6
    class autocast:
        def __init__(self, enabled):
            pass

        def __enter__(self):
            pass

        def __exit__(self, *args):
            pass


@torch.no_grad()
def validate(model, args, mixed_prec=False, completion_split='val'):
    """ validation for SDG-Depth """

    model.eval()
    aug_params = {}

    if 'kitti_completion' == args.test_datasets:
        val_dataset = datasets.KITTI_completion(aug_params, image_set=completion_split, args=args)
    elif 'vkitti2' == args.test_datasets:
        val_dataset = datasets.VKITTI2(aug_params, image_set=completion_split, args=args, single_scena=True)
    elif 'ms2' == args.test_datasets:
        val_dataset = datasets.MS2(aug_params, image_set=completion_split, args=args)

    rmse_num = 0
    mae_sum, rmse_sum, imae_sum, irmse_sum = 0, 0, 0, 0

    depth_range_min = args.eval_depth_range_min
    depth_range_max = args.eval_depth_range_max

    time_count = 0

    for val_id in tqdm(range(len(val_dataset))):
        if args.guided_flag and (args.test_datasets in ['kitti_completion', 'vkitti2', 'ms2']):
            _, image1, image2, flow_gt, valid_gt, hint, conversion_rate = val_dataset[val_id]
            conversion_rate = conversion_rate[None]
            hint = hint[None].cuda()

        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        if not image1.shape[-1] in [960]:
            pad_size = 64
        else:
            pad_size = 128
        padder = InputPadder(image1.shape, divis_by=pad_size)
        image1, image2 = padder.pad(image1, image2)
        hint = padder.pad(hint)[0]

        with autocast(enabled=mixed_prec):
            sparse_mask = (hint > 0).int()
            start = time.perf_counter()
            depth_pr, _, _, _, _, _ = model(image1, image2, sparse=hint,
                                            sparse_mask=sparse_mask,
                                            conversion_rate=conversion_rate.cuda())
            end = time.perf_counter()

            time_count += end - start

    # avg_time = time_count / len(val_dataset)
    # avg_fps = 1 / avg_time

    # print(f'time: {avg_time}, fps: {avg_fps}')

        depth_pr = depth_pr[-1].unsqueeze(1)
        c_rate = conversion_rate.unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, 1, depth_pr.shape[-2],
                                                                               depth_pr.shape[-1]).cuda()
        flow_pr = c_rate / (depth_pr + 1e-6)
        flow_pr = torch.where(depth_pr == 0, torch.zeros_like(depth_pr, device=depth_pr.device), flow_pr)
        flow_pr = torch.clamp(flow_pr, min=0, max=args.max_disp)
    
        flow_pr = padder.unpad(flow_pr).cpu().squeeze(0)
        depth_pr = padder.unpad(depth_pr).cpu().squeeze(0)
    
        assert flow_pr.shape == flow_gt.shape, (flow_pr.shape, flow_gt.shape)
    
        c_rate = conversion_rate
        conversion_rate = conversion_rate.unsqueeze(1).unsqueeze(1).repeat(1, flow_pr.shape[-2],
                                                                           flow_pr.shape[-1])
        valid_tmp = (valid_gt.unsqueeze(0).bool()) & (
                flow_gt > (c_rate / 100.)) & (
                            flow_gt < args.max_disp) & (flow_gt < (c_rate / depth_range_min)) & (
                            flow_gt > (c_rate / depth_range_max)) & (flow_pr > 0) & (flow_pr > (c_rate / 100.))
        valid_tmp = valid_tmp & (depth_pr > 0)
        depth_est = depth_pr[valid_tmp]
        depth_gt = conversion_rate[valid_tmp] / flow_gt[valid_tmp]
        mae = torch.abs(depth_gt - depth_est).mean()
        rmse = torch.sqrt(F.mse_loss(depth_est, depth_gt))
        imae = torch.abs(
            flow_pr[valid_tmp] / conversion_rate[valid_tmp] - flow_gt[
                valid_tmp] / conversion_rate[valid_tmp]).mean()
        irmse = torch.sqrt(F.mse_loss(flow_pr[valid_tmp] / conversion_rate[valid_tmp],
                                      flow_gt[valid_tmp] / conversion_rate[valid_tmp]))
        mae_sum += mae
        rmse_sum += rmse
        rmse_num += 1
        imae_sum += imae
        irmse_sum += irmse
    
    mae = mae_sum / (rmse_num + 1e-6)
    rmse = rmse_sum / (rmse_num + 1e-6)
    imae = imae_sum / (rmse_num + 1e-6)
    irmse = irmse_sum / (rmse_num + 1e-6)
    print(
        f"mae:{round(mae.item(), 6)}, rmse: {round(rmse.item(), 5)}, imae: {round(imae.item(), 7)}, irmse:{round(irmse.item(), 7)}")
    return
