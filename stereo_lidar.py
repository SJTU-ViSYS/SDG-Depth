from __future__ import print_function, division
import argparse
from loguru import logger as logging
import numpy as np
from pathlib import Path
import os
import random
import sys

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim

from core.sdg_depth.net import SDGDepth
from evaluate import validate
import core.datasets as datasets

try:
    from torch.cuda.amp import GradScaler
except:
    # dummy GradScaler for PyTorch < 1.6
    class GradScaler:
        def __init__(self):
            pass

        def scale(self, loss):
            return loss

        def unscale_(self, optimizer):
            pass

        def step(self, optimizer):
            optimizer.step()

        def update(self):
            pass

parser = argparse.ArgumentParser()

# Training parameters
parser.add_argument('--checkpoint_dir', default='checkpoints', help="name your experiment")
parser.add_argument('--image_size', type=int, nargs='+', default=[320, 720],
                    help="size of the random image crops used during training.")
parser.add_argument('--max_disp', type=int, default=192, help="max-disp for correlation-costVol")
parser.add_argument('--num_epoch', type=int, default=20)
parser.add_argument('--num_steps', type=int, default=300000, help="length of training schedule.")
parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
parser.add_argument('--batch_size', type=int, default=6, help="batch size used during training.")
parser.add_argument('--num_workers', type=int, default=8, help="num_workers")
parser.add_argument('--train_datasets', nargs='+', default=['kitti_completion'], help="training datasets.")
parser.add_argument('--test_datasets', default='kitti_completion', type=str)
parser.add_argument('--lr', type=float, default=0.0002, help="max learning rate.")
parser.add_argument('--lr_scheduler_type', default='MultiStepLR', type=str)
parser.add_argument('--milestones', type=str, default='15', help='milestones for MultiStepLR ')
parser.add_argument('--lr_gamma', default=0.1, type=float)
parser.add_argument('--val_epoch', default=1, type=int)

parser.add_argument('--resume_ckpt', type=str, help="restore checkpoint")
parser.add_argument('--strict_resume', default=1, type=int)
parser.add_argument('--resume_optimizer', default=1, type=int)
parser.add_argument('--resume_epoch', default=1, type=int)

parser.add_argument('--pred_hint_weight', default=0.6, type=float)
parser.add_argument('--disp_to_depth_convert_loss_weight1', default=0, type=float, help='')
parser.add_argument('--disp_to_depth_convert_loss_weight2', default=0, type=float, help='')
parser.add_argument('--disp_to_depth_convert_flag', default=0, type=int, help='')

parser.add_argument('--save_latest_ckpt_freq', default=1000, type=int, help='save model and optimizer for resume')

# Data augmentation
parser.add_argument('--img_gamma', type=float, nargs='+', default=None, help="gamma range")
parser.add_argument('--saturation_range', type=float, nargs='+', default=None, help='color saturation')
parser.add_argument('--do_flip', default=False, choices=['h', 'v'],
                    help='flip the images horizontally or vertically')
parser.add_argument('--spatial_scale', type=float, nargs='+', default=[0, 0], help='re-scale the images randomly')
parser.add_argument('--noyjitter', action='store_true', help='don\'t simulate imperfect rectification')
parser.add_argument('--occlusion_aug_prob', default=0., type=float)
parser.add_argument('--y_start', default=125, type=int, help='crop image')

# distributed training
parser.add_argument('--local_rank', default=0, type=int)
parser.add_argument('--distributed', action='store_true')
parser.add_argument('--launcher', default='none', type=str, choices=['none', 'pytorch'])
parser.add_argument('--gpu_ids', default=0, type=int, nargs='+')

# net
parser.add_argument('--guided_flag', default=0, type=int)
parser.add_argument('--hints_density', default=0.05, type=float)
parser.add_argument('--more_bottom', default=0.0, type=float)
parser.add_argument('--expand_flag', default=1, type=int)
parser.add_argument('--refine_spn_resolution', default=1, type=int, help='')
parser.add_argument('--gaussian_h', default=10, type=float)
parser.add_argument('--gaussian_w', default=1, type=float)
parser.add_argument('--cfnet_confidence_value', default=0., type=float, help='')
parser.add_argument('--gsm_validhint', default='conf_04', type=str)

# DDC module
parser.add_argument('--disp_to_depth_convert_resolution', default=1, type=int, help='')
parser.add_argument('--disp_to_depth_convert_disp_range', default=0.5, type=float, help='')
parser.add_argument('--disp_to_depth_convert_depth_range', default=1.2, type=float, help='')
parser.add_argument('--disp_to_depth_convert_gate', default=3.8, type=float, help='')

# DP_module
parser.add_argument('--refine_spn_r', default=1, type=int, help='')
parser.add_argument('--refine_spn_offset_flag', default=1, type=int, help='')
parser.add_argument('--refine_spn_offset_range', default=1, type=float, help='')
parser.add_argument('--refine_spn_conf_pixel', default='sparse_valid_any', type=str, help='')

# evaluation
parser.add_argument('--first_test', default=0, type=int)
parser.add_argument('--first_test_type', default='test', type=str)
parser.add_argument('--eval_depth_range_min', default=0, type=float)
parser.add_argument('--eval_depth_range_max', default=100., type=float)

args = parser.parse_args()


def iMAE_metric(D_est, D_gt, mask, conversion_rate, depth_est=None):
    b, c, h, w = D_est.shape
    conversion_rate = conversion_rate.unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, 1, h, w)
    if depth_est is None:
        D_est, D_gt = D_est[mask] / conversion_rate[mask], D_gt[mask] / conversion_rate[mask]
    else:
        D_est, D_gt = 1 / depth_est[mask], D_gt[mask] / conversion_rate[mask]
    E = torch.abs(D_gt - D_est)
    return torch.mean(E.float())


def iRMSE_metric(D_est, D_gt, mask, conversion_rate, depth_est=None):
    b, c, h, w = D_est.shape
    conversion_rate = conversion_rate.unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, 1, h, w)
    if depth_est is None:
        D_est, D_gt = D_est[mask] / conversion_rate[mask], D_gt[mask] / conversion_rate[mask]
    else:
        D_est, D_gt = 1 / depth_est[mask], D_gt[mask] / conversion_rate[mask]
    E = F.mse_loss(D_est, D_gt, size_average=True)
    return torch.sqrt(E)


def MAE_metric(D_est, D_gt, mask, conversion_rate, depth_est=None):
    b, c, h, w = D_est.shape
    conversion_rate = conversion_rate.unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, 1, h, w)
    if depth_est is None:
        D_est, D_gt = conversion_rate[mask] / D_est[mask], conversion_rate[mask] / D_gt[mask]
    else:
        D_est, D_gt = depth_est[mask], conversion_rate[mask] / D_gt[mask]
    D_est = D_est.clamp(max=100.0)
    E = torch.abs(D_gt - D_est)
    return torch.mean(E.float())


def RMSE_metric(D_est, D_gt, mask, conversion_rate, depth_est=None):
    b, c, h, w = D_est.shape
    conversion_rate = conversion_rate.unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, 1, h, w)
    # disp to depth
    if depth_est is None:
        D_est, D_gt = conversion_rate[mask] / D_est[mask], conversion_rate[mask] / D_gt[mask]
    else:
        D_est, D_gt = depth_est[mask], conversion_rate[mask] / D_gt[mask]
    D_est = D_est.clamp(max=100.0)
    E = F.mse_loss(D_est, D_gt, size_average=True)
    return torch.sqrt(E)


def seq_loss(disp_ests, disp_gt, mask, weight_flag='default'):
    if len(disp_ests) == 9:
        if weight_flag == 'default':
            weights = [0.5 * 0.5, 0.5 * 0.7, 0.5 * 1.0, 1 * 0.5, 1 * 0.7, 1 * 1.0, 2 * 0.5, 2 * 0.7,
                       2 * 1.0]  # 0.5,1,2,default
        elif weight_flag == 'larger':
            weights = [2, 2.3, 2.6, 2.9, 3.2, 3.5, 3.8, 4.1, 4.4]

    elif len(disp_ests) == 10:
        weights = [0.5 * 0.5, 0.5 * 0.7, 0.5 * 1.0, 1 * 0.5, 1 * 0.7, 1 * 1.0, 2 * 0.5, 2 * 0.7, 2 * 1.0,
                   2 * 1.0]  # 0.5,1,2
    elif len(disp_ests) == 11:
        weights = [0.5 * 0.5, 0.5 * 0.7, 0.5 * 1.0, 1 * 0.5, 1 * 0.7, 1 * 1.0, 2 * 0.5, 2 * 0.7, 2 * 1.0,
                   2 * 1.0, 2.3]  # 0.5,1,2
    elif len(disp_ests) == 1:
        weights = [3 * 1.0]
    else:
        weights = [0.5 * 1.0, 1 * 1.0, 2 * 1.0]  # 0.5,1,2
    all_losses = []
    for disp_est, weight in zip(disp_ests, weights):
        all_losses.append(weight * F.smooth_l1_loss(disp_est[mask], disp_gt[mask], size_average=True))
    return sum(all_losses)


class Logger:
    SUM_FREQ = 100

    def __init__(self, model, scheduler, summary_writer=None, total_steps=0):
        self.model = model
        self.scheduler = scheduler
        self.total_steps = total_steps
        self.running_loss = {}
        if summary_writer == None:
            self.writer = SummaryWriter(log_dir='runs')
        else:
            self.writer = summary_writer

    def _print_training_status(self):
        metrics_data = [self.running_loss[k] / Logger.SUM_FREQ for k in sorted(self.running_loss.keys())]
        training_str = "[{:6d}, {:10.7f}] ".format(self.total_steps + 1, self.scheduler.get_last_lr()[0])
        metrics_str = ("{:10.4f}, " * len(metrics_data)).format(*metrics_data)

        # print the training status
        logging.info(f"Training Metrics ({self.total_steps}): {training_str + metrics_str}")

        if self.writer is None:
            self.writer = SummaryWriter(log_dir='runs')

        for k in self.running_loss:
            self.writer.add_scalar(k, self.running_loss[k] / Logger.SUM_FREQ, self.total_steps)
            self.running_loss[k] = 0.0

    def push(self, metrics):
        self.total_steps += 1

        for key in metrics:
            if key not in self.running_loss:
                self.running_loss[key] = 0.0

            self.running_loss[key] += metrics[key]

        if self.total_steps % Logger.SUM_FREQ == Logger.SUM_FREQ - 1:
            self._print_training_status()
            self.running_loss = {}

    def write_dict(self, results):
        if self.writer is None:
            self.writer = SummaryWriter(log_dir='runs')

        for key in results:
            self.writer.add_scalar(key, results[key], self.total_steps)

    def close(self):
        self.writer.close()


def main(args):
    Path(args.checkpoint_dir).mkdir(exist_ok=True, parents=True)

    model = SDGDepth(args.max_disp, use_concat_volume=True, args=args).cuda()
    model = torch.nn.DataParallel(model)
    model_without_ddp = model.module

    train_loader, train_sampler = datasets.fetch_dataloader(args)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))

    start_epoch = 0
    start_step = 0

    if args.resume_ckpt is not None and args.local_rank == 0:
        assert os.path.isfile(args.resume_ckpt)
        logging.info(f"Loading checkpoint from: {args.resume_ckpt}")
        loc = 'cuda:{}'.format(args.local_rank) if torch.cuda.is_available() else 'cpu'
        checkpoint = torch.load(args.resume_ckpt, map_location=loc)
        model_without_ddp.load_state_dict(checkpoint['model'], strict=args.strict_resume)
        del checkpoint

    milestones = [int(milestone) for milestone in args.milestones.split(',')]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=args.lr_gamma)

    if args.local_rank == 0:
        train_writer = SummaryWriter(args.checkpoint_dir)
        logger = Logger(model, scheduler, train_writer, total_steps=start_step)

    scaler = GradScaler(enabled=args.mixed_precision)

    should_keep_training = True
    global_batch_num = start_step

    epoch = start_epoch
    total_steps = start_step
    time_steps = 0

    if args.local_rank == 0 and args.first_test:
        if args.resume_ckpt is not None:
            validate(model.module, args=args, completion_split=args.first_test_type)
            sys.exit()
            model.train()

    while should_keep_training:

        model.train()

        for i_batch, (_, *data_blob) in enumerate(train_loader):

            optimizer.zero_grad()

            if args.guided_flag and (args.train_datasets[0] in ['kitti_completion', 'vkitti2', 'ms2']):
                image1, image2, flow, valid, hint, conversion_rate = [x.cuda(non_blocking=True) for x in data_blob]

            flow = flow.squeeze(1)  # [b,h,w]
            mask = (flow < args.max_disp) & (valid > 0)  # [b,h,w]

            mask.detach()
            sparse_mask = (hint > 0).int()
            pred_hint, dense_confidence_out, pred_rgb_hint, pred_d_hint, stereo_disp_list, fusion_disp_list, depth = \
                model(image1, image2, sparse=hint, sparse_mask=sparse_mask, conversion_rate=conversion_rate)
            mask_tmp = (flow < args.max_disp) & (flow > 0)
            loss1 = seq_loss(stereo_disp_list, flow, mask_tmp)

            loss2 = args.pred_hint_weight * F.smooth_l1_loss((pred_hint[-1].squeeze(1))[mask],
                                                             flow[mask],
                                                             size_average=True)
            loss = loss1 + loss2
            mask_tmp = (flow < args.max_disp) & (valid > 0)
            c_rate = conversion_rate.unsqueeze(1).unsqueeze(1).repeat(1, flow.shape[-2], flow.shape[-1])
            D_gt = c_rate[mask_tmp] / flow[mask_tmp]

            loss3 = args.disp_to_depth_convert_loss_weight1 * F.smooth_l1_loss(depth[mask_tmp], D_gt,
                                                                               size_average=True) + \
                    args.disp_to_depth_convert_loss_weight2 * F.mse_loss(depth[mask_tmp], D_gt,
                                                                         size_average=True)
            loss = loss + loss3

            c_rate = conversion_rate.unsqueeze(1).unsqueeze(1).repeat(1, depth.shape[-2], depth.shape[-1])
            output3 = c_rate / (depth + 1e-6)
            output3[depth == 0] = 0.

            flow = flow.unsqueeze(1)
            epe = torch.sum((output3.unsqueeze(1) - flow) ** 2, dim=1).sqrt()
            mask = mask.unsqueeze(1)
            epe = epe.view(-1)[mask[:, 0:1, :, :].bool().contiguous().view(-1)]

            if conversion_rate is not None:
                mask = mask & (depth.unsqueeze(1) > 0)

                mae = MAE_metric(output3.unsqueeze(1), flow, mask, conversion_rate,
                                 depth_est=depth.unsqueeze(1) if args.disp_to_depth_convert_flag else None)
                rmse = RMSE_metric(output3.unsqueeze(1), flow, mask, conversion_rate,
                                   depth_est=depth.unsqueeze(1) if args.disp_to_depth_convert_flag else None)

                imae = iMAE_metric(output3.unsqueeze(1), flow, mask, conversion_rate,
                                   depth_est=depth.unsqueeze(1) if args.disp_to_depth_convert_flag else None)
                irmse = iRMSE_metric(output3.unsqueeze(1), flow, mask, conversion_rate,
                                     depth_est=depth.unsqueeze(1) if args.disp_to_depth_convert_flag else None)
                metrics = {
                    'epe': epe.mean().item(),
                    '1px': (epe < 1).float().mean().item(),
                    '3px': (epe < 3).float().mean().item(),
                    '5px': (epe < 5).float().mean().item(),
                    'mae': mae.item(),
                    'rmse': rmse.item(),
                    'imae': imae.item(),
                    'irmse': irmse.item()
                }

            if isinstance(loss, float):
                logging.info(f'{total_steps} step loss is float. skipping...')
                continue

            if torch.isnan(loss) or torch.isinf(loss):
                logging.info(f'{total_steps} step loss is NaN. skipping...')
                total_steps += 1
                continue

            if args.local_rank == 0:
                logger.writer.add_scalar("live_loss", loss.item(), global_batch_num)
            global_batch_num += 1

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            scaler.step(optimizer)
            scaler.update()

            if args.local_rank == 0:
                logger.push(metrics)

            total_steps += 1
            time_steps += 1

            if total_steps % args.save_latest_ckpt_freq == 0:
                if args.local_rank == 0:
                    save_path = Path(args.checkpoint_dir + '/ckpt_latest.pth')
                    torch.save({'model': model_without_ddp.state_dict(),
                                'optimizer': optimizer.state_dict(),
                                'step': total_steps,
                                'epoch': epoch}, save_path)

            if total_steps >= args.num_steps:
                should_keep_training = False
                break

        if args.lr_scheduler_type == 'MultiStepLR':
            scheduler.step()

        if args.local_rank == 0 and (
                (epoch + 1) % args.val_epoch == 0 or (epoch + 1) == args.num_epoch):
            logging.info(f'Save checkpoint at step: {total_steps}')
            save_path = Path(args.checkpoint_dir + '/%06d.pth' % total_steps)
            logging.info(f"Saving file {save_path.absolute()}")
            torch.save({'model': model_without_ddp.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'step': total_steps,
                        'epoch': epoch + 1}, save_path)

        if args.local_rank == 0 and ((epoch + 1) % args.val_epoch == 0 or (epoch + 1) == args.num_epoch):
            validate(model.module, args=args)
            model.train()

        epoch += 1
        if epoch >= args.num_epoch:
            should_keep_training = False
            break

    logging.info("FINISHED TRAINING")
    if args.local_rank == 0:
        logger.close()

    return


if __name__ == '__main__':
    seed = 1000
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = True

    main(args)
