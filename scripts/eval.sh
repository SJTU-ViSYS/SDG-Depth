#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0 python -u stereo_lidar.py \
--checkpoint_dir checkpoints \
--first_test 1 \
--resume_ckpt premodel/model_kitti.pth \
--strict_resume 1 \
--train_datasets kitti_completion \
--test_datasets kitti_completion \
--max_disp 192 \
--image_size 256 512 \
--occlusion_aug_prob 0.5 \
--guided_flag 1 \
--cfnet_confidence_value 0.4 \
--gsm_validhint conf_04 \
--gaussian_h 2 \
--gaussian_w 8 \
--refine_spn_r 4 \
--refine_spn_resolution 4 \
--refine_spn_conf_pixel sparse_valid_all \
--refine_spn_offset_flag 1 \
--disp_to_depth_convert_flag 1 \
--disp_to_depth_convert_resolution 2 \
--disp_to_depth_convert_gate 0.1 \
--disp_to_depth_convert_disp_range 0.2 \
--disp_to_depth_convert_depth_range 0.6 \
--pred_hint_weight 0.5 \
--disp_to_depth_convert_loss_weight1 0.7 \
--disp_to_depth_convert_loss_weight2 0.7 \
--batch_size 1 \
--num_steps 268430 \
--num_epoch 25 \
--lr 1e-3 \
--lr_scheduler_type MultiStepLR \
--milestones 14,17,19,24 \
--lr_gamma 0.5