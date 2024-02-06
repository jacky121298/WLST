import os
import sys
import glob
import torch
import argparse
import datetime
import numpy as np
import torch.distributed as dist
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import DataLoader

import _init_path
from autolabeler.ST.model import LateFusionNetwork
from autolabeler.ST.model import WeakLabelerDataset
from autolabeler.ST.model import WeakLabeler, WeakLabelerTwoStage
from autolabeler.ST.model import WeakLabelerWithImage, WeakLabelerTwoStageWithImage
from pcdet.utils import fusion_utils
from pcdet.utils import common_utils
from pcdet.datasets import build_dataloader
from pcdet.models import build_network, model_fn_decorator
from pcdet.config import cfg, log_config_to_file, cfg_from_list, cfg_from_yaml_file

def parse_config():
    # TASK: waymo -> kitti
    # CUDA_VISIBLE_DEVICES=0 python3 train_fusion.py --cfg_file cfgs/da-waymo-kitti_models/pvrcnn/late_fusion.yaml --batch_size 128 --batch_size_3d 2 --batch_size_2d3d 128 --pretrained_model ../output/da-waymo-kitti_models/pvrcnn/wl_labeler_ros/center_T_rgb_T_two/ckpt/checkpoint_epoch_80.pth --pretrained_model_M3D ../output/da-waymo-kitti_models/pvrcnn/pvrcnn_old_anchor_ros/default/ckpt/checkpoint_epoch_10.pth --extra_tag late_fusion
    # TASK: waymo -> nuscenes
    # CUDA_VISIBLE_DEVICES=0 python3 train_fusion.py --cfg_file cfgs/da-waymo-nus_models/pvrcnn/late_fusion.yaml --batch_size 128 --batch_size_3d 2 --batch_size_2d3d 128 --pretrained_model ../output/da-waymo-nus_models/pvrcnn/wl_labeler_ros/center_T_rgb_T_two/ckpt/checkpoint_epoch_80.pth --pretrained_model_M3D ../output/da-waymo-nus_models/pvrcnn/pvrcnn/default/ckpt/checkpoint_epoch_30.pth --extra_tag late_fusion
    # TASK: nuscenes -> kitti
    # CUDA_VISIBLE_DEVICES=0 python3 train_fusion.py --cfg_file cfgs/da-nuscenes-kitti_models/pvrcnn/late_fusion.yaml --batch_size 128 --batch_size_3d 2 --batch_size_2d3d 128 --pretrained_model ../output/da-nuscenes-kitti_models/pvrcnn/wl_labeler_ros/center_T_rgb_T_two/ckpt/checkpoint_epoch_80.pth --pretrained_model_M3D ../output/da-nuscenes-kitti_models/pvrcnn/pvrcnn_old_anchor_ros/default/ckpt/checkpoint_epoch_50.pth --extra_tag late_fusion
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config for training')
    parser.add_argument('--batch_size', type=int, default=128, required=False, help='batch size for training')
    parser.add_argument('--batch_size_3d', type=int, default=4, required=False, help='batch size for training')
    parser.add_argument('--batch_size_2d3d', type=int, default=128, required=False, help='batch size for training')
    parser.add_argument('--epochs', type=int, default=100, required=False, help='number of epochs to train for')
    parser.add_argument('--workers', type=int, default=4, help='number of workers for dataloader')
    parser.add_argument('--extra_tag', type=str, default='default', help='extra tag for this experiment')
    parser.add_argument('--pretrained_model', type=str, default=None, help='pretrained_model')
    parser.add_argument('--pretrained_model_M3D', type=str, default=None, help='pretrained_model_M3D')
    parser.add_argument('--fix_random_seed', action='store_true', default=False, help='fix_random_seed')
    parser.add_argument('--dann', action='store_true', default=False, help='dann')
    parser.add_argument('--track_running_stats', default=True, help='track_running_stats')
    parser.add_argument('--lr', type=float, default=1e-3, help='Initial learning rate [default: 0.001].')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight Decay of Adam [default: 1e-4].')
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER, help='set extra config keys if needed')
    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])  # remove 'cfgs' and 'xxxx.yaml'

    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)
    return args, cfg

def main():
    args, cfg = parse_config()
    if args.fix_random_seed:
        common_utils.set_random_seed(seed=10922081)

    output_dir = cfg.ROOT_DIR / 'output' / cfg.EXP_GROUP_PATH / cfg.TAG / args.extra_tag
    ckpt_dir = output_dir / 'ckpt'
    ps_label_dir = output_dir / 'ps_label'

    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ps_label_dir.mkdir(parents=True, exist_ok=True)

    log_file = output_dir / ('log_train_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)

    # Log to file
    logger.info('**********************Start logging**********************')
    gpu_list = os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ.keys() else 'ALL'
    logger.info('CUDA_VISIBLE_DEVICES=%s' % gpu_list)

    for key, val in vars(args).items():
        logger.info('{:16} {}'.format(key, val))
    log_config_to_file(cfg, logger=logger)
    if cfg.LOCAL_RANK == 0:
        os.system('cp %s %s' % (args.cfg_file, output_dir))

    # Model
    if cfg.MODEL_WL.USE_IMG:
        if cfg.MODEL_WL.TWO_STAGE:
            model_M2D3D = WeakLabelerTwoStageWithImage(cfg.DATA_CONFIG, n_channel=cfg.MODEL_WL.CHANNEL, track_running_stats=args.track_running_stats, dann=args.dann).cuda()
        else:
            model_M2D3D = WeakLabelerWithImage(cfg.DATA_CONFIG, n_channel=cfg.MODEL_WL.CHANNEL, track_running_stats=args.track_running_stats, dann=args.dann).cuda()
    else:
        if cfg.MODEL_WL.TWO_STAGE:
            model_M2D3D = WeakLabelerTwoStage(cfg.DATA_CONFIG, n_channel=cfg.MODEL_WL.CHANNEL, track_running_stats=args.track_running_stats).cuda()
        else:
            model_M2D3D = WeakLabeler(cfg.DATA_CONFIG, n_channel=cfg.MODEL_WL.CHANNEL, track_running_stats=args.track_running_stats).cuda()
    logger.info(model_M2D3D)

    source_set, source_loader, source_sampler = build_dataloader(
            dataset_cfg=cfg.DATA_CONFIG,
            class_names=cfg.CLASS_NAMES,
            batch_size=args.batch_size_3d,
            dist=False,
            workers=args.workers,
            logger=logger,
            training=True,
            merge_all_iters_to_one_epoch=False,
            total_epochs=args.epochs
        )
    model_M3D = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=source_set).cuda()
    model_fusion = LateFusionNetwork(n_channel=cfg.LATE_FUSION.CHANNEL).cuda()

    # Load pretrained model
    logger.info(f'Load model M2D3D from: {args.pretrained_model}')
    checkpoint = torch.load(args.pretrained_model)
    model_M2D3D.load_state_dict(checkpoint['model_state_dict'])
    model_M2D3D.fix_seg()

    logger.info(f'Load model M3D from: {args.pretrained_model_M3D}')
    model_M3D.load_params_from_file(filename=args.pretrained_model_M3D, to_cpu=dist, logger=logger, state_name='model_state')

    ps_M2D3D_path = ps_label_dir / 'ps_label_M2D3D.pkl'
    if os.path.exists(ps_M2D3D_path):
        fusion_utils.load_M2D3D(ps_M2D3D_path)
    else:
        logger.info('Loading WeakLabeler dataset')
        dataset = WeakLabelerDataset(cfg.DATA_CONFIG, source_loader, use_rgb=cfg.MODEL_WL.USE_RGB, training=True, fusion_dataset=True)
        dataloader = DataLoader(dataset, batch_size=args.batch_size_2d3d, shuffle=True)
        logger.info('Total samples for WeakLabeler dataset: %d' % (len(dataset)))

        # Preparing M2D3D input data
        logger.info('Preparing M2D3D input data')
        fusion_utils.prepare_M2D3D_epoch(model_M2D3D, dataloader, cfg, ps_label_dir)

    ps_M3D_path = ps_label_dir / 'ps_label_M3D.pkl'
    if os.path.exists(ps_M3D_path):
        fusion_utils.load_M3D(ps_M3D_path)
    else:
        _, dataloader_M3D, _ = build_dataloader(
            dataset_cfg=cfg.DATA_CONFIG_3D,
            class_names=cfg.DATA_CONFIG_3D.CLASS_NAMES,
            batch_size=args.batch_size_3d,
            dist=False,
            workers=args.workers,
            logger=logger,
            training=True,
            merge_all_iters_to_one_epoch=False,
            total_epochs=args.epochs
        )

        # Preparing M3D input data
        logger.info('Preparing M3D input data')
        fusion_utils.prepare_M3D_epoch(model_M3D, dataloader_M3D, cfg, ps_label_dir)
    
    # Start training
    logger.info('**********************Start training %s/%s(%s)**********************'
        % (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))

    # Prepare fusion dataset
    fusion_dataset = fusion_utils.FusionDataset(cfg, ps_label_dir)
    fusion_dataloader = DataLoader(fusion_dataset, batch_size=args.batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model_fusion.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    for epoch in range(args.epochs):
        fusion_utils.train_fusion_epoch(model_fusion, optimizer, fusion_dataloader, epoch, args.epochs, cfg, args.batch_size, logger, ckpt_dir, ps_label_dir)

    logger.info('**********************End training %s/%s(%s)**********************\n\n\n'
        % (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))

if __name__ == '__main__':
    main()