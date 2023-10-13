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
from autolabeler.ST.model import WeakLabeler, WeakLabelerTwoStage
from autolabeler.ST.model import WeakLabelerWithImage, WeakLabelerTwoStageWithImage
from autolabeler.ST.model import WeakLabelerLoss, WeakLabelerLossTwoStage
from autolabeler.ST.model import WeakLabelerDataset
from pcdet.utils import common_utils
from pcdet.datasets import build_dataloader
from pcdet.models import build_network, model_fn_decorator
from train_utils.optimization import build_optimizer, build_scheduler
from pcdet.config import cfg, log_config_to_file, cfg_from_list, cfg_from_yaml_file
from train_utils.train_utils_wl import train_model
from train_utils.train_st_utils_wl import train_model_st
from train_utils.train_st_utils_wl import train_model_M2D3D_st

sys.setrecursionlimit(10000)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config for training')
    parser.add_argument('--seg_model', type=str, default=None, help='pretrained segmentation model')
    parser.add_argument('--batch_size', type=int, default=128, required=False, help='batch size for training')
    parser.add_argument('--batch_size_3d', type=int, default=4, required=False, help='batch size for training')
    parser.add_argument('--step_size', type=int, default=5, required=False, help='step size for scheduler')
    parser.add_argument('--epochs', type=int, default=100, required=False, help='number of epochs to train for')
    parser.add_argument('--workers', type=int, default=4, help='number of workers for dataloader')
    parser.add_argument('--extra_tag', type=str, default='default', help='extra tag for this experiment')
    parser.add_argument('--ckpt', type=str, default=None, help='checkpoint to start from')
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

    # Dataloader
    if cfg.get('SELF_TRAIN', None):
        target_set, target_loader, target_sampler = build_dataloader(
            dataset_cfg=cfg.DATA_CONFIG_TAR,
            class_names=cfg.DATA_CONFIG_TAR.CLASS_NAMES,
            batch_size=args.batch_size_3d,
            dist=False,
            workers=args.workers,
            logger=logger,
            training=True,
            merge_all_iters_to_one_epoch=False,
            total_epochs=args.epochs
        )
    else:
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

    logger.info('Loading WeakLabeler dataset')
    if cfg.get('SELF_TRAIN', None):
        dataset = WeakLabelerDataset(cfg.DATA_CONFIG_TAR, target_loader, use_rgb=cfg.MODEL_WL.USE_RGB, training=True, fusion_dataset=True)
    else:
        dataset = WeakLabelerDataset(cfg.DATA_CONFIG, source_loader, use_rgb=cfg.MODEL_WL.USE_RGB, training=True)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    logger.info('Total samples for WeakLabeler dataset: %d' % (len(dataset)))

    if args.dann:
        target_set, target_loader, target_sampler = build_dataloader(
            dataset_cfg=cfg.DATA_CONFIG_TAR,
            class_names=cfg.DATA_CONFIG_TAR.CLASS_NAMES,
            batch_size=args.batch_size_3d,
            dist=False,
            workers=args.workers,
            logger=logger,
            training=True,
            merge_all_iters_to_one_epoch=False,
            total_epochs=args.epochs
        )
        dataset_target = WeakLabelerDataset(cfg.DATA_CONFIG_TAR, target_loader, use_rgb=cfg.MODEL_WL.USE_RGB, training=True)
        dataloader_target = DataLoader(dataset_target, batch_size=args.batch_size, shuffle=True)
    else:
        dataloader_target = None

    # Model
    if cfg.MODEL_WL.USE_IMG:
        if cfg.MODEL_WL.TWO_STAGE:
            model_M2D3D = WeakLabelerTwoStageWithImage(cfg.DATA_CONFIG, n_channel=cfg.MODEL_WL.CHANNEL, seg_model=args.seg_model, track_running_stats=args.track_running_stats, dann=args.dann).cuda()
        else:
            model_M2D3D = WeakLabelerWithImage(cfg.DATA_CONFIG, n_channel=cfg.MODEL_WL.CHANNEL, seg_model=args.seg_model, track_running_stats=args.track_running_stats, dann=args.dann).cuda()
    else:
        if cfg.MODEL_WL.TWO_STAGE:
            model_M2D3D = WeakLabelerTwoStage(cfg.DATA_CONFIG, n_channel=cfg.MODEL_WL.CHANNEL, seg_model=args.seg_model, track_running_stats=args.track_running_stats).cuda()
        else:
            model_M2D3D = WeakLabeler(cfg.DATA_CONFIG, n_channel=cfg.MODEL_WL.CHANNEL, seg_model=args.seg_model, track_running_stats=args.track_running_stats).cuda()
    logger.info(model_M2D3D)
    logger.info(f'model_M2D3D parameter size: {count_parameters(model_M2D3D)}')

    if cfg.get('SELF_TRAIN', None):
        model_M3D = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=target_set).cuda()
        model_M3D.load_params_from_file(filename=args.pretrained_model_M3D, to_cpu=dist, logger=logger, state_name='model_state')
        logger.info(f'model_M3D parameter size: {count_parameters(model_M3D)}')
        model_fusion = LateFusionNetwork(n_channel=cfg.SELF_TRAIN.LATE_FUSION.CHANNEL).cuda()
        if cfg.SELF_TRAIN.LATE_FUSION.get('MODEL_PATH', None) and cfg.SELF_TRAIN.LATE_FUSION.MODEL_PATH != 'None':
            logger.info(f'Load fusion model from: {cfg.SELF_TRAIN.LATE_FUSION.MODEL_PATH}')
            checkpoint = torch.load(cfg.SELF_TRAIN.LATE_FUSION.MODEL_PATH)
            model_fusion.load_state_dict(checkpoint['model_state_dict'])

    # Criterion
    if cfg.MODEL_WL.TWO_STAGE:
        criterion = WeakLabelerLossTwoStage(cfg.DATA_CONFIG)
    else:
        criterion = WeakLabelerLoss(cfg.DATA_CONFIG)

    # Load pretrained model
    cur_epoch = 0
    if args.pretrained_model is not None:
        logger.info(f'Load model M2D3D from: {args.pretrained_model}')
        checkpoint = torch.load(args.pretrained_model)
        model_M2D3D.load_state_dict(checkpoint['model_state_dict'])
        model_M2D3D.fix_seg()
    
    # Optimizer & Scheduler
    optimizer_M2D3D = torch.optim.Adam(filter(lambda p: p.requires_grad, model_M2D3D.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    def lr_func(epoch, init_lr=args.lr, step_size=args.step_size, gamma=0.7, eta_min=0.00001):
        f = gamma ** (epoch // step_size)
        if init_lr * f > eta_min:
            return f
        else: return eta_min / init_lr
    scheduler_M2D3D = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer_M2D3D, lr_lambda=lr_func)

    if cfg.get('SELF_TRAIN', None):
        optimizer_M3D = build_optimizer(model_M3D, cfg.OPTIMIZATION)
        lr_scheduler_M3D, lr_warmup_scheduler_M3D = build_scheduler(
            optimizer_M3D,
            total_iters_each_epoch=len(target_loader),
            total_epochs=args.epochs,
            last_epoch=-1,
            optim_cfg=cfg.OPTIMIZATION,
        )

    # Load checkpoint if it is possible
    if args.ckpt is not None:
        logger.info(f'Load model M2D3D from: {args.ckpt}')
        checkpoint = torch.load(args.ckpt)
        model_M2D3D.load_state_dict(checkpoint['model_state_dict'])
        optimizer_M2D3D.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler_M2D3D.load_state_dict(checkpoint['scheduler_state_dict'])
        cur_epoch = checkpoint['epoch']
    
    # Start training
    logger.info('**********************Start training %s/%s(%s)**********************'
        % (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))
    
    torch.autograd.set_detect_anomaly(True)
    if cfg.get('SELF_TRAIN', None):
        # train_model_M2D3D_st(model_M2D3D, dataloader, criterion, optimizer_M2D3D, scheduler_M2D3D, cur_epoch, args.epochs, ckpt_dir, ps_label_dir, logger, cfg.SELF_TRAIN)
        train_model_st(model_M2D3D, model_M3D, model_fusion, dataloader, target_loader, criterion, optimizer_M2D3D, optimizer_M3D, scheduler_M2D3D, lr_scheduler_M3D, lr_warmup_scheduler_M3D, cur_epoch, args.epochs, ckpt_dir, ps_label_dir, logger, cfg.SELF_TRAIN, model_func=model_fn_decorator(), w_iou_M2D3D=2.0, w_box_M2D3D=1.0)
    else:
        train_model(model_M2D3D, dataloader, criterion, optimizer_M2D3D, scheduler_M2D3D, cur_epoch, args.epochs, ckpt_dir, logger, dataloader_target, dann=args.dann)

    logger.info('**********************End training %s/%s(%s)**********************\n\n\n'
        % (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))

if __name__ == '__main__':
    main()