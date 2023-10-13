import os
import re
import glob
import torch
import argparse
import datetime
import numpy as np
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import DataLoader

import _init_path
from autolabeler.ST.model import WeakLabeler, WeakLabelerTwoStage
from autolabeler.ST.model import WeakLabelerWithImage, WeakLabelerTwoStageWithImage
from autolabeler.ST.model import WeakLabelerDataset
from pcdet.utils import common_utils
from pcdet.datasets import build_dataloader
from pcdet.config import cfg, log_config_to_file, cfg_from_list, cfg_from_yaml_file
from eval_utils.eval_utils_wl import test_one_epoch

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config for training')
    parser.add_argument('--batch_size', type=int, default=128, required=False, help='batch size for training')
    parser.add_argument('--ckpt', type=str, default=None, help='checkpoint to start from')
    parser.add_argument('--seg_model', type=str, default=None, help='pretrained segmentation model')
    parser.add_argument('--workers', type=int, default=4, help='number of workers for dataloader')
    parser.add_argument('--extra_tag', type=str, default='default', help='extra tag for this experiment')
    parser.add_argument('--track_running_stats', default=True, help='track_running_stats')
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER, help='set extra config keys if needed')
    parser.add_argument('--eval_tag', type=str, default='default', help='eval tag for this experiment')
    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])  # remove 'cfgs' and 'xxxx.yaml'

    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)
    return args, cfg

def main():
    args, cfg = parse_config()
    common_utils.set_random_seed(seed=10922081)
    
    output_dir = cfg.ROOT_DIR / 'output' / cfg.EXP_GROUP_PATH / cfg.TAG / args.extra_tag
    eval_output_dir = output_dir / 'eval'

    num_list = re.findall(r'\d+', args.ckpt) if args.ckpt is not None else []
    epoch_id = num_list[-1] if num_list.__len__() > 0 else 'no_number'
    eval_output_dir = eval_output_dir / ('epoch_%s' % epoch_id) / cfg.DATA_CONFIG_TAR.DATA_SPLIT['test']
    if args.eval_tag is not None:
        eval_output_dir = eval_output_dir / args.eval_tag

    output_dir.mkdir(parents=True, exist_ok=True)
    eval_output_dir.mkdir(parents=True, exist_ok=True)

    log_file = eval_output_dir / ('log_eval_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
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
    test_set, test_loader, sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG_TAR,
        class_names=cfg.DATA_CONFIG_TAR.CLASS_NAMES,
        batch_size=args.batch_size,
        dist=False,
        workers=args.workers,
        logger=logger,
        training=False,
    )
    logger.info('Loading WeakLabeler dataset')
    dataset = WeakLabelerDataset(cfg.DATA_CONFIG_TAR, test_loader, use_rgb=cfg.MODEL_WL.USE_RGB, training=False)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    logger.info('Total samples for WeakLabeler dataset: %d' % (len(dataset)))

    # Model
    if cfg.MODEL_WL.USE_IMG:
        if cfg.MODEL_WL.TWO_STAGE:
            model = WeakLabelerTwoStageWithImage(cfg.DATA_CONFIG_TAR, n_channel=cfg.MODEL_WL.CHANNEL, seg_model=args.seg_model, track_running_stats=args.track_running_stats, dann=False).cuda()
        else:
            model = WeakLabelerWithImage(cfg.DATA_CONFIG_TAR, n_channel=cfg.MODEL_WL.CHANNEL, seg_model=args.seg_model, track_running_stats=args.track_running_stats, dann=False).cuda()
    else:
        if cfg.MODEL_WL.TWO_STAGE:
            model = WeakLabelerTwoStage(cfg.DATA_CONFIG_TAR, n_channel=cfg.MODEL_WL.CHANNEL, seg_model=args.seg_model, track_running_stats=args.track_running_stats).cuda()
        else:
            model = WeakLabeler(cfg.DATA_CONFIG_TAR, n_channel=cfg.MODEL_WL.CHANNEL, seg_model=args.seg_model, track_running_stats=args.track_running_stats).cuda()
    logger.info(model)
    
    logger.info(f'Load model from: {args.ckpt}')
    checkpoint = torch.load(args.ckpt)
    model.load_state_dict(checkpoint['model_state_dict'])
    # model.load_seg(args.seg_model)
    
    # Start testing
    test_one_epoch(model, dataloader, eval_output_dir, logger)

if __name__ == '__main__':
    main()