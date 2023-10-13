import os
import sys
import glob
import torch
import argparse
import datetime
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import DataLoader

import _init_path
from autolabeler.ST.model import WeakLabelerDataset
from autolabeler.ST.model import PointNetInstanceSeg
from pcdet.utils import common_utils
from pcdet.datasets import build_dataloader
from pcdet.config import cfg, log_config_to_file, cfg_from_list, cfg_from_yaml_file
from train_utils.train_utils_wl import train_model

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config for training')
    parser.add_argument('--batch_size', type=int, default=128, required=False, help='batch size for training')
    parser.add_argument('--epochs', type=int, default=100, required=False, help='number of epochs to train for')
    parser.add_argument('--workers', type=int, default=4, help='number of workers for dataloader')
    parser.add_argument('--extra_tag', type=str, default='default', help='extra tag for this experiment')
    parser.add_argument('--ckpt', type=str, default=None, help='checkpoint to start from')
    parser.add_argument('--fix_random_seed', action='store_true', default=False, help='fix random seed')
    parser.add_argument('--seed', type=int, default=10922081, help='random seed')
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

def train_seg_model_one_epoch(model, dataloader, optimizer, epoch, epochs, ckpt_dir, logger):
    # Record for one epoch
    train_total_loss = 0.0
    train_seg_acc = 0.0
    
    n_samples = 0
    with tqdm(dataloader, desc=f'Epoch {epoch + 1:02d}') as t:
        for i, data in enumerate(t):
            model.train()
            img, pts, box, mask_label, center_label, heading_class_label, \
                heading_residual_label, size_class_label, size_residual_label = data
            n_samples += pts.shape[0]
            pts = pts.transpose(2, 1).float().cuda()
            mask_label = mask_label.float().cuda()
            
            logits = model(pts) # (bs, n, 2)
            mask_loss = F.nll_loss(F.log_softmax(logits.view(-1, 2), dim=1), mask_label.view(-1).long())
            
            optimizer.zero_grad()
            mask_loss.backward()
            optimizer.step()

            # Calculate loss, seg_acc
            train_total_loss += mask_loss.item()

            correct = torch.argmax(logits, 2).eq(mask_label.long()).cpu().detach().numpy()
            train_seg_acc += np.sum(correct)
            t.set_postfix({'seg_acc': train_seg_acc / (n_samples * float(cfg.DATA_CONFIG.WEAK_LABEL.NUM_POINT)), 'loss': train_total_loss / n_samples})

    train_total_loss /= n_samples
    train_seg_acc /= (n_samples * float(cfg.DATA_CONFIG.WEAK_LABEL.NUM_POINT))

    logger.info(f'=== Epoch [{epoch + 1}/{epochs}] ===')
    logger.info(f'[Train] loss: {train_total_loss:.4f}, seg acc: {train_seg_acc:.4f}')

    # Save model
    savepath = ckpt_dir / f'checkpoint_epoch_{epoch + 1}_seg.pth'
    state = {
        'epoch': epoch + 1,
        'train_seg_acc': train_seg_acc,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(state, savepath)

def train_seg_model(model, dataloader, optimizer, cur_epoch, epochs, ckpt_dir, logger):
    for epoch in range(cur_epoch, epochs):
        train_seg_model_one_epoch(model, dataloader, optimizer, epoch, epochs, ckpt_dir, logger)
    
    logger.info(f'Done.')

def main():
    args, cfg = parse_config()
    if args.fix_random_seed:
        common_utils.set_random_seed(seed=args.seed)

    output_dir = cfg.ROOT_DIR / 'output' / cfg.EXP_GROUP_PATH / cfg.TAG / args.extra_tag
    ckpt_dir = output_dir / 'ckpt'

    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

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
    source_set, source_loader, source_sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=args.batch_size,
        dist=False,
        workers=args.workers,
        logger=logger,
        training=True,
        merge_all_iters_to_one_epoch=False,
        total_epochs=args.epochs
    )

    logger.info('Loading WeakLabeler dataset')
    dataset = WeakLabelerDataset(cfg.DATA_CONFIG, source_loader, use_rgb=cfg.MODEL_WL.USE_RGB, training=True)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    logger.info('Total samples for WeakLabeler dataset: %d' % (len(dataset)))

    # Model
    model = PointNetInstanceSeg(n_channel=cfg.MODEL_WL.CHANNEL).cuda()
    logger.info(model)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Load checkpoint if it is possible
    cur_epoch = 0
    if args.ckpt is not None:
        logger.info(f'Load model from: {args.ckpt}')
        checkpoint = torch.load(args.ckpt)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        cur_epoch = checkpoint['epoch']
        
    # Start training
    logger.info('**********************Start training %s/%s(%s)**********************'
        % (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))
    
    torch.autograd.set_detect_anomaly(True)
    train_seg_model(model, dataloader, optimizer, cur_epoch, args.epochs, ckpt_dir, logger)

    logger.info('**********************End training %s/%s(%s)**********************\n\n\n'
        % (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))

if __name__ == '__main__':
    main()