import torch
import numpy as np
from tqdm import tqdm
from torch.nn.utils import clip_grad_norm_
from pcdet.utils import self_training_utils_wl
from pcdet.ops.iou3d_nms import iou3d_nms_utils
from tools.train_wlst_seg import train_seg_model_one_epoch
from tools.train_utils.train_utils_wl import train_model_one_epoch

def class2angle(cfg, pred_cls, residual, to_label_format=True):
    NUM_HEADING_BIN = cfg.WEAK_LABEL.NUM_HEADING_BIN
    angle_per_class = 2 * np.pi / float(NUM_HEADING_BIN)
    angle_center = pred_cls * angle_per_class
    angle = angle_center + residual
    if to_label_format and angle > np.pi:
        angle = angle - 2 * np.pi
    return angle

def class2size(cfg, pred_cls, residual):
    MEAN_SIZE_ARR = np.array(cfg.WEAK_LABEL.MEAN_SIZE_ARR)
    mean_size = MEAN_SIZE_ARR[pred_cls]
    return mean_size + residual

def compute_box3d_iou(cfg, center_pred, heading_logits, heading_residuals, size_logits, size_residuals, center_label, heading_class_label, heading_residual_label, size_class_label, size_residual_label):
    bs = heading_logits.shape[0]
    heading_class = np.argmax(heading_logits, 1)
    heading_residual = np.array([heading_residuals[i, heading_class[i]] for i in range(bs)])
    size_class = np.argmax(size_logits, 1)
    size_residual = np.vstack([size_residuals[i, size_class[i], :] for i in range(bs)])

    iou2d_list = []
    iou3d_list = []
    for i in range(bs):
        heading_angle = class2angle(cfg, heading_class[i], heading_residual[i])
        box_size = class2size(cfg, size_class[i], size_residual[i])
        box_pred = np.hstack([center_pred[i].reshape(1, 3), box_size.reshape(1, 3), heading_angle.reshape(1, 1)])
        box_pred = torch.from_numpy(box_pred).float().cuda()

        heading_angle_label = class2angle(cfg, heading_class_label[i], heading_residual_label[i])
        box_size_label = class2size(cfg, size_class_label[i], size_residual_label[i])
        box_label = np.hstack([center_label[i].reshape(1, 3), box_size_label.reshape(1, 3), heading_angle_label.reshape(1, 1)])
        box_label = torch.from_numpy(box_label).float().cuda()

        iou_2d = iou3d_nms_utils.boxes_iou_bev(box_pred, box_label).cpu().detach().numpy()[0, 0]
        iou_3d = iou3d_nms_utils.boxes_iou3d_gpu(box_pred, box_label).cpu().detach().numpy()[0, 0]

        if iou_2d > 1: iou_2d = 0
        if iou_3d > 1: iou_3d = 0
        
        iou2d_list.append(iou_2d)
        iou3d_list.append(iou_3d)
    
    return np.array(iou2d_list, dtype=np.float32), np.array(iou3d_list, dtype=np.float32)

def roty(angle):
    c = np.cos(angle)
    s = np.sin(angle)
    roty = np.array([
        [ c, 0, s],
        [ 0, 1, 0],
        [-s, 0, c],
    ])
    return roty

def train_model_M2D3D_st(model, dataloader, criterion, optimizer, scheduler, cur_epoch, epochs, ckpt_dir, ps_label_dir, logger, cfg, seg_epochs=10, seg_lr=1e-4, seg_weight_decay=1e-4, w_iou=1.0, w_box=1.0):
    # For continue training
    ps_pkl = self_training_utils_wl.check_already_exsit_pseudo_label_M2D3D(ps_label_dir, cur_epoch, cfg)
    if ps_pkl is not None:
        logger.info(f'Loading pseudo labels from {ps_pkl}')

    if cfg.get('PROG_AUG', None) and cfg.PROG_AUG.ENABLED and cur_epoch > 0:
        for epoch in range(cur_epoch):
            if epoch in cfg.PROG_AUG.UPDATE_AUG:
                dataloader.dataset.data_augmentor_wl.re_prepare(intensity=cfg.PROG_AUG.SCALE)
    
    for epoch in range(cur_epoch, epochs):
        # Update pseudo labels
        if (epoch in cfg.UPDATE_PSEUDO_LABEL) or (epoch % cfg.UPDATE_PSEUDO_LABEL_INTERVAL == 0):
            dataloader.dataset.eval()
            logger.info('Generating pseudo label')
            r_pseudo_label = self_training_utils_wl.save_pseudo_label_epoch_M2D3D(model, dataloader, ps_label_dir, epoch, cfg)
            dataloader.dataset.train()
            logger.info(f'{100 * r_pseudo_label:.2f}% of pseudo label saved.')

            # Train seg model
            logger.info(f'Train segmentation model for ps_label_e{epoch}')
            model.free_seg()
            seg_optimizer = torch.optim.Adam(model.ins_seg.parameters(), lr=seg_lr, weight_decay=seg_weight_decay)
            for e in range(seg_epochs):
                train_seg_model_one_epoch(model.ins_seg, dataloader, seg_optimizer, e, seg_epochs, ckpt_dir, logger)
            model.fix_seg()
            logger.info(f'Train segmentation model for ps_label_e{epoch} (end)')

        # Curriculum data augmentation (CDA)
        if cfg.get('PROG_AUG', None) and cfg.PROG_AUG.ENABLED and (epoch in cfg.PROG_AUG.UPDATE_AUG):
            dataloader.dataset.data_augmentor_wl.re_prepare(intensity=cfg.PROG_AUG.SCALE)

        # Train one epoch
        train_model_one_epoch(model, dataloader, criterion, optimizer, scheduler, epoch, epochs, ckpt_dir, logger, w_iou=w_iou, w_box=w_box)

    logger.info(f'Done.')

def train_model_M3D_one_epoch(model, dataloader, optimizer, scheduler, epoch, epochs, ckpt_dir, logger, GRAD_NORM_CLIP, model_func, accumulated_iter):
    # Record for one epoch
    train_total_loss = 0.0
    
    model.train()
    n_samples = 0
    with tqdm(dataloader, desc=f'Epoch {epoch + 1:02d}') as t:
        for data in t:
            scheduler.step(accumulated_iter)
            optimizer.zero_grad()
            st_loss, st_tb_dict, st_disp_dict = model_func(model, data)
            st_loss.backward()
            clip_grad_norm_(model.parameters(), GRAD_NORM_CLIP)
            optimizer.step()

            n_samples += 1
            train_total_loss += st_loss.item()
            t.set_postfix({'loss': train_total_loss / n_samples})
            accumulated_iter += 1
    
    train_total_loss /= n_samples
    logger.info(f'[Train] M3D loss: {train_total_loss:.4f}')
    
    # Save model
    savepath = ckpt_dir / f'checkpoint_epoch_{epoch + 1}_M3D.pth'
    state = {
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(state, savepath)
    return accumulated_iter

def train_model_st(model_M2D3D, model_M3D, model_fusion, dataloader_M2D3D, dataloader_M3D, criterion_M2D3D, optimizer_M2D3D, optimizer_M3D, scheduler_M2D3D, lr_scheduler_M3D, lr_warmup_scheduler_M3D, cur_epoch, epochs, ckpt_dir, ps_label_dir, logger, cfg, model_func, accumulated_iter=0, seg_epochs=10, seg_lr=1e-4, seg_weight_decay=1e-4, w_iou_M2D3D=1.0, w_box_M2D3D=1.0):
    # For continue training (ps_pkl)
    ps_pkl_M2D3D = self_training_utils_wl.check_already_exsit_pseudo_label_M2D3D(ps_label_dir, cur_epoch, cfg)
    if ps_pkl_M2D3D is not None:
        logger.info(f'Loading pseudo labels M2D3D from {ps_pkl_M2D3D}')
    
    ps_pkl_M3D = self_training_utils_wl.check_already_exsit_pseudo_label_M3D(ps_label_dir, cur_epoch, cfg)
    if ps_pkl_M3D is not None:
        logger.info(f'Loading pseudo labels M3D from {ps_pkl_M3D}')

    # For continue training (PROG_AUG)
    if cfg.get('PROG_AUG', None) and cfg.PROG_AUG.ENABLED and cur_epoch > 0:
        for epoch in range(cur_epoch):
            if epoch in cfg.PROG_AUG.UPDATE_AUG:
                dataloader_M2D3D.dataset.data_augmentor_wl.re_prepare(intensity=cfg.PROG_AUG.SCALE)
                dataloader_M3D.dataset.data_augmentor.re_prepare(intensity=cfg.PROG_AUG.SCALE)
    
    # Start training
    for epoch in range(cur_epoch, epochs):
        # Update pseudo labels
        if (epoch in cfg.UPDATE_PSEUDO_LABEL) or (epoch % cfg.UPDATE_PSEUDO_LABEL_INTERVAL == 0):
            ## Generating M2D3D pseudo label
            dataloader_M2D3D.dataset.eval()
            dataloader_M2D3D.dataset.fusion_dataset = True
            logger.info('Generating M2D3D pseudo label')
            r_pseudo_label = self_training_utils_wl.save_pseudo_label_epoch_M2D3D(model_M2D3D, dataloader_M2D3D, ps_label_dir, epoch, cfg)
            dataloader_M2D3D.dataset.train()
            dataloader_M2D3D.dataset.fusion_dataset = False
            logger.info(f'{100 * r_pseudo_label:.2f}% of pseudo label saved.')

            ## Generating M3D pseudo label
            dataloader_M3D.dataset.eval()
            logger.info('Generating M3D pseudo label')
            avg_pseudo_label = self_training_utils_wl.save_pseudo_label_epoch_M3D(model_M3D, dataloader_M3D, ps_label_dir, epoch, cfg)
            dataloader_M3D.dataset.train()
            logger.info(f'{avg_pseudo_label:.2f} pseudo labels (per frame) saved.')

            ## Late-fusion
            logger.info('Late-fusion pseudo label')
            self_training_utils_wl.late_fusion_epoch(ps_label_dir, epoch, cfg, model_fusion, dataloader_M3D.dataset.nusc)

            # Train M2D3D's seg model
            logger.info(f'Train segmentation model for ps_label_e{epoch}_M2D3D')
            model_M2D3D.free_seg()
            seg_optimizer = torch.optim.Adam(model_M2D3D.ins_seg.parameters(), lr=seg_lr, weight_decay=seg_weight_decay)
            for e in range(seg_epochs):
                train_seg_model_one_epoch(model_M2D3D.ins_seg, dataloader_M2D3D, seg_optimizer, e, seg_epochs, ckpt_dir, logger)
            model_M2D3D.fix_seg()
            logger.info(f'Train segmentation model for ps_label_e{epoch}_M2D3D (end)')

        # Curriculum data augmentation (CDA)
        if cfg.get('PROG_AUG', None) and cfg.PROG_AUG.ENABLED and (epoch in cfg.PROG_AUG.UPDATE_AUG):
            dataloader_M2D3D.dataset.data_augmentor_wl.re_prepare(intensity=cfg.PROG_AUG.SCALE)
            dataloader_M3D.dataset.data_augmentor.re_prepare(intensity=cfg.PROG_AUG.SCALE)

        # Train one epoch
        ## M2D3D
        train_model_one_epoch(model_M2D3D, dataloader_M2D3D, criterion_M2D3D, optimizer_M2D3D, scheduler_M2D3D, epoch, epochs, ckpt_dir, logger, w_iou=w_iou_M2D3D, w_box=w_box_M2D3D)
        ## M3D
        if lr_warmup_scheduler_M3D != None and epoch < cfg.WARMUP_EPOCH:
            lr_scheduler = lr_warmup_scheduler_M3D
        else:
            lr_scheduler = lr_scheduler_M3D
        accumulated_iter = train_model_M3D_one_epoch(model_M3D, dataloader_M3D, optimizer_M3D, lr_scheduler, epoch, epochs, ckpt_dir, logger, cfg.GRAD_NORM_CLIP, model_func, accumulated_iter)

    logger.info(f'Done.')