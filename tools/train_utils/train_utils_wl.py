import copy
import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from pcdet.ops.iou3d_nms import iou3d_nms_utils

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

def train_model_one_epoch(model, dataloader, criterion, optimizer, scheduler, epoch, epochs, ckpt_dir, logger, w_iou=None, w_box=None):
    # Record for one epoch
    train_total_loss = 0.0
    train_iou2d = 0.0
    train_iou3d = 0.0
    train_seg_acc = 0.0
    train_iou3d_acc = 0.0
    
    n_samples = 0
    with tqdm(dataloader, desc=f'Epoch {epoch + 1:02d}') as t:
        for i, data in enumerate(t):
            model.train()
            img, pts, box, mask_label, center_label, heading_class_label, \
                heading_residual_label, size_class_label, size_residual_label = data
            n_samples += pts.shape[0]
            pts = pts.transpose(2, 1).float().cuda()
            box = box.squeeze(1).float().cuda()
            img = img.float().cuda()
            mask_label = mask_label.float().cuda()
            center_label = center_label.float().cuda()
            heading_class_label = heading_class_label.long().cuda()
            heading_residual_label = heading_residual_label.float().cuda()
            size_class_label = size_class_label.long().cuda()
            size_residual_label = size_residual_label.float().cuda()
            
            output = model(img, pts, box)
            if w_iou != None and w_box != None:
                losses = criterion(output, mask_label, center_label, \
                    heading_class_label, heading_residual_label, size_class_label, size_residual_label, w_iou=w_iou, w_box=w_box)
            else:
                losses = criterion(output, mask_label, center_label, heading_class_label, heading_residual_label, size_class_label, size_residual_label)
            
            total_loss = losses['total_loss']
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # Calculate loss, seg_acc, iou2d, iou3d, iou3d_acc
            train_total_loss += total_loss.item()
            iou2ds, iou3ds = compute_box3d_iou(
                model.cfg,
                output['center'].cpu().detach().numpy(),
                output['heading_scores'].cpu().detach().numpy(),
                output['heading_residuals'].cpu().detach().numpy(),
                output['size_scores'].cpu().detach().numpy(),
                output['size_residuals'].cpu().detach().numpy(),
                center_label.cpu().detach().numpy(),
                heading_class_label.cpu().detach().numpy(),
                heading_residual_label.cpu().detach().numpy(),
                size_class_label.cpu().detach().numpy(),
                size_residual_label.cpu().detach().numpy(),
            )

            train_iou2d += np.sum(iou2ds)
            train_iou3d += np.sum(iou3ds)
            train_iou3d_acc += np.sum(iou3ds >= 0.7)

            correct = torch.argmax(output['logits'], 2).eq(mask_label.long()).cpu().detach().numpy()
            train_seg_acc += np.sum(correct)
            t.set_postfix({'loss': train_total_loss / n_samples, 'seg_acc': train_seg_acc / (n_samples * float(model.cfg.WEAK_LABEL.NUM_POINT)), 'iou3d': train_iou3d / n_samples, 'recall_0.7': train_iou3d_acc / n_samples})

    scheduler.step()

    train_total_loss /= n_samples
    train_seg_acc /= (n_samples * float(model.cfg.WEAK_LABEL.NUM_POINT))
    train_iou2d /= n_samples
    train_iou3d /= n_samples
    train_iou3d_acc /= n_samples

    logger.info(f'=== Epoch [{epoch + 1}/{epochs}] ===')
    logger.info(f'[Train] loss: {train_total_loss:.4f}, seg acc: {train_seg_acc:.4f}')
    logger.info(f'[Train] Box IoU (2D/3D): {train_iou2d:.4f}/{train_iou3d:.4f}')
    logger.info(f'[Train] Box estimation accuracy (IoU=0.7): {train_iou3d_acc:.4f}')

    # Save model
    savepath = ckpt_dir / f'checkpoint_epoch_{epoch + 1}.pth'
    state = {
        'epoch': epoch + 1,
        'train_iou3d_acc': train_iou3d_acc,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }
    torch.save(state, savepath)

def train_model_one_epoch_dann(model, dataloader, criterion, optimizer, scheduler, epoch, epochs, ckpt_dir, logger, dataloader_target, w_iou=1.0, w_box=2.0, w_domain=0.5):
    # Record for one epoch
    train_total_loss = 0.0
    train_iou2d = 0.0
    train_iou3d = 0.0
    train_seg_acc = 0.0
    train_iou3d_acc = 0.0
    
    dataset_length = min(len(dataloader), len(dataloader_target))
    data_source_iter = iter(dataloader)
    data_target_iter = iter(dataloader_target)

    n_samples = 0
    model.train()
    with tqdm(range(dataset_length), desc=f'Epoch {epoch + 1:02d}') as t:
        for idx in t:
            # Alpha for GradientReverse
            p = float(idx + epoch * dataset_length) / epochs / dataset_length
            alpha = 2. / (1. + np.exp(-10 * p)) - 1

            # Training model using source data
            data_source = data_source_iter.next()
            img, pts, box, mask_label, center_label, heading_class_label, \
                heading_residual_label, size_class_label, size_residual_label = data_source
            n_samples += pts.shape[0]
            pts = pts.transpose(2, 1).float().cuda()
            box = box.squeeze(1).float().cuda()
            img = img.float().cuda()
            mask_label = mask_label.float().cuda()
            center_label = center_label.float().cuda()
            heading_class_label = heading_class_label.long().cuda()
            heading_residual_label = heading_residual_label.float().cuda()
            size_class_label = size_class_label.long().cuda()
            size_residual_label = size_residual_label.float().cuda()
            domain_type = torch.zeros(pts.shape[0]).long().cuda()

            output = model(img, pts, box, alpha)
            if w_iou != None and w_box != None:
                losses_source = criterion(output, mask_label, center_label, \
                    heading_class_label, heading_residual_label, size_class_label, size_residual_label, w_iou=w_iou, w_box=w_box)
            else:
                losses_source = criterion(output, mask_label, center_label, heading_class_label, heading_residual_label, size_class_label, size_residual_label)
            
            losses_source = losses_source['total_loss']
            loss_source_domain = F.cross_entropy(output['domain_output'], domain_type)

            train_total_loss += losses_source.item()
            # Calculate loss, seg_acc, iou2d, iou3d, iou3d_acc
            iou2ds, iou3ds = compute_box3d_iou(
                model.cfg,
                output['center'].cpu().detach().numpy(),
                output['heading_scores'].cpu().detach().numpy(),
                output['heading_residuals'].cpu().detach().numpy(),
                output['size_scores'].cpu().detach().numpy(),
                output['size_residuals'].cpu().detach().numpy(),
                center_label.cpu().detach().numpy(),
                heading_class_label.cpu().detach().numpy(),
                heading_residual_label.cpu().detach().numpy(),
                size_class_label.cpu().detach().numpy(),
                size_residual_label.cpu().detach().numpy(),
            )

            train_iou2d += np.sum(iou2ds)
            train_iou3d += np.sum(iou3ds)
            train_iou3d_acc += np.sum(iou3ds >= 0.7)

            correct = torch.argmax(output['logits'], 2).eq(mask_label.long()).cpu().detach().numpy()
            train_seg_acc += np.sum(correct)
            t.set_postfix({'loss': train_total_loss / n_samples, 'seg_acc': train_seg_acc / (n_samples * float(model.cfg.WEAK_LABEL.NUM_POINT)), 'iou3d': train_iou3d / n_samples, 'recall_0.7': train_iou3d_acc / n_samples})

            # Training model using target data
            data_target = data_target_iter.next()
            img, pts, box, mask_label, center_label, heading_class_label, \
                heading_residual_label, size_class_label, size_residual_label = data_target

            pts = pts.transpose(2, 1).float().cuda()
            box = box.squeeze(1).float().cuda()
            img = img.float().cuda()
            domain_type = torch.ones(pts.shape[0]).long().cuda()

            output = model(img, pts, box, alpha)
            loss_target_domain = F.cross_entropy(output['domain_output'], domain_type)

            total_loss = losses_source + w_domain * (loss_source_domain + loss_target_domain)
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

    scheduler.step()

    train_total_loss /= n_samples
    train_seg_acc /= (n_samples * float(model.cfg.WEAK_LABEL.NUM_POINT))
    train_iou2d /= n_samples
    train_iou3d /= n_samples
    train_iou3d_acc /= n_samples

    logger.info(f'=== Epoch [{epoch + 1}/{epochs}] ===')
    logger.info(f'[Train] loss: {train_total_loss:.4f}, seg acc: {train_seg_acc:.4f}')
    logger.info(f'[Train] Box IoU (2D/3D): {train_iou2d:.4f}/{train_iou3d:.4f}')
    logger.info(f'[Train] Box estimation accuracy (IoU=0.7): {train_iou3d_acc:.4f}')

    # Save model
    savepath = ckpt_dir / f'checkpoint_epoch_{epoch + 1}.pth'
    state = {
        'epoch': epoch + 1,
        'train_iou3d_acc': train_iou3d_acc,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }
    torch.save(state, savepath)

def train_model(model, dataloader, criterion, optimizer, scheduler, cur_epoch, epochs, ckpt_dir, logger, dataloader_target, dann=False):
    for epoch in range(cur_epoch, epochs):
        if dann:
            train_model_one_epoch_dann(model, dataloader, criterion, optimizer, scheduler, epoch, epochs, ckpt_dir, logger, dataloader_target)
        else:
            train_model_one_epoch(model, dataloader, criterion, optimizer, scheduler, epoch, epochs, ckpt_dir, logger)

    logger.info(f'Done.')