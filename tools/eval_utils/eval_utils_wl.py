import os
import torch
import pickle
import pathlib
import argparse
import numpy as np
from tqdm import tqdm
from pcdet.utils import box_utils
from pcdet.utils import calibration_kitti
from pcdet.utils import calibration_waymo
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

def rotz(angle):
    c = np.cos(angle)
    s = np.sin(angle)
    rotz = np.array([
        [c, -s, 0],
        [s,  c, 0],
        [0,  0, 1],
    ])
    return rotz

def roty(angle):
    c = np.cos(angle)
    s = np.sin(angle)
    roty = np.array([
        [ c, 0, s],
        [ 0, 1, 0],
        [-s, 0, c],
    ])
    return roty

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

def test_one_epoch(model, dataloader, eval_output_dir, logger):
    dataset_name = model.cfg.DATA_PATH.split('/')[-1]
    final_output_dir = eval_output_dir / dataset_name / 'labels'

    if final_output_dir.exists():
        logger.info(f'Removing (existed) label path {final_output_dir}.')
        os.system(f'rm -rf {final_output_dir}')
    final_output_dir.mkdir(parents=True, exist_ok=True)

    eval_iou2d = 0.0
    eval_iou3d = 0.0
    eval_seg_acc = 0.0
    eval_iou3d_acc = 0.0
    
    n_samples = 0
    with tqdm(dataloader, desc='Eval') as t:
        for data in t:
            model.eval()
            img, pts, box, theta_wl, calib_file, frame_id, mask_label, center_label, \
                heading_class_label, heading_residual_label, size_class_label, size_residual_label = data
            n_samples += pts.shape[0]
            pts = pts.transpose(2, 1).float().cuda()
            img = img.float().cuda()
            mask_label = mask_label.float().cuda()
            center_label = center_label.float().cuda()
            heading_class_label = heading_class_label.long().cuda()
            heading_residual_label = heading_residual_label.float().cuda()
            size_class_label = size_class_label.long().cuda()
            size_residual_label = size_residual_label.float().cuda()

            output = model(img, pts, box)
            center = output['center'].cpu().detach().numpy()
            heading_scores = output['heading_scores'].cpu().detach().numpy()
            heading_residuals = output['heading_residuals'].cpu().detach().numpy()
            size_scores = output['size_scores'].cpu().detach().numpy()
            size_residuals = output['size_residuals'].cpu().detach().numpy()
            iou_pred = output['iou_pred'].cpu().detach().numpy()

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

            eval_iou2d += np.sum(iou2ds)
            eval_iou3d += np.sum(iou3ds)
            eval_iou3d_acc += np.sum(iou3ds >= 0.7)

            correct = torch.argmax(output['logits'], 2).eq(mask_label.long()).cpu().detach().numpy()
            eval_seg_acc += np.sum(correct)

            bs = output['center'].shape[0]
            heading_class = np.argmax(heading_scores, 1)
            heading_residual = np.array([heading_residuals[i, heading_class[i]] for i in range(bs)])
            size_class = np.argmax(size_scores, 1)
            size_residual = np.vstack([size_residuals[i, size_class[i], :] for i in range(bs)])
            
            for i in range(bs):
                heading_angle = class2angle(model.cfg, heading_class[i], heading_residual[i])
                size = class2size(model.cfg, size_class[i], size_residual[i])

                if model.cfg.get('SHIFT_COOR', None):
                    center[i] -= np.array(model.cfg.SHIFT_COOR, dtype=np.float32)

                if model.cfg.WEAK_LABEL.TRANSFER_TO_CENTER:
                    heading_angle += theta_wl[i]
                    if dataset_name == 'kitti':
                        calib = calibration_kitti.Calibration(calib_file[i])
                        center[i] = calib.lidar_to_rect(center[i].reshape(1, 3))
                        center[i] = np.squeeze(roty(-theta_wl[i]) @ center[i].reshape(3, 1))
                        center[i] = calib.rect_to_lidar(center[i].reshape(1, 3))
                    else:
                        raise NotImplementedError(f'Dataset {dataset_name} is not supported.')

                pred_box = np.hstack([center[i].reshape(1, 3), size.reshape(1, 3), heading_angle.reshape(1, 1)])

                if dataset_name == 'kitti':
                    calib = calibration_kitti.Calibration(calib_file[i])
                    pred_box_camera = box_utils.boxes3d_lidar_to_kitti_camera(pred_box, calib).squeeze()
                    iou = float(torch.sigmoid(torch.from_numpy(iou_pred[i])))

                    label = 'Car '
                    label += '0.00 0 0.00 0.00 0.00 0.00 0.00 '
                    label += '%.2f %.2f %.2f ' % (pred_box_camera[4], pred_box_camera[5], pred_box_camera[3])
                    label += '%.2f %.2f %.2f ' % (pred_box_camera[0], pred_box_camera[1], pred_box_camera[2])
                    label += '%.2f %.2f' % (pred_box_camera[6], iou)

                    label_path = final_output_dir / f'{frame_id[i]}.txt'
                
                    with open(label_path, 'a+') as f:
                        f.write(label + '\n')
                
                else:
                    raise NotImplementedError(f'Dataset {dataset_name} is not supported.')

            t.set_postfix({'seg_acc': eval_seg_acc / (n_samples * float(model.cfg.WEAK_LABEL.NUM_POINT)), 'iou3d': eval_iou3d / n_samples, 'recall_0.7': eval_iou3d_acc / n_samples})

    eval_seg_acc /= (n_samples * float(model.cfg.WEAK_LABEL.NUM_POINT))
    eval_iou2d /= n_samples
    eval_iou3d /= n_samples
    eval_iou3d_acc /= n_samples

    logger.info(f'[Eval] seg acc: {eval_seg_acc:.4f}')
    logger.info(f'[Eval] Box IoU (2D/3D): {eval_iou2d:.4f}/{eval_iou3d:.4f}')
    logger.info(f'[Eval] Box estimation accuracy (IoU=0.7): {eval_iou3d_acc:.4f}')
    logger.info(f'Done.')