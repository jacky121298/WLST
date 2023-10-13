import os
import glob
import torch
import argparse
import numpy as np
from tqdm import tqdm
from pcdet.utils import box_utils
from pcdet.utils import calibration_kitti
from pcdet.ops.iou3d_nms import iou3d_nms_utils

def get_quality(tp, num_gt, num_pred):
    recall = tp / num_gt
    precision = tp / num_pred
    if precision + recall > 0.0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0
    return recall, precision, f1

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--labels', help='Path to gt labels.')
    parser.add_argument('--pseudo', help='Path to pseudo labels.')
    parser.add_argument('--pos_thres', type=float, default=0.6, help='Positive IOU threshold.')
    args = parser.parse_args()

    with open('/tmp2/jacky1212/kitti/ImageSets/train.txt', 'r') as f:
        split = f.readlines()
    
    iou3d = []
    used_class = ['Car']
    
    num, num_gt, num_pred, tp, tp_01 = 0, 0, 0, 0, 0
    for s in tqdm(split):
        # Labels
        with open(os.path.join(args.labels, s.strip() + '.txt'), 'r') as f:
            gt = f.readlines()

        labels, dontcare = [], []
        for i in range(len(gt)):
            g = gt[i].split()
            if g[0] in ['Van', 'Truck', 'DontCare']:
                dontcare.append(np.array([g[11], g[12], g[13], g[10], g[8], g[9], g[14]]).astype(np.float))
            if g[0] not in used_class:
                continue
            # (x, y, z, l, h, w, rot)
            labels.append(np.array([g[11], g[12], g[13], g[10], g[8], g[9], g[14]]).astype(np.float))
        
        labels = np.array(labels)
        dontcare = np.array(dontcare)
        num_gt += labels.shape[0]
        
        # Pseudo
        if not os.path.exists(os.path.join(args.pseudo, s.strip() + '.txt')):
            for _ in range(labels.shape[0]):
                iou3d.append(0)
            continue

        with open(os.path.join(args.pseudo, s.strip() + '.txt'), 'r') as f:
            ps = f.readlines()
        
        pseudo = []
        for i in range(len(ps)):
            p = ps[i].split()
            if p[0] not in used_class:
                continue
            if float(p[15]) < args.pos_thres:
                continue
            # (x, y, z, l, h, w, rot)
            pseudo.append(np.array([p[11], p[12], p[13], p[10], p[8], p[9], p[14]]).astype(np.float))
        
        pseudo = np.array(pseudo)
        num_pred += pseudo.shape[0]
        num += 1

        if pseudo.shape[0] == 0:
            for _ in range(labels.shape[0]):
                iou3d.append(0)
            continue

        if labels.shape[0] == 0:
            continue
        
        calib = calibration_kitti.Calibration('/tmp2/jacky1212/kitti/training/calib/' + s.strip() + '.txt')
        pseudo = box_utils.boxes3d_kitti_camera_to_lidar(pseudo, calib)
        labels = box_utils.boxes3d_kitti_camera_to_lidar(labels, calib)
        if len(dontcare) > 0:
            dontcare = box_utils.boxes3d_kitti_camera_to_lidar(dontcare, calib)

        pseudo = torch.tensor(pseudo).float().cuda()
        labels = torch.tensor(labels).float().cuda()
        if len(dontcare) > 0:
            dontcare = torch.tensor(dontcare).float().cuda()

        iou = iou3d_nms_utils.boxes_iou3d_gpu(labels, pseudo)
        iou = iou.cpu().data.numpy()
        
        for i in iou.max(axis=1):
            assert np.around(i) <= 1.0, 'Error: IOU > 1.'
            if i >= 0.7: tp += 1
            if i >= 0.1: tp_01 += 1
            iou3d.append(i)

        if len(dontcare) > 0:
            iou = iou3d_nms_utils.boxes_iou3d_gpu(dontcare, pseudo)
            iou = iou.cpu().data.numpy()

            for i in iou.max(axis=1):
                assert np.around(i) <= 1.0, 'Error: IOU > 1.'
                if i >= 0.7:
                    num_pred -= 1
    
    recall, precision, f1 = get_quality(tp, num_gt, num_pred)
    print(f'mIOU = {np.mean(iou3d):.4f}')
    print(f'TP = {tp}')
    print(f'FP = {num_pred - tp}')
    print(f'FN = {num_gt - tp}')
    print(f'Recall_0.1 = {tp_01 / num_gt:.4f}')
    print(f'Recall_0.7 = {recall:.4f}')
    print(f'Precision_0.7 = {precision:.4f}')
    print(f'F1_0.7 = {f1:.4f}')
    print(f'Avg # of ps = {num_pred / num:.4f}')