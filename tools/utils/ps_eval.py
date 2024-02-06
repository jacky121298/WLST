import os
import torch
import pickle
import argparse
import numpy as np
import skimage.io as io
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

def get_fov_flag(pts_rect, img_shape, calib, margin=0):
    pts_img, pts_rect_depth = calib.rect_to_img(pts_rect)
    val_flag_1 = np.logical_and(pts_img[:, 0] >= 0 - margin, pts_img[:, 0] < img_shape[1] + margin)
    val_flag_2 = np.logical_and(pts_img[:, 1] >= 0 - margin, pts_img[:, 1] < img_shape[0] + margin)
    val_flag_merge = np.logical_and(val_flag_1, val_flag_2)
    pts_valid_flag = np.logical_and(val_flag_merge, pts_rect_depth >= 0)
    return pts_valid_flag

if __name__ == '__main__':
    # python ./utils/ps_eval.py --ps ../output/da-waymo-kitti_models/pvrcnn_st3d/pvrcnn_st3d/default/ps_label/ps_label_e0.pkl --ps_from st3d
    # python ./utils/ps_eval.py --ps ../output/da-waymo-kitti_models/pvrcnn_st3d/wl_labeler_ros_st/consistency/ps_label/ps_label_e0_M3D.pkl --ps_from wlst
    parser = argparse.ArgumentParser()
    parser.add_argument('--ps', help='Path to pseudo labels.')
    parser.add_argument('--ps_from', help='Where the pseudo labels from.')
    args = parser.parse_args()

    assert args.ps_from in ['st3d', 'wlst']
    ps = pickle.load(open(args.ps, 'rb'))
    with open('../data/kitti/ImageSets/train.txt', 'r') as f:
        split = f.readlines()

    iou3d = []
    used_class = ['Car']
    num, num_gt, num_pred, tp = 0, 0, 0, 0
    
    for s in tqdm(split):
        s = s.strip()
        
        # GT labels
        with open(os.path.join('/tmp2/jacky1212/kitti/training/label_2', s + '.txt'), 'r') as f:
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
                
        # Pseudo labels
        if s not in ps.keys():
            for _ in range(labels.shape[0]):
                iou3d.append(0)
            continue
        
        ps_boxes = ps[s]['gt_boxes'] # LCS: (x, y, z, l, w, h, r), shape: (N, 7)
        if args.ps_from == 'st3d':
            iou_scores = ps[s]['iou_scores']
        if args.ps_from == 'wlst':
            iou_scores = ps[s]['iou_pred']
        if len(ps_boxes) == 0:
            for _ in range(labels.shape[0]):
                iou3d.append(0)
            continue
        
        ps_boxes_lidar_center = ps_boxes[:, :3]
        calib = calibration_kitti.Calibration('/tmp2/jacky1212/kitti/training/calib/' + s + '.txt')
        pts_rect = calib.lidar_to_rect(ps_boxes_lidar_center)
        img = io.imread('/tmp2/jacky1212/kitti/training/image_2/' + s + '.png')
        fov_flag = get_fov_flag(pts_rect, img.shape[:2], calib, margin=5)
        
        ps_boxes = ps_boxes[fov_flag]
        iou_scores = iou_scores[fov_flag]
        if args.ps_from == 'st3d':
            ps_boxes = ps_boxes[ps_boxes[:, 7] > 0]

        if len(ps_boxes) == 0:
            for _ in range(labels.shape[0]):
                iou3d.append(0)
            continue
        
        ps_boxes = ps_boxes[:, :7]
        ps_boxes[:, :3] -= np.array([0.0, 0.0, 1.6], dtype=np.float32).reshape(1, 3)
        ps_boxes_camera = box_utils.boxes3d_lidar_to_kitti_camera(ps_boxes, calib) # CCS: (x, y, z, l, h, w, r), shape: (N, 7)
        
        num += 1
        num_pred += ps_boxes_camera.shape[0]

        if labels.shape[0] == 0:
            continue

        pseudo = box_utils.boxes3d_kitti_camera_to_lidar(ps_boxes_camera, calib)
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
            if i >= 0.7:
                tp += 1
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
    print(f'Recall_0.7 = {recall:.4f}')
    print(f'Precision_0.7 = {precision:.4f}')
    print(f'F1_0.7 = {f1:.4f}')
    print(f'Avg # of ps = {num_pred / num:.4f}')

### autolabeler
# e0
# =======================
# mIOU = 0.4624
# TP = 6538
# FP = 2483
# FN = 7819
# Recall_0.7 = 0.4554
# Precision_0.7 = 0.7248
# F1_0.7 = 0.5593
# Avg # of ps = 2.7825
# =======================

### st3d
# e0
# =======================
# mIOU = 0.4969
# TP = 7233
# FP = 3233
# FN = 7124
# Recall_0.7 = 0.5038
# Precision_0.7 = 0.6911
# F1_0.7 = 0.5828
# Avg # of ps = 3.5215
# =======================
# e4
# =======================
# mIOU = 0.4612
# TP = 7144
# FP = 2018
# FN = 7213
# Recall_0.7 = 0.4976
# Precision_0.7 = 0.7797
# F1_0.7 = 0.6075
# Avg # of ps = 3.1702
# =======================
# e8
# =======================
# mIOU = 0.4349
# TP = 6906
# FP = 1539
# FN = 7451
# Recall_0.7 = 0.4810
# Precision_0.7 = 0.8178
# F1_0.7 = 0.6057
# Avg # of ps = 2.9894
# =======================
# e12
# =======================
# mIOU = 0.4309
# TP = 6756
# FP = 1704
# FN = 7601
# Recall_0.7 = 0.4706
# Precision_0.7 = 0.7986
# F1_0.7 = 0.5922
# Avg # of ps = 2.9989
# =======================
# e16
# =======================
# mIOU = 0.4175
# TP = 6507
# FP = 1742
# FN = 7850
# Recall_0.7 = 0.4532
# Precision_0.7 = 0.7888
# F1_0.7 = 0.5757
# Avg # of ps = 2.9705
# =======================
# e20
# =======================
# mIOU = 0.4212
# TP = 6695
# FP = 1555
# FN = 7662
# Recall_0.7 = 0.4663
# Precision_0.7 = 0.8115
# F1_0.7 = 0.5923
# Avg # of ps = 2.9549
# =======================
# e24
# =======================
# mIOU = 0.4206
# TP = 6705
# FP = 1553
# FN = 7652
# Recall_0.7 = 0.4670
# Precision_0.7 = 0.8119
# F1_0.7 = 0.5930
# Avg # of ps = 2.9556
# =======================
# e28
# =======================
# mIOU = 0.4217
# TP = 6627
# FP = 1707
# FN = 7730
# Recall_0.7 = 0.4616
# Precision_0.7 = 0.7952
# F1_0.7 = 0.5841
# Avg # of ps = 2.9786
# =======================

## wlst
# e0
# =======================
# mIOU = 0.4626
# TP = 6893
# FP = 1922
# FN = 7464
# Recall_0.7 = 0.4801
# Precision_0.7 = 0.7820
# F1_0.7 = 0.5949
# Avg # of ps = 3.0897
# =======================
# e10
# =======================
# mIOU = 0.5401
# TP = 8242
# FP = 1959
# FN = 6115
# Recall_0.7 = 0.5741
# Precision_0.7 = 0.8080
# F1_0.7 = 0.6712
# Avg # of ps = 3.3174
# =======================
# e20
# =======================
# mIOU = 0.5608
# TP = 8586
# FP = 1990
# FN = 5771
# Recall_0.7 = 0.5980
# Precision_0.7 = 0.8118
# F1_0.7 = 0.6887
# Avg # of ps = 3.4607
# =======================
# e30
# =======================
# mIOU = 0.5658
# TP = 8643
# FP = 2053
# FN = 5714
# Recall_0.7 = 0.6020
# Precision_0.7 = 0.8081
# F1_0.7 = 0.6900
# Avg # of ps = 3.4716
# =======================
# e40
# =======================
# mIOU = 0.5650
# TP = 8650
# FP = 2029
# FN = 5707
# Recall_0.7 = 0.6025
# Precision_0.7 = 0.8100
# F1_0.7 = 0.6910
# Avg # of ps = 3.4605
# =======================