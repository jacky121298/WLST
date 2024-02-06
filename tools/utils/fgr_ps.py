import os
import torch
import pickle
import argparse
import numpy as np
import skimage.io as io
from tqdm import tqdm
from pathlib import Path
from shapely.geometry import MultiPoint
from shapely.geometry import box as Box
from pcdet.utils import box_utils
from pcdet.utils import calibration_kitti
from pcdet.ops.iou3d_nms import iou3d_nms_utils

def get_fov_flag(pts_rect, img_shape, calib, margin=0):
    pts_img, pts_rect_depth = calib.rect_to_img(pts_rect)
    val_flag_1 = np.logical_and(pts_img[:, 0] >= 0 - margin, pts_img[:, 0] < img_shape[1] + margin)
    val_flag_2 = np.logical_and(pts_img[:, 1] >= 0 - margin, pts_img[:, 1] < img_shape[0] + margin)
    val_flag_merge = np.logical_and(val_flag_1, val_flag_2)
    pts_valid_flag = np.logical_and(val_flag_merge, pts_rect_depth >= 0)
    return pts_valid_flag

def rotz(angle):
    c = np.cos(angle)
    s = np.sin(angle)
    rotz = np.array([
        [c, -s, 0],
        [s,  c, 0],
        [0,  0, 1],
    ])
    return rotz

def get_corners(box):
    x, y, z, l, w, h, r = box
    x_corners = l / 2 * np.array([1,  1,  1,  1, -1, -1, -1, -1])
    y_corners = w / 2 * np.array([1, -1, -1,  1,  1, -1, -1,  1])
    z_corners = h / 2 * np.array([1,  1, -1, -1,  1,  1, -1, -1])
    corners = np.vstack((x_corners, y_corners, z_corners))

    corners = np.dot(rotz(r), corners)
    corners[0, :] = corners[0, :] + x
    corners[1, :] = corners[1, :] + y
    corners[2, :] = corners[2, :] + z
    return corners

def post_process_coords(corner_coords, imsize):
    polygon_from_2d_box = MultiPoint(corner_coords).convex_hull
    img_canvas = Box(0, 0, imsize[0], imsize[1])

    if polygon_from_2d_box.intersects(img_canvas):
        img_intersection = polygon_from_2d_box.intersection(img_canvas)
        intersection_coords = np.array([coord for coord in img_intersection.exterior.coords])

        min_x = min(intersection_coords[:, 0])
        min_y = min(intersection_coords[:, 1])
        max_x = max(intersection_coords[:, 0])
        max_y = max(intersection_coords[:, 1])
        return min_x, min_y, max_x, max_y
    
    else:
        return None

def bb_intersection_over_union(boxA, boxB):
    # https://gist.github.com/meyerjo/dd3533edc97c81258898f60d8978eddc
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    if interArea == 0:
        return 0
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou

def get_iou2d(box, bb, calib, image_shape):
    corners = get_corners(box)     # (3, 8)
    in_front = np.argwhere(corners[0, :] > 0).flatten()
    corners = corners[:, in_front] # (3, n)
    corners = corners.T            # (n, 3)

    corners, _ = calib.lidar_to_img(corners)
    corners = corners.tolist() # (n, 2)
    
    final_coords = post_process_coords(corners, (image_shape[1], image_shape[0]))
    if final_coords == None:
        return None
    min_x, min_y, max_x, max_y = final_coords

    boxA = [min_x, min_y, max_x, max_y]
    boxB = [bb[0], bb[1], bb[2], bb[3]]
    iou2d = bb_intersection_over_union(boxA, boxB)
    return iou2d

def write_label(box, calib, label_path):
    box = box_utils.boxes3d_lidar_to_kitti_camera(box[np.newaxis, ...], calib).squeeze() # CCS: (x, y, z, l, h, w, r)
    label = 'Car 0.00 0 0.00 0.00 0.00 0.00 0.00 '
    label += '%.2f %.2f %.2f ' % (box[4], box[5], box[3]) # (h, w, l)
    label += '%.2f %.2f %.2f ' % (box[0], box[1], box[2]) # (x, y, z)
    label += '%.2f\n' % (box[6]) # r
    with open(label_path, 'a+') as f:
        f.write(label)

if __name__ == '__main__':
    # python ./utils/fgr_ps.py --fgr ./autolabeler/FGR/kitti/generation/labels --ps ../output/da-waymo-kitti_models/pvrcnn_st3d/pvrcnn_st3d/default/ps_label/ps_label_e0.pkl
    parser = argparse.ArgumentParser()
    parser.add_argument('--fgr', help='Path to pseudo labels of FGR.')
    parser.add_argument('--ps', help='Path to pseudo labels of 3D detector.')
    parser.add_argument('--T_pos', type=float, default=0.6, help='Positive IOU threshold.')
    parser.add_argument('--T_exist', type=float, default=0.7, help='Existence probability threshold.')
    args = parser.parse_args()

    output_dir = Path('../output/da-waymo-kitti_models/pvrcnn_st3d/pvrcnn_fgr_wl/default/ps_label')
    if output_dir.exists():
        print(f'Removing (existed) label path {output_dir}.')
        os.system(f'rm -rf {output_dir}')
    output_dir.mkdir(parents=True, exist_ok=True)

    used_class = ['Car']
    ps = pickle.load(open(args.ps, 'rb'))
    for frame_id in tqdm(ps):
        label_path = output_dir / f'{frame_id}.txt'
        with open(os.path.join('/tmp2/jacky1212/kitti/training/label_2', f'{frame_id}.txt'), 'r') as f:
            gt = f.readlines()
        
        box_2d = []
        for i in range(len(gt)):
            g = gt[i].split()
            if g[0] not in used_class:
                continue
            box_2d.append(np.array([g[4], g[5], g[6], g[7]]).astype(np.float))
        box_2d = np.array(box_2d)

        with open(os.path.join(args.fgr, f'{frame_id}.txt'), 'r') as f:
            fgr = f.readlines()

        ps_fgr = []
        for i in range(len(fgr)):
            p = fgr[i].split()
            if p[0] not in used_class:
                continue
            # (x, y, z, l, h, w, r)
            ps_fgr.append(np.array([p[11], p[12], p[13], p[10], p[8], p[9], p[14]]).astype(np.float))
        ps_fgr = np.array(ps_fgr)

        calib = calibration_kitti.Calibration('/tmp2/jacky1212/kitti/training/calib/' + frame_id + '.txt')
        if len(ps_fgr) != 0:
            ps_fgr = box_utils.boxes3d_kitti_camera_to_lidar(ps_fgr, calib)

        ps_3d = ps[frame_id]['gt_boxes']
        sc_3d = ps[frame_id]['iou_scores']
        ps_3d[:, :3] -= np.array([0.0, 0.0, 1.6])

        ps_3d_center = ps_3d[:, :3]
        pts_rect = calib.lidar_to_rect(ps_3d_center)
        img = io.imread('/tmp2/jacky1212/kitti/training/image_2/' + frame_id + '.png')
        fov_flag = get_fov_flag(pts_rect, img.shape[:2], calib, margin=5)

        ps_3d = ps_3d[fov_flag]
        sc_3d = sc_3d[fov_flag]

        remain = ps_3d[:, 7] > 0
        ps_3d = ps_3d[remain]
        sc_3d = sc_3d[remain]
        ps_3d = ps_3d[:, :7]
        if len(ps_3d) == 0:
            continue

        # Calculate the existence probability of ps
        prob_fgr, prob_3d = [], []
        for box in ps_fgr:
            prob_max = 0.0
            for bb in box_2d:
                prob = get_iou2d(box, bb, calib, img.shape[:2])
                if prob != None and prob > prob_max:
                    prob_max = prob
            prob_fgr.append(prob_max)
        
        for box in ps_3d:
            prob_max = 0.0
            for bb in box_2d:
                prob = get_iou2d(box, bb, calib, img.shape[:2])
                if prob != None and prob > prob_max:
                    prob_max = prob
            prob_3d.append(prob_max)

        # Consistency fusion strategy
        matched = [False for _ in range(len(ps_3d))]
        for i, box in enumerate(ps_fgr):
            b1 = box[np.newaxis, ...]
            b1 = torch.tensor(b1).float().cuda()
            b2 = torch.tensor(ps_3d).float().cuda()

            iou = iou3d_nms_utils.boxes_iou3d_gpu(b1, b2)
            iou = iou.cpu().detach().numpy().squeeze(axis=0)

            if np.max(iou) > 0.1:
                if max(prob_fgr[i], prob_3d[np.argmax(iou)]) >= args.T_exist:
                    matched[np.argmax(iou)] = True
                    if sc_3d[np.argmax(iou)] >= args.T_pos:
                        write_label(ps_3d[np.argmax(iou)], calib, label_path)

        for i in range(len(matched)):
            if matched[i] == False:
                if sc_3d[i] * prob_3d[i] >= args.T_pos:
                    write_label(ps_3d[i], calib, label_path)