import os
import re
import glob
import copy
import torch
import pickle
import numpy as np
from tqdm import tqdm
from nuscenes.utils.data_classes import Box
from shapely.geometry import MultiPoint, box, LineString
from pyquaternion.quaternion import Quaternion
from nuscenes.utils.geometry_utils import view_points
from pcdet.models import load_data_to_gpu
from pcdet.ops.iou3d_nms import iou3d_nms_utils
from pcdet.utils import LidarPointCloud, calibration_kitti, calibration_waymo

THRESH = None
PSEUDO_LABELS_M2D3D = {}
PSEUDO_LABELS_M3D = {}

def roty(angle):
    c = np.cos(angle)
    s = np.sin(angle)
    roty = np.array([
        [ c, 0, s],
        [ 0, 1, 0],
        [-s, 0, c],
    ])
    return roty

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

# From: https://github.com/Jamie725/Multimodal-Object-Detection-via-Probabilistic-Ensembling/blob/main/demo/FLIR/demo_bayesian_fusion.py
def bayesian_fusion(match_score_vec):
    log_positive_scores = np.log(match_score_vec)
    log_negative_scores = np.log(1 - match_score_vec)
    fused_positive = np.exp(np.sum(log_positive_scores))
    fused_negative = np.exp(np.sum(log_negative_scores))
    fused_positive_normalized = fused_positive / (fused_positive + fused_negative)
    return fused_positive_normalized

def check_already_exsit_pseudo_label_M2D3D(ps_label_dir, cur_epoch, cfg):
    global THRESH, PSEUDO_LABELS_M2D3D
    THRESH = cfg.THRESH
    ps_label_list = glob.glob(os.path.join(ps_label_dir, 'ps_label_e*_M2D3D.pkl'))
    if len(ps_label_list) == 0:
        return None

    ps_label_list.sort(key=os.path.getmtime, reverse=True)
    for cur_pkl in ps_label_list:
        num_epoch = re.findall('ps_label_e(.*)_M2D3D.pkl', cur_pkl)
        assert len(num_epoch) == 1

        if int(num_epoch[0]) < cur_epoch:
            latest_ps_label = pickle.load(open(cur_pkl, 'rb'))
            PSEUDO_LABELS_M2D3D.update(latest_ps_label)
            return cur_pkl
    return None

def check_already_exsit_pseudo_label_M3D(ps_label_dir, cur_epoch, cfg):
    global PSEUDO_LABELS_M3D
    ps_label_list = glob.glob(os.path.join(ps_label_dir, 'ps_label_e*_M3D.pkl'))
    if len(ps_label_list) == 0:
        return None

    ps_label_list.sort(key=os.path.getmtime, reverse=True)
    for cur_pkl in ps_label_list:
        num_epoch = re.findall('ps_label_e(.*)_M3D.pkl', cur_pkl)
        assert len(num_epoch) == 1

        if int(num_epoch[0]) < cur_epoch:
            latest_ps_label = pickle.load(open(cur_pkl, 'rb'))
            PSEUDO_LABELS_M3D.update(latest_ps_label)
            return cur_pkl
    return None

def save_pseudo_label_epoch_M2D3D(model, dataloader, ps_label_dir, epoch, cfg):
    n_pseudo_label = 0
    global PSEUDO_LABELS_M2D3D
    dataset_name = dataloader.dataset.cfg.DATA_PATH.split('/')[-1]
    with tqdm(dataloader, desc=f'generate_ps_e{epoch}_M2D3D') as t:
        for data in t:
            model.eval()
            if dataset_name == 'nuscenes':
                img, pts, box, theta_wl, calib_file, frame_id, image_shape, pose, bbox_wl, sample_token, where_wl = data
            else:
                img, pts, box, theta_wl, calib_file, frame_id, image_shape, pose, bbox_wl = data
            pts = pts.transpose(2, 1).float().cuda()
            img = img.float().cuda()

            output = model(img, pts, box)
            center = output['center'].cpu().detach().numpy()
            heading_scores = output['heading_scores'].cpu().detach().numpy()
            heading_residuals = output['heading_residuals'].cpu().detach().numpy()
            size_scores = output['size_scores'].cpu().detach().numpy()
            size_residuals = output['size_residuals'].cpu().detach().numpy()
            iou_pred = output['iou_pred'].cpu().detach().numpy()

            bs = output['center'].shape[0]
            heading_class = np.argmax(heading_scores, 1)
            heading_residual = np.array([heading_residuals[i, heading_class[i]] for i in range(bs)])
            size_class = np.argmax(size_scores, 1)
            size_residual = np.vstack([size_residuals[i, size_class[i], :] for i in range(bs)])
            
            for i in range(bs):
                heading_angle = class2angle(model.cfg, heading_class[i], heading_residual[i])
                size = class2size(model.cfg, size_class[i], size_residual[i])
                pred_box = np.hstack([center[i].reshape(1, 3), size.reshape(1, 3), heading_angle.reshape(1, 1)])
                iou = float(torch.sigmoid(torch.from_numpy(iou_pred[i])))

                if cfg.DATASET == 'kitti':
                    calib = calibration_kitti.Calibration(calib_file[i])
                elif cfg.DATASET == 'waymo':
                    c_file = {}
                    for key in calib_file:
                        c_file[key] = calib_file[key][i].cpu().detach().numpy()
                    calib = calibration_waymo.Calibration(c_file)
                elif cfg.DATASET == 'nuscenes':
                    calib = None

                if iou >= cfg.THRESH:
                    n_pseudo_label += 1

                if frame_id[i] not in PSEUDO_LABELS_M2D3D:
                    PSEUDO_LABELS_M2D3D[frame_id[i]] = {}
                PSEUDO_LABELS_M2D3D[frame_id[i]][f'{theta_wl[i].item()}'] = {
                    'pose': pose[i],
                    'calib': calib,
                    'bbox_wl': bbox_wl[i],
                    'image_shape': image_shape[i],
                    'gt_boxes': pred_box.squeeze(axis=0),
                    'iou_pred': iou,
                    'sample_token': sample_token[i] if dataset_name == 'nuscenes' else None,
                    'where_wl' : where_wl[i] if dataset_name == 'nuscenes' else None,
                }

    ps_path = os.path.join(ps_label_dir, f'ps_label_e{epoch}_M2D3D.pkl')
    with open(ps_path, 'wb') as f:
        pickle.dump(PSEUDO_LABELS_M2D3D, f)
    return n_pseudo_label / dataloader.dataset.__len__()

def save_pseudo_label_epoch_M3D(model, dataloader, ps_label_dir, epoch, cfg):
    n_pseudo_label = 0
    global PSEUDO_LABELS_M3D
    with tqdm(dataloader, desc=f'generate_ps_e{epoch}_M3D') as t:
        for data in t:
            model.eval()
            load_data_to_gpu(data)
            with torch.no_grad():
                pred_dicts, ret_dict = model(data)

            for b_idx in range(len(pred_dicts)):
                pred_iou_scores = None
                if 'pred_boxes' in pred_dicts[b_idx]:
                    pred_boxes = pred_dicts[b_idx]['pred_boxes'].detach().cpu().numpy()
                    pred_labels = pred_dicts[b_idx]['pred_labels'].detach().cpu().numpy()
                    pred_scores = pred_dicts[b_idx]['pred_scores'].detach().cpu().numpy()
                    if 'pred_iou_scores' in pred_dicts[b_idx]:
                        pred_iou_scores = pred_dicts[b_idx]['pred_iou_scores'].detach().cpu().numpy()

                    # Remove boxes under negative threshold
                    labels_remove_scores = np.array([cfg.NEG_THRESH])[pred_labels - 1]
                    remain_mask = pred_scores >= labels_remove_scores
                    
                    pred_boxes = pred_boxes[remain_mask]
                    pred_labels = pred_labels[remain_mask]
                    pred_scores = pred_scores[remain_mask]
                    if 'pred_iou_scores' in pred_dicts[b_idx]:
                        pred_iou_scores = pred_iou_scores[remain_mask]
                
                else:
                    pred_boxes = np.zeros((0, 7), dtype=np.float32)

                n_pseudo_label += pred_boxes.shape[0]
                PSEUDO_LABELS_M3D[data['frame_id'][b_idx]] = {
                    'gt_boxes': pred_boxes,
                    'iou_pred': pred_iou_scores,
                }

    ps_path = os.path.join(ps_label_dir, f'ps_label_e{epoch}_M3D.pkl')
    with open(ps_path, 'wb') as f:
        pickle.dump(PSEUDO_LABELS_M3D, f)
    return n_pseudo_label / dataloader.dataset.__len__()

def load_ps_label_M2D3D(frame_id, key):
    global THRESH, PSEUDO_LABELS_M2D3D
    if (frame_id in PSEUDO_LABELS_M2D3D) and (key in PSEUDO_LABELS_M2D3D[frame_id]):
        if PSEUDO_LABELS_M2D3D[frame_id][key]['iou_pred'] >= THRESH:
            box = PSEUDO_LABELS_M2D3D[frame_id][key]['gt_boxes']
            return np.copy(box)
        return np.array([])
    else:
        raise ValueError(f'Cannot find pseudo label for frame+key: {frame_id} + {key}')

def load_ps_label_M3D(frame_id):
    global PSEUDO_LABELS_M3D
    if frame_id in PSEUDO_LABELS_M3D:
        gt_boxes = PSEUDO_LABELS_M3D[frame_id]['gt_boxes']
        return np.copy(gt_boxes)
    else:
        return np.array([])

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

def lidar_to_camera_pc(pc, lidar_cs_rec, lidar_pose_rec, camera_cs_rec, camera_pose_rec):
    # Move to lidar ego vehicle coord system
    pc.rotate(Quaternion(lidar_cs_rec['rotation']).rotation_matrix)
    pc.translate(np.array(lidar_cs_rec['translation']))
    
    # Move to global coord system
    pc.rotate(Quaternion(lidar_pose_rec['rotation']).rotation_matrix)
    pc.translate(np.array(lidar_pose_rec['translation']))

    # Move to camera ego vehicle coord system
    pc.translate(-np.array(camera_pose_rec['translation']))
    pc.rotate(Quaternion(camera_pose_rec['rotation']).rotation_matrix.T)

    # Move to camera coord system
    pc.translate(-np.array(camera_cs_rec['translation']))
    pc.rotate(Quaternion(camera_cs_rec['rotation']).rotation_matrix.T)
    return pc

def post_process_coords(corner_coords, imsize):
    polygon_from_2d_box = MultiPoint(corner_coords).convex_hull
    img_canvas = box(0, 0, imsize[0], imsize[1])

    if polygon_from_2d_box.intersects(img_canvas):
        img_intersection = polygon_from_2d_box.intersection(img_canvas)
        if isinstance(img_intersection, LineString):
            return None
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

def get_iou2d(box, bbox_wl, calib, pose, image_shape, cfg, lidar_cs_rec, lidar_pose_rec, camera_cs_rec, camera_pose_rec):
    corners = get_corners(box)         # (3, 8)
    if lidar_cs_rec is None: # KITTI
        in_front = np.argwhere(corners[0, :] > 0).flatten()
        corners = corners[:, in_front] # (3, n)
    corners = corners.T                # (n, 3)

    if cfg.DATASET == 'kitti':
        if cfg.LATE_FUSION.get('SHIFT_COOR', None):
            corners -= np.array(cfg.LATE_FUSION.SHIFT_COOR, dtype=np.float32)
        corners, _ = calib.lidar_to_img(corners)
    elif cfg.DATASET == 'nuscenes':
        if cfg.LATE_FUSION.get('SHIFT_COOR', None):
            corners -= np.array(cfg.LATE_FUSION.SHIFT_COOR, dtype=np.float32)
        pts = LidarPointCloud.from_points(corners.T)
        pts = lidar_to_camera_pc(pts, lidar_cs_rec, lidar_pose_rec, camera_cs_rec, camera_pose_rec) 
        cam_intrinsic = np.array(camera_cs_rec['camera_intrinsic'])       
        pts_img = np.copy(view_points(pts.points, cam_intrinsic, normalize=True))
        corners = pts_img[:2, :].T # (n, 2)
    elif cfg.DATASET == 'waymo':
        image = {'image_shape_0': image_shape}
        corners, _ = calib.veh_to_img(corners, 'CAMERA_FRONT', pose, image)
    corners = corners.tolist() # (n, 2)
    
    final_coords = post_process_coords(corners, (image_shape[1], image_shape[0]))
    if final_coords == None:
        return None
    min_x, min_y, max_x, max_y = final_coords

    boxA = [min_x, min_y, max_x, max_y]
    boxB = [bbox_wl[2], bbox_wl[0], bbox_wl[3], bbox_wl[1]]
    iou2d = bb_intersection_over_union(boxA, boxB)
    return iou2d

def lidar_to_camera_box(box, lidar_cs_rec, lidar_pose_rec, camera_cs_rec, camera_pose_rec):
    # Move to lidar ego vehicle coord system
    box.rotate(Quaternion(lidar_cs_rec['rotation']))
    box.translate(np.array(lidar_cs_rec['translation']))
    
    # Move to global coord system
    box.rotate(Quaternion(lidar_pose_rec['rotation']))
    box.translate(np.array(lidar_pose_rec['translation']))

    # Move to camera ego vehicle coord system
    box.translate(-np.array(camera_pose_rec['translation']))
    box.rotate(Quaternion(camera_pose_rec['rotation']).inverse)

    # Move to camera coord system
    box.translate(-np.array(camera_cs_rec['translation']))
    box.rotate(Quaternion(camera_cs_rec['rotation']).inverse)
    return box

def camera_to_lidar_box(box, lidar_cs_rec, lidar_pose_rec, camera_cs_rec, camera_pose_rec):
    # Move to camera ego vehicle coord system
    box.rotate(Quaternion(camera_cs_rec['rotation']))
    box.translate(np.array(camera_cs_rec['translation']))

    # Move to global coord system
    box.rotate(Quaternion(camera_pose_rec['rotation']))
    box.translate(np.array(camera_pose_rec['translation']))

    # Move to lidar ego vehicle coord system
    box.translate(-np.array(lidar_pose_rec['translation']))
    box.rotate(Quaternion(lidar_pose_rec['rotation']).inverse)
    
    # Move to lidar coord system
    box.translate(-np.array(lidar_cs_rec['translation']))
    box.rotate(Quaternion(lidar_cs_rec['rotation']).inverse)
    return box

def late_fusion_epoch(ps_label_dir, epoch, cfg, model_fusion, nusc, T_exist=0.7):
    ps = {}
    global PSEUDO_LABELS_M2D3D, PSEUDO_LABELS_M3D
    for frame_id in PSEUDO_LABELS_M2D3D:
        ps[frame_id] = {
            'gt_boxes': [],
            'iou_pred': [],
        }
        bboxes = []
        box_M3D = PSEUDO_LABELS_M3D[frame_id]['gt_boxes'] # (n, 7)
        iou_M3D = PSEUDO_LABELS_M3D[frame_id]['iou_pred'] # (n,)
        matched = [False for _ in range(len(box_M3D))]
        camera_name = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']

        for key in PSEUDO_LABELS_M2D3D[frame_id]:
            box_M2D3D = PSEUDO_LABELS_M2D3D[frame_id][key]['gt_boxes'] # (7,)
            iou_M2D3D = PSEUDO_LABELS_M2D3D[frame_id][key]['iou_pred'] # float
            calib = PSEUDO_LABELS_M2D3D[frame_id][key]['calib']
            pose = np.array(PSEUDO_LABELS_M2D3D[frame_id][key]['pose'])
            image_shape = np.array(PSEUDO_LABELS_M2D3D[frame_id][key]['image_shape'])
            bbox_wl = np.array(PSEUDO_LABELS_M2D3D[frame_id][key]['bbox_wl'])
            bboxes.append(bbox_wl)
            
            lidar_cs_rec, lidar_pose_rec, camera_cs_rec, camera_pose_rec = None, None, None, None
            if cfg.DATASET == 'nuscenes':
                sample_token = PSEUDO_LABELS_M2D3D[frame_id][key]['sample_token']
                where_wl = PSEUDO_LABELS_M2D3D[frame_id][key]['where_wl']
                s_record = nusc.get('sample', sample_token)
                lidar_token = s_record['data']['LIDAR_TOP']
                camera_token = s_record['data'][camera_name[where_wl]]
                lidar_record = nusc.get('sample_data', lidar_token)
                camera_record = nusc.get('sample_data', camera_token)
                lidar_cs_rec = nusc.get('calibrated_sensor', lidar_record['calibrated_sensor_token'])
                lidar_pose_rec = nusc.get('ego_pose', lidar_record['ego_pose_token'])
                camera_cs_rec = nusc.get('calibrated_sensor', camera_record['calibrated_sensor_token'])
                camera_pose_rec = nusc.get('ego_pose', camera_record['ego_pose_token'])
            
            box_M2D3D_lidar = np.copy(box_M2D3D)
            if cfg.TRANSFER_TO_CENTER:
                if cfg.DATASET == 'kitti':
                    box_M2D3D_lidar[6] += float(key)
                    calib = calibration_kitti.Calibration(os.path.join('/tmp2/jacky1212/kitti/training/calib', f'{frame_id}.txt'))
                    center = calib.lidar_to_rect(box_M2D3D_lidar[:3].reshape(1, 3))
                    center = roty(-float(key)) @ center.reshape(3, 1)
                    box_M2D3D_lidar[:3] = np.squeeze(calib.rect_to_lidar(center.reshape(1, 3)))
                elif cfg.DATASET == 'nuscenes':
                    box_wl = Box(box_M2D3D_lidar[:3], box_M2D3D_lidar[[4, 3, 5]], Quaternion(axis=[0, 0, 1], radians=box_M2D3D_lidar[6]))
                    box_wl = lidar_to_camera_box(box_wl, lidar_cs_rec, lidar_pose_rec, camera_cs_rec, camera_pose_rec)
                    box_wl.rotate(Quaternion(axis=[0, 1, 0], radians=-float(key)))
                    box_wl = camera_to_lidar_box(box_wl, lidar_cs_rec, lidar_pose_rec, camera_cs_rec, camera_pose_rec)
                    box_wl = np.concatenate((box_wl.center.reshape((1, 3)), box_wl.wlh.reshape((1, 3)), np.array(box_wl.orientation.yaw_pitch_roll[0]).reshape((1, 1))), axis=1)
                    box_M2D3D_lidar = np.squeeze(box_wl)[[0, 1, 2, 4, 3, 5, 6]]
                else:
                    raise NotImplementedError(f'Dataset {cfg.DATASET} is not supported.')

            # Case 1: There is no 3D pseudo labels.
            if len(box_M3D) == 0:
                if cfg.LATE_FUSION.NAME == 'consistency':
                    prob_2d = get_iou2d(box_M2D3D_lidar, bbox_wl, calib, pose, image_shape, cfg, lidar_cs_rec, lidar_pose_rec, camera_cs_rec, camera_pose_rec)
                    if prob_2d == None:
                        prob_2d = 0.0
                    iou_M2D3D = iou_M2D3D * prob_2d
                    if iou_M2D3D >= cfg.THRESH:
                        ps[frame_id]['gt_boxes'].append(box_M2D3D_lidar)
                        ps[frame_id]['iou_pred'].append(iou_M2D3D)
                
                elif cfg.LATE_FUSION.NAME == 'nms' or cfg.LATE_FUSION.NAME == 'proben':
                    if iou_M2D3D >= cfg.THRESH:
                        ps[frame_id]['gt_boxes'].append(box_M2D3D_lidar)
                        ps[frame_id]['iou_pred'].append(iou_M2D3D)

                elif cfg.LATE_FUSION.NAME == 'learning':
                    model_fusion.eval()
                    iou2d_2d = get_iou2d(box_M2D3D_lidar, bbox_wl, calib, pose, image_shape, cfg, lidar_cs_rec, lidar_pose_rec, camera_cs_rec, camera_pose_rec)
                    if iou2d_2d == None:
                        iou2d_2d = 0.0
                    data = torch.from_numpy(np.array([-1, iou_M2D3D, -1, iou2d_2d, -1]))
                    data = data.float().cuda().reshape(1, cfg.LATE_FUSION.CHANNEL, 1)
                    out = model_fusion(data)
                    iou = float(torch.sigmoid(torch.from_numpy(np.squeeze(out.cpu().detach().numpy()))))
                    if iou >= cfg.THRESH:
                        ps[frame_id]['gt_boxes'].append(box_M2D3D_lidar)
                        ps[frame_id]['iou_pred'].append(iou)
                
                continue
            
            # Case 2: IoU(2D pseudo label, 3D pseudo labels) > 0.
            iou = iou3d_nms_utils.boxes_iou3d_gpu(torch.tensor(box_M2D3D_lidar[np.newaxis, ...]).float().cuda(), torch.tensor(box_M3D).float().cuda()).cpu().detach().numpy().squeeze(axis=0)
            if np.max(iou) > 0.0:
                matched[np.argmax(iou)] = True
                if cfg.LATE_FUSION.NAME == 'consistency':
                    prob_2d = get_iou2d(box_M2D3D_lidar, bbox_wl, calib, pose, image_shape, cfg, lidar_cs_rec, lidar_pose_rec, camera_cs_rec, camera_pose_rec)
                    if prob_2d == None:
                        prob_2d = 0.0
                    prob_3d = get_iou2d(box_M3D[np.argmax(iou)], bbox_wl, calib, pose, image_shape, cfg, lidar_cs_rec, lidar_pose_rec, camera_cs_rec, camera_pose_rec)
                    if prob_3d == None:
                        prob_3d = 0.0
                    if iou_M3D[np.argmax(iou)] > iou_M2D3D:
                        if iou_M3D[np.argmax(iou)] >= cfg.THRESH and prob_2d >= T_exist and prob_3d >= T_exist:
                            ps[frame_id]['gt_boxes'].append(box_M3D[np.argmax(iou)])
                            ps[frame_id]['iou_pred'].append(iou_M3D[np.argmax(iou)])
                    elif iou_M2D3D >= cfg.THRESH and prob_2d >= T_exist and prob_3d >= T_exist:
                        ps[frame_id]['gt_boxes'].append(box_M2D3D_lidar)
                        ps[frame_id]['iou_pred'].append(iou_M2D3D)

                elif cfg.LATE_FUSION.NAME == 'nms':
                    if iou_M3D[np.argmax(iou)] > iou_M2D3D:
                        if iou_M3D[np.argmax(iou)] >= cfg.THRESH:
                            ps[frame_id]['gt_boxes'].append(box_M3D[np.argmax(iou)])
                            ps[frame_id]['iou_pred'].append(iou_M3D[np.argmax(iou)])
                    elif iou_M2D3D >= cfg.THRESH:
                        ps[frame_id]['gt_boxes'].append(box_M2D3D_lidar)
                        ps[frame_id]['iou_pred'].append(iou_M2D3D)

                elif cfg.LATE_FUSION.NAME == 'proben':
                    iou_proben = bayesian_fusion(np.array([iou_M2D3D, iou_M3D[np.argmax(iou)]]))
                    if iou_proben >= cfg.THRESH:
                        if iou_M2D3D >= iou_M3D[np.argmax(iou)]:
                            ps[frame_id]['gt_boxes'].append(box_M2D3D_lidar)
                        else:
                            ps[frame_id]['gt_boxes'].append(box_M3D[np.argmax(iou)])
                        ps[frame_id]['iou_pred'].append(iou_proben)

                elif cfg.LATE_FUSION.NAME == 'learning':
                    model_fusion.eval()
                    iou2d_2d = get_iou2d(box_M2D3D_lidar, bbox_wl, calib, pose, image_shape, cfg, lidar_cs_rec, lidar_pose_rec, camera_cs_rec, camera_pose_rec)
                    if iou2d_2d == None:
                        iou2d_2d = 0.0
                    iou2d_3d = get_iou2d(box_M3D[np.argmax(iou)], bbox_wl, calib, pose, image_shape, cfg, lidar_cs_rec, lidar_pose_rec, camera_cs_rec, camera_pose_rec)
                    if iou2d_3d == None:
                        iou2d_3d = 0.0
                    data = torch.from_numpy(np.array([np.max(iou), iou_M2D3D, iou_M3D[np.argmax(iou)], iou2d_2d, iou2d_3d]))
                    data = data.float().cuda().reshape(1, cfg.LATE_FUSION.CHANNEL, 1)
                    out = model_fusion(data)
                    iou = float(torch.sigmoid(torch.from_numpy(np.squeeze(out.cpu().detach().numpy()))))
                    if iou >= cfg.THRESH:
                        if iou_M2D3D >= iou_M3D[np.argmax(iou)]:
                            ps[frame_id]['gt_boxes'].append(box_M2D3D_lidar)
                        else:
                            ps[frame_id]['gt_boxes'].append(box_M3D[np.argmax(iou)])
                        ps[frame_id]['iou_pred'].append(iou)

            # Case 3: There is no IoU(2D pseudo label, 3D pseudo labels) > 0.
            else:
                if cfg.LATE_FUSION.NAME == 'consistency':
                    prob_2d = get_iou2d(box_M2D3D_lidar, bbox_wl, calib, pose, image_shape, cfg, lidar_cs_rec, lidar_pose_rec, camera_cs_rec, camera_pose_rec)
                    if prob_2d == None:
                        prob_2d = 0.0
                    iou_M2D3D = iou_M2D3D * prob_2d
                    if iou_M2D3D >= cfg.THRESH:
                        ps[frame_id]['gt_boxes'].append(box_M2D3D_lidar)
                        ps[frame_id]['iou_pred'].append(iou_M2D3D)
                
                elif cfg.LATE_FUSION.NAME == 'nms' or cfg.LATE_FUSION.NAME == 'proben':
                    if iou_M2D3D >= cfg.THRESH:
                        ps[frame_id]['gt_boxes'].append(box_M2D3D_lidar)
                        ps[frame_id]['iou_pred'].append(iou_M2D3D)

                elif cfg.LATE_FUSION.NAME == 'learning':
                    model_fusion.eval()
                    iou2d_2d = get_iou2d(box_M2D3D_lidar, bbox_wl, calib, pose, image_shape, cfg, lidar_cs_rec, lidar_pose_rec, camera_cs_rec, camera_pose_rec)
                    if iou2d_2d == None:
                        iou2d_2d = 0.0
                    data = torch.from_numpy(np.array([-1, iou_M2D3D, -1, iou2d_2d, -1]))
                    data = data.float().cuda().reshape(1, cfg.LATE_FUSION.CHANNEL, 1)
                    out = model_fusion(data)
                    iou = float(torch.sigmoid(torch.from_numpy(np.squeeze(out.cpu().detach().numpy()))))
                    if iou >= cfg.THRESH:
                        ps[frame_id]['gt_boxes'].append(box_M2D3D_lidar)
                        ps[frame_id]['iou_pred'].append(iou)

        # Case 4: The unmatched M3D.
        for i in range(len(matched)):
            if matched[i] == False:
                if cfg.LATE_FUSION.NAME == 'consistency':
                    prob_3d = 0.0
                    for bb in bboxes:
                        prob = get_iou2d(box_M3D[i], bb, calib, pose, image_shape, cfg, lidar_cs_rec, lidar_pose_rec, camera_cs_rec, camera_pose_rec)
                        if prob == None:
                            continue
                        if prob > prob_3d:
                            prob_3d = prob
                    iou_M3D[i] = iou_M3D[i] * prob_3d
                    if iou_M3D[i] >= cfg.THRESH:
                        ps[frame_id]['gt_boxes'].append(box_M3D[i])
                        ps[frame_id]['iou_pred'].append(iou_M3D[i])
                
                elif cfg.LATE_FUSION.NAME == 'nms' or cfg.LATE_FUSION.NAME == 'proben':
                    if iou_M3D[i] >= cfg.THRESH:
                        ps[frame_id]['gt_boxes'].append(box_M3D[i])
                        ps[frame_id]['iou_pred'].append(iou_M3D[i])

                elif cfg.LATE_FUSION.NAME == 'learning':
                    model_fusion.eval()
                    data = torch.from_numpy(np.array([-1, -1, iou_M3D[i], -1, -1]))
                    data = data.float().cuda().reshape(1, cfg.LATE_FUSION.CHANNEL, 1)
                    out = model_fusion(data)
                    iou = float(torch.sigmoid(torch.from_numpy(np.squeeze(out.cpu().detach().numpy()))))
                    if iou >= cfg.THRESH:
                        ps[frame_id]['gt_boxes'].append(box_M3D[i])
                        ps[frame_id]['iou_pred'].append(iou)

        ps[frame_id]['gt_boxes'] = np.array(ps[frame_id]['gt_boxes'])
        ps[frame_id]['iou_pred'] = np.array(ps[frame_id]['iou_pred'])

    ps_M2D3D_path = os.path.join(ps_label_dir, f'ps_label_e{epoch}_M2D3D.pkl')
    with open(ps_M2D3D_path, 'wb') as f:
        pickle.dump(PSEUDO_LABELS_M2D3D, f)
    
    PSEUDO_LABELS_M3D = copy.deepcopy(ps)
    ps_M3D_path = os.path.join(ps_label_dir, f'ps_label_e{epoch}_M3D.pkl')
    with open(ps_M3D_path, 'wb') as f:
        pickle.dump(PSEUDO_LABELS_M3D, f)