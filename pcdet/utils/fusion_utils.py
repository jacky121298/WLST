import os
import re
import glob
import copy
import torch
import pickle
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from shapely.geometry import MultiPoint, box
from pcdet.models import load_data_to_gpu
from pcdet.ops.iou3d_nms import iou3d_nms_utils
from pcdet.utils import calibration_kitti, calibration_waymo

PSEUDO_LABELS_M3D = {}
PSEUDO_LABELS_M2D3D = {}

class FusionDataset(Dataset):
    def __init__(self, cfg, data_dir):
        self.cfg = cfg
        self.data = []
        self.label = []

        data_path = os.path.join(data_dir, 'data.pkl')
        label_path = os.path.join(data_dir, 'label.pkl')
        if os.path.exists(data_path) and os.path.exists(label_path):
            self.data = pickle.load(open(data_path, 'rb'))
            self.label = pickle.load(open(label_path, 'rb'))
            return

        global PSEUDO_LABELS_M2D3D, PSEUDO_LABELS_M3D
        for frame_id in PSEUDO_LABELS_M2D3D:
            box_M3D = PSEUDO_LABELS_M3D[frame_id]['pred_box']  # (n, 7)
            iou_M3D = PSEUDO_LABELS_M3D[frame_id]['iou_pred']  # (n,)
            gt_boxes = PSEUDO_LABELS_M3D[frame_id]['gt_boxes'] # (N, 7)
            matched = [False for _ in range(len(box_M3D))]
            if len(box_M3D) == 0:
                continue

            for key in PSEUDO_LABELS_M2D3D[frame_id]:
                box_M2D3D = PSEUDO_LABELS_M2D3D[frame_id][key]['pred_box'] # (7,)
                iou_M2D3D = PSEUDO_LABELS_M2D3D[frame_id][key]['iou_pred'] # float
                gt_box = PSEUDO_LABELS_M2D3D[frame_id][key]['gt_box'] # (7,)
                calib = PSEUDO_LABELS_M2D3D[frame_id][key]['calib']
                pose = np.array(PSEUDO_LABELS_M2D3D[frame_id][key]['pose'])
                image_shape = np.array(PSEUDO_LABELS_M2D3D[frame_id][key]['image_shape'])
                bbox_wl = self.get_2dbox(gt_box, calib, pose, image_shape)
                if bbox_wl == None:
                    continue
                
                iou = iou3d_nms_utils.boxes_iou3d_gpu(torch.tensor(box_M2D3D[np.newaxis, ...]).float().cuda(), torch.tensor(box_M3D).float().cuda()).cpu().detach().numpy().squeeze(axis=0)
                if np.max(iou) > 0.0:
                    matched[np.argmax(iou)] = True
                    iou2d_2d = self.get_iou2d(box_M2D3D, bbox_wl, calib, pose, image_shape)
                    iou2d_3d = self.get_iou2d(box_M3D[np.argmax(iou)], bbox_wl, calib, pose, image_shape)
                    if iou2d_2d == None or iou2d_3d == None:
                        continue

                    gt_iou2d = iou3d_nms_utils.boxes_iou3d_gpu(torch.tensor(box_M2D3D[np.newaxis, ...]).float().cuda(), torch.tensor(gt_box[np.newaxis, ...]).float().cuda()).cpu().detach().numpy().squeeze(axis=0)
                    gt_iou3d = iou3d_nms_utils.boxes_iou3d_gpu(torch.tensor(box_M3D[np.argmax(iou)][np.newaxis, ...]).float().cuda(), torch.tensor(gt_box[np.newaxis, ...]).float().cuda()).cpu().detach().numpy().squeeze(axis=0)
                    self.data.append(np.array([np.max(iou), iou_M2D3D, iou_M3D[np.argmax(iou)], iou2d_2d, iou2d_3d]))
                    if np.max(gt_iou2d) >= 0.7 or np.max(gt_iou3d) >= 0.7:
                        self.label.append(1.0)
                    else:
                        self.label.append(0.0)

                else:
                    iou2d_2d = self.get_iou2d(box_M2D3D, bbox_wl, calib, pose, image_shape)
                    if iou2d_2d == None:
                        continue
                    self.data.append(np.array([-1, iou_M2D3D, -1, iou2d_2d, -1]))
                    gt_iou2d = iou3d_nms_utils.boxes_iou3d_gpu(torch.tensor(box_M2D3D[np.newaxis, ...]).float().cuda(), torch.tensor(gt_box[np.newaxis, ...]).float().cuda()).cpu().detach().numpy().squeeze(axis=0)
                    if np.max(gt_iou2d) >= 0.7:
                        self.label.append(1.0)
                    else:
                        self.label.append(0.0)

            # For unmatched M3D
            for i in range(len(matched)):
                if matched[i] == False:
                    self.data.append(np.array([-1, -1, iou_M3D[i], -1, -1]))
                    gt_iou3d = iou3d_nms_utils.boxes_iou3d_gpu(torch.tensor(box_M3D[i][np.newaxis, ...]).float().cuda(), torch.tensor(gt_boxes).float().cuda()).cpu().detach().numpy().squeeze(axis=0)
                    if np.max(gt_iou3d) >= 0.7:
                        self.label.append(1.0)
                    else:
                        self.label.append(0.0)

        with open(os.path.join(data_dir, 'data.pkl'), 'wb') as f:
            pickle.dump(self.data, f)
        with open(os.path.join(data_dir, 'label.pkl'), 'wb') as f:
            pickle.dump(self.label, f)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        data = self.data[index]
        label = self.label[index]
        return torch.from_numpy(data), torch.from_numpy(np.array([label]))

    def get_corners(self, box):
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

    def post_process_coords(self, corner_coords, imsize):
        polygon_from_2d_box = MultiPoint(corner_coords).convex_hull
        img_canvas = box(0, 0, imsize[0], imsize[1])

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

    def bb_intersection_over_union(self, boxA, boxB):
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
    
    def get_iou2d(self, box, bbox_wl, calib, pose, image_shape):
        corners = self.get_corners(box) # (3, 8)
        in_front = np.argwhere(corners[0, :] > 0).flatten()
        corners = corners[:, in_front]   # (3, n)

        if self.cfg.LATE_FUSION.DATASET == 'kitti':
            corners = calib.lidar_to_img(corners.T)
        elif self.cfg.LATE_FUSION.DATASET == 'waymo':
            image = {'image_shape_0': image_shape}
            corners, _ = calib.veh_to_img(corners.T, 'CAMERA_FRONT', pose, image)
        corners = corners.tolist() # (n, 2)
        
        final_coords = self.post_process_coords(corners, (image_shape[1], image_shape[0]))
        if final_coords == None:
            return None
        min_x, min_y, max_x, max_y = final_coords

        boxA = [min_x, min_y, max_x, max_y]
        boxB = [bbox_wl[0], bbox_wl[1], bbox_wl[2], bbox_wl[3]]
        iou2d = self.bb_intersection_over_union(boxA, boxB)
        return iou2d

    def get_2dbox(self, box, calib, pose, image_shape):
        corners = self.get_corners(box) # (3, 8)
        in_front = np.argwhere(corners[0, :] > 0).flatten()
        corners = corners[:, in_front]   # (3, n)

        if self.cfg.LATE_FUSION.DATASET == 'kitti':
            corners = calib.lidar_to_img(corners.T)
        elif self.cfg.LATE_FUSION.DATASET == 'waymo':
            image = {'image_shape_0': image_shape}
            corners, _ = calib.veh_to_img(corners.T, 'CAMERA_FRONT', pose, image)
        corners = corners.tolist() # (n, 2)
        
        final_coords = self.post_process_coords(corners, (image_shape[1], image_shape[0]))
        return final_coords

def roty(angle):
    c = np.cos(angle)
    s = np.sin(angle)
    roty = np.array([
        [ c, 0, s],
        [ 0, 1, 0],
        [-s, 0, c],
    ])
    return roty

def rotz(angle):
    c = np.cos(angle)
    s = np.sin(angle)
    rotz = np.array([
        [c, -s, 0],
        [s,  c, 0],
        [0,  0, 1],
    ])
    return rotz

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

def load_M2D3D(data_path):
    global PSEUDO_LABELS_M2D3D
    PSEUDO_LABELS_M2D3D = pickle.load(open(data_path, 'rb'))

def load_M3D(data_path):
    global PSEUDO_LABELS_M3D
    PSEUDO_LABELS_M3D = pickle.load(open(data_path, 'rb'))

def prepare_M2D3D_epoch(model, dataloader, cfg, ps_label_dir):
    global PSEUDO_LABELS_M2D3D
    with tqdm(dataloader, desc='Preparing M2D3D input data') as t:
        for data in t:
            model.eval()
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
                pred_box = pred_box.squeeze(axis=0) # (7,)
                iou = float(torch.sigmoid(torch.from_numpy(iou_pred[i])))

                if cfg.LATE_FUSION.DATASET == 'kitti':
                    calib = calibration_kitti.Calibration(calib_file[i])
                elif cfg.LATE_FUSION.DATASET == 'waymo':
                    c_file = {}
                    for key in calib_file:
                        c_file[key] = calib_file[key][i].cpu().detach().numpy()
                    calib = calibration_waymo.Calibration(c_file)

                gt_box = np.copy(box[i].numpy())
                if cfg.LATE_FUSION.TRANSFER_TO_CENTER:
                    gt_box[6] += float(theta_wl[i])
                    pred_box[6] += float(theta_wl[i])
                    if cfg.LATE_FUSION.DATASET == 'kitti':
                        xyz = calib.lidar_to_rect(gt_box[:3].reshape(1, 3))
                        xyz = roty(-float(theta_wl[i])) @ xyz.reshape(3, 1)
                        gt_box[:3] = np.squeeze(calib.rect_to_lidar(xyz.reshape(1, 3)))

                        xyz = calib.lidar_to_rect(pred_box[:3].reshape(1, 3))
                        xyz = roty(-float(theta_wl[i])) @ xyz.reshape(3, 1)
                        pred_box[:3] = np.squeeze(calib.rect_to_lidar(xyz.reshape(1, 3)))
                    elif cfg.LATE_FUSION.DATASET == 'waymo':
                        xyz = calib.veh_to_rect(gt_box[:3].reshape(1, 3))
                        xyz = rotz(float(theta_wl[i])) @ xyz.reshape(3, 1)
                        gt_box[:3] = np.squeeze(calib.rect_to_veh(xyz.reshape(1, 3)))

                        xyz = calib.veh_to_rect(pred_box[:3].reshape(1, 3))
                        xyz = rotz(float(theta_wl[i])) @ xyz.reshape(3, 1)
                        pred_box[:3] = np.squeeze(calib.rect_to_veh(xyz.reshape(1, 3)))
                    else:
                        raise NotImplementedError(f'Dataset {cfg.LATE_FUSION.DATASET} is not supported.')
                
                if frame_id[i] not in PSEUDO_LABELS_M2D3D:
                    PSEUDO_LABELS_M2D3D[frame_id[i]] = {}
                PSEUDO_LABELS_M2D3D[frame_id[i]][f'{theta_wl[i].item()}'] = {
                    'pose': pose[i],
                    'calib': calib,
                    'image_shape': image_shape[i],
                    'gt_box': gt_box,
                    'pred_box': pred_box,
                    'iou_pred': iou,
                }

    ps_path = os.path.join(ps_label_dir, f'ps_label_M2D3D.pkl')
    with open(ps_path, 'wb') as f:
        pickle.dump(PSEUDO_LABELS_M2D3D, f)

def prepare_M3D_epoch(model, dataloader, cfg, ps_label_dir):
    global PSEUDO_LABELS_M3D
    with tqdm(dataloader, desc='Preparing M3D input data') as t:
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
                    labels_remove_scores = np.array([cfg.LATE_FUSION.NEG_THRESH])[pred_labels - 1]
                    remain_mask = pred_scores >= labels_remove_scores
                    
                    pred_boxes = pred_boxes[remain_mask]
                    pred_labels = pred_labels[remain_mask]
                    pred_scores = pred_scores[remain_mask]
                    if 'pred_iou_scores' in pred_dicts[b_idx]:
                        pred_iou_scores = pred_iou_scores[remain_mask]
                
                else:
                    pred_boxes = np.zeros((0, 7), dtype=np.float32)

                gt_boxes = data['gt_boxes'].detach().cpu().numpy()
                gt_boxes = gt_boxes[b_idx][:, :7]

                PSEUDO_LABELS_M3D[data['frame_id'][b_idx]] = {
                    'gt_boxes': gt_boxes,
                    'pred_box': pred_boxes,
                    'iou_pred': pred_iou_scores,
                }
    
    ps_path = os.path.join(ps_label_dir, f'ps_label_M3D.pkl')
    with open(ps_path, 'wb') as f:
        pickle.dump(PSEUDO_LABELS_M3D, f)

def train_fusion_epoch(model, optimizer, dataloader, epoch, epochs, cfg, batch_size, logger, ckpt_dir, ps_label_dir):
    # Record for one epoch
    train_total_loss = 0.0
    
    model.train()
    n_samples = 0
    with tqdm(dataloader, desc=f'Epoch {epoch + 1:02d}') as t:
        for x in t:
            data, label = x
            data = data.float().cuda().reshape(-1, cfg.LATE_FUSION.CHANNEL, 1)
            label = label.float().cuda()

            out = model(data)
            loss = F.binary_cross_entropy_with_logits(out.reshape(-1), label.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            n_samples += 1
            train_total_loss += loss.item()
            t.set_postfix({'loss': train_total_loss / n_samples})
    
    train_total_loss /= n_samples
    logger.info(f'=== Epoch [{epoch + 1}/{epochs}] ===')
    logger.info(f'[Train] Fusion model loss: {train_total_loss:.4f}')
    
    # Save model
    savepath = ckpt_dir / f'checkpoint_epoch_{epoch + 1}_fusion.pth'
    state = {
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(state, savepath)