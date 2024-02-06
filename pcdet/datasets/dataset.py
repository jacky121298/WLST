import copy
import torch
import numpy as np
import torch.utils.data as torch_data
from skimage import io
from pathlib import Path
from collections import defaultdict
from skimage.transform import resize
from typing import List, Tuple, Union
from nuscenes.utils.data_classes import Box
from shapely.geometry import MultiPoint, box
from pyquaternion.quaternion import Quaternion
from pcdet.utils import LidarPointCloud
from .augmentor.data_augmentor import DataAugmentor
from .processor.data_processor import DataProcessor
from .augmentor.data_augmentor import DataAugmentorWL
from nuscenes.utils.geometry_utils import view_points
from ..ops.roiaware_pool3d import roiaware_pool3d_utils
from .processor.point_feature_encoder import PointFeatureEncoder
from ..utils import common_utils, box_utils, self_training_utils, self_training_utils_wl

class DatasetTemplate(torch_data.Dataset):
    def __init__(self, dataset_cfg=None, class_names=None, training=True, root_path=None, logger=None):
        super().__init__()
        self.dataset_cfg = dataset_cfg
        self.training = training
        self.class_names = class_names
        self.logger = logger
        self.root_path = root_path if root_path is not None else Path(self.dataset_cfg.DATA_PATH)
        self.logger = logger
        if self.dataset_cfg is None or class_names is None:
            return

        self.point_cloud_range = np.array(self.dataset_cfg.POINT_CLOUD_RANGE, dtype=np.float32)
        self.point_feature_encoder = PointFeatureEncoder(
            self.dataset_cfg.POINT_FEATURE_ENCODING,
            point_cloud_range=self.point_cloud_range
        )
        self.data_augmentor = DataAugmentor(
            self.root_path, self.dataset_cfg.DATA_AUGMENTOR, self.class_names, logger=self.logger
        ) if self.training else None
        self.data_processor = DataProcessor(
            self.dataset_cfg.DATA_PROCESSOR, point_cloud_range=self.point_cloud_range, training=self.training
        )

        self.grid_size = self.data_processor.grid_size
        self.voxel_size = self.data_processor.voxel_size
        self.total_epochs = 0
        self._merge_all_iters_to_one_epoch = False

    @property
    def mode(self):
        return 'train' if self.training else 'test'

    def __getstate__(self):
        d = dict(self.__dict__)
        del d['logger']
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)

    @staticmethod
    def generate_prediction_dicts(batch_dict, pred_dicts, class_names, output_path=None):
        """
        To support a custom dataset, implement this function to receive the predicted results from the model, and then
        transform the unified normative coordinate to your required coordinate, and optionally save them to disk.

        Args:
            batch_dict: dict of original data from the dataloader
            pred_dicts: dict of predicted results from the model
                pred_boxes: (N, 7), Tensor
                pred_scores: (N), Tensor
                pred_labels: (N), Tensor
            class_names:
            output_path: if it is not None, save the results to this path
        Returns:

        """
        raise NotImplementedError

    @staticmethod
    def __vis__(points, gt_boxes, ref_boxes=None, scores=None, use_fakelidar=False):
        import visual_utils.visualize_utils as vis
        import mayavi.mlab as mlab
        gt_boxes = copy.deepcopy(gt_boxes)
        if use_fakelidar:
            gt_boxes = box_utils.boxes3d_kitti_lidar_to_fakelidar(gt_boxes)

        if ref_boxes is not None:
            ref_boxes = copy.deepcopy(ref_boxes)
            if use_fakelidar:
                ref_boxes = box_utils.boxes3d_kitti_lidar_to_fakelidar(ref_boxes)

        vis.draw_scenes(points, gt_boxes, ref_boxes=ref_boxes, ref_scores=scores)
        mlab.show(stop=True)

    @staticmethod
    def __vis_fake__(points, gt_boxes, ref_boxes=None, scores=None, use_fakelidar=True):
        import visual_utils.visualize_utils as vis
        import mayavi.mlab as mlab
        gt_boxes = copy.deepcopy(gt_boxes)
        if use_fakelidar:
            gt_boxes = box_utils.boxes3d_kitti_lidar_to_fakelidar(gt_boxes)

        if ref_boxes is not None:
            ref_boxes = copy.deepcopy(ref_boxes)
            if use_fakelidar:
                ref_boxes = box_utils.boxes3d_kitti_lidar_to_fakelidar(ref_boxes)

        vis.draw_scenes(points, gt_boxes, ref_boxes=ref_boxes, ref_scores=scores)
        mlab.show(stop=True)

    @staticmethod
    def extract_fov_data(points, fov_degree, heading_angle):
        """
        Args:
            points: (N, 3 + C)
            fov_degree: [0~180]
            heading_angle: [0~360] in lidar coords, 0 is the x-axis, increase clockwise
        Returns:
        """
        half_fov_degree = fov_degree / 180 * np.pi / 2
        heading_angle = -heading_angle / 180 * np.pi
        points_new = common_utils.rotate_points_along_z(
            points.copy()[np.newaxis, :, :], np.array([heading_angle])
        )[0]
        angle = np.arctan2(points_new[:, 1], points_new[:, 0])
        fov_mask = ((np.abs(angle) < half_fov_degree) & (points_new[:, 0] > 0))
        points = points_new[fov_mask]
        return points

    @staticmethod
    def extract_fov_gt(gt_boxes, fov_degree, heading_angle):
        """
        Args:
            anno_dict:
            fov_degree: [0~180]
            heading_angle: [0~360] in lidar coords, 0 is the x-axis, increase clockwise
        Returns:
        """
        half_fov_degree = fov_degree / 180 * np.pi / 2
        heading_angle = -heading_angle / 180 * np.pi
        gt_boxes_lidar = copy.deepcopy(gt_boxes)
        gt_boxes_lidar = common_utils.rotate_points_along_z(
            gt_boxes_lidar[np.newaxis, :, :], np.array([heading_angle])
        )[0]
        gt_boxes_lidar[:, 6] += heading_angle
        gt_angle = np.arctan2(gt_boxes_lidar[:, 1], gt_boxes_lidar[:, 0])
        fov_gt_mask = ((np.abs(gt_angle) < half_fov_degree) & (gt_boxes_lidar[:, 0] > 0))
        return fov_gt_mask

    def fill_pseudo_labels(self, input_dict):
        gt_boxes = self_training_utils.load_ps_label(input_dict['frame_id'])
        gt_scores = gt_boxes[:, 8]
        gt_classes = gt_boxes[:, 7]
        gt_boxes = gt_boxes[:, :7]

        # only suitable for only one classes, generating gt_names for prepare data
        gt_names = np.array([self.class_names[0] for n in gt_boxes])

        input_dict['gt_boxes'] = gt_boxes
        input_dict['gt_names'] = gt_names
        input_dict['gt_classes'] = gt_classes
        input_dict['gt_scores'] = gt_scores
        input_dict['pos_ps_bbox'] = (gt_classes > 0).sum()
        input_dict['ign_ps_bbox'] = gt_boxes.shape[0] - input_dict['pos_ps_bbox']
        input_dict.pop('num_points_in_gt', None)

    def fill_pseudo_labels_3d(self, input_dict):
        gt_boxes = self_training_utils_wl.load_ps_label_M3D(input_dict['frame_id'])
        if len(gt_boxes) == 0:
            return False
        # only suitable for only one classes, generating gt_names for prepare data
        gt_names = np.array([self.class_names[0] for n in gt_boxes])
        input_dict['gt_boxes'] = gt_boxes
        input_dict['gt_names'] = gt_names
        input_dict.pop('num_points_in_gt', None)
        return True

    def merge_all_iters_to_one_epoch(self, merge=True, epochs=None):
        if merge:
            self._merge_all_iters_to_one_epoch = True
            self.total_epochs = epochs
        else:
            self._merge_all_iters_to_one_epoch = False

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, index):
        """
        To support a custom dataset, implement this function to load the raw data (and labels), then transform them to
        the unified normative coordinate and call the function self.prepare_data() to process the data and send them
        to the model.

        Args:
            index:

        Returns:

        """
        raise NotImplementedError
    
    def rotz(self, angle):
        c = np.cos(angle)
        s = np.sin(angle)
        rotz = np.array([
            [c, -s, 0],
            [s,  c, 0],
            [0,  0, 1],
        ])
        return rotz

    def roty(self, angle):
        c = np.cos(angle)
        s = np.sin(angle)
        roty = np.array([
            [ c, 0, s],
            [ 0, 1, 0],
            [-s, 0, c],
        ])
        return roty

    def prepare_data_wl_kitti(self, input_dict):
        image = input_dict['image']
        pts_lidar = np.copy(input_dict['points'][:, :3])
        gt_boxes = np.copy(input_dict['gt_boxes'])
        if input_dict['dataset_cfg'].get('SHIFT_COOR', None):
            pts_lidar -= np.array(input_dict['dataset_cfg'].SHIFT_COOR, dtype=np.float32)
        
        pts_wl, rgb_wl, box_wl, seg_wl, fru_wl, lab_wl, theta_wl, image_wl = [], [], [], [], [], [], [], []
        pts_img, pts_depth = input_dict['calib'].lidar_to_img(pts_lidar)

        for i, bbox in enumerate(input_dict['bbox']):
            if input_dict['gt_names'][i] not in input_dict['dataset_cfg'].CLASS_NAMES:
                continue

            val_flag_1 = np.logical_and(pts_img[:, 0] >= bbox[0], pts_img[:, 0] <= bbox[2])
            val_flag_2 = np.logical_and(pts_img[:, 1] >= bbox[1], pts_img[:, 1] <= bbox[3])
            val_flag_merge = np.logical_and(val_flag_1, val_flag_2)
            
            pts_valid_flag = np.logical_and(val_flag_merge, pts_depth >= 0)
            pts = pts_lidar[pts_valid_flag]
            if len(pts) == 0:
                continue
            if input_dict['dataset_cfg'].get('SHIFT_COOR', None):
                pts += np.array(input_dict['dataset_cfg'].SHIFT_COOR, dtype=np.float32)
            
            seg = roiaware_pool3d_utils.points_in_boxes_cpu(pts, gt_boxes[np.newaxis, i]).squeeze(axis=0)
            if np.count_nonzero(seg) == 0:
                continue

            img_idx = pts_img[pts_valid_flag].astype(np.int) # (N, 2)
            pts_rgb = image[img_idx[:, 1], img_idx[:, 0], :] # (N, 3)
            image_wl.append([
                str(input_dict['image_path']),
                str(bbox[1]),
                str(bbox[3]),
                str(bbox[0]),
                str(bbox[2])
            ])
            seg_wl.append(seg)
            lab_wl.append(input_dict['gt_names'][i])

            if input_dict['dataset_cfg'].WEAK_LABEL.TRANSFER_TO_CENTER:
                u = (bbox[0] + bbox[2]) / 2
                v = (bbox[1] + bbox[3]) / 2
                u = np.array(u).reshape(1, 1)
                v = np.array(v).reshape(1, 1)
                depth_rect = np.array(input_dict['dataset_cfg'].WEAK_LABEL.FRUSTUM_DIS).reshape(1, 1)
                pt = input_dict['calib'].img_to_rect(u, v, depth_rect=depth_rect).squeeze()
                
                theta = np.arctan2(-pt[0], pt[2])
                phi = gt_boxes[i][6]
                
                pts = input_dict['calib'].lidar_to_rect(pts)
                pts = (self.roty(theta) @ pts.T).T
                pts = input_dict['calib'].rect_to_lidar(pts)

                gt_boxes[i][:3] = input_dict['calib'].lidar_to_rect(gt_boxes[i][:3].reshape(1, 3)).squeeze()
                gt_boxes[i][:3] = np.squeeze(self.roty(theta) @ gt_boxes[i][:3].reshape(3, 1))
                gt_boxes[i][:3] = input_dict['calib'].rect_to_lidar(gt_boxes[i][:3].reshape(1, 3)).squeeze()
                gt_boxes[i][6] = phi - theta

                theta_wl.append(theta)

            pts_wl.append(pts)
            rgb_wl.append(pts_rgb)
            box_wl.append(gt_boxes[i])

            fru = []
            border = [(bbox[0], bbox[1]), (bbox[2], bbox[1]), (bbox[0], bbox[3]), (bbox[2], bbox[3])]
            for b in border:
                u = np.array(b[0]).reshape(1, 1)
                v = np.array(b[1]).reshape(1, 1)
                depth_rect = np.array(input_dict['dataset_cfg'].WEAK_LABEL.FRUSTUM_DIS).reshape(1, 1)
                pt_rect = input_dict['calib'].img_to_rect(u, v, depth_rect=depth_rect)
                pt = input_dict['calib'].rect_to_lidar(pt_rect).squeeze()
                if input_dict['dataset_cfg'].WEAK_LABEL.TRANSFER_TO_CENTER:
                    pt = input_dict['calib'].lidar_to_rect(pt.reshape(1, 3)).squeeze()
                    pt = (self.roty(theta) @ pt.reshape(3, 1)).squeeze()
                    pt = input_dict['calib'].rect_to_lidar(pt.reshape(1, 3)).squeeze()
                if input_dict['dataset_cfg'].get('SHIFT_COOR', None):
                    pt += np.array(input_dict['dataset_cfg'].SHIFT_COOR, dtype=np.float32)
                fru.append(pt)

            pt = input_dict['calib'].rect_to_lidar(np.array([0, 0, 0]).reshape(1, 3)).squeeze()
            if input_dict['dataset_cfg'].get('SHIFT_COOR', None):
                pt += np.array(input_dict['dataset_cfg'].SHIFT_COOR, dtype=np.float32)
            fru.append(pt)
            fru_wl.append(fru)

        input_dict['pts_wl'] = pts_wl
        input_dict['rgb_wl'] = rgb_wl
        input_dict['box_wl'] = np.array(box_wl)
        input_dict['seg_wl'] = seg_wl
        input_dict['fru_wl'] = np.array(fru_wl)
        input_dict['lab_wl'] = lab_wl
        input_dict['theta_wl'] = np.array(theta_wl)
        input_dict['image_wl'] = np.array(image_wl, dtype=object)

        input_dict.pop('bbox', None)
        input_dict.pop('image', None)
        input_dict.pop('dataset_cfg', None)
        return input_dict

    def prepare_data_wl_waymo(self, input_dict):
        image = input_dict['image']
        pts_veh = np.copy(input_dict['points'][:, :3])
        gt_boxes = np.copy(input_dict['gt_boxes'])

        pts_wl, rgb_wl, box_wl, seg_wl, fru_wl, lab_wl, theta_wl, image_wl = [], [], [], [], [], [], [], []
        for i, bbox in enumerate(input_dict['bbox']):
            if input_dict['bbox_where'][i] == 'NONE':
                continue
            if input_dict['dataset_cfg'].WEAK_LABEL.CAMERA_FRONT_ONLY and input_dict['bbox_where'][i] != 'CAMERA_FRONT':
                continue
            if input_dict['num_points_in_gt'][i] == 0:
                continue
            if input_dict['gt_names'][i] not in input_dict['dataset_cfg'].CLASS_NAMES:
                continue
            if bbox[2] < 20.0 or bbox[3] < 20.0:
                continue

            pts_img, pts_valid = input_dict['calib'].veh_to_img(pts_veh, input_dict['bbox_where'][i], input_dict['pose'], input_dict['image_shape'])
            val_flag_1 = np.logical_and(pts_img[:, 0] >= (bbox[0] - bbox[3] / 2), pts_img[:, 0] <= (bbox[0] + bbox[3] / 2))
            val_flag_2 = np.logical_and(pts_img[:, 1] >= (bbox[1] - bbox[2] / 2), pts_img[:, 1] <= (bbox[1] + bbox[2] / 2))
            val_flag_merge = np.logical_and(val_flag_1, val_flag_2)

            pts_valid_flag = np.logical_and(val_flag_merge, pts_valid.astype(np.bool))
            pts = pts_veh[pts_valid_flag]
            if len(pts) == 0:
                continue

            seg = roiaware_pool3d_utils.points_in_boxes_cpu(pts, gt_boxes[np.newaxis, i]).squeeze(axis=0)
            if np.count_nonzero(seg) == 0:
                continue

            img_idx = pts_img[pts_valid_flag].astype(np.int) # (N, 2)
            pts_rgb = image[input_dict['bbox_where'][i]][img_idx[:, 1], img_idx[:, 0], :] # (N, 3)
            
            image_wl.append([
                str(input_dict['image_path'][input_dict['bbox_where'][i]]),
                str(bbox[1] - bbox[2] / 2),
                str(bbox[1] + bbox[2] / 2),
                str(bbox[0] - bbox[3] / 2),
                str(bbox[0] + bbox[3] / 2)
            ])
            seg_wl.append(seg)
            lab_wl.append(input_dict['gt_names'][i])

            if input_dict['dataset_cfg'].WEAK_LABEL.TRANSFER_TO_CENTER:
                u = np.array(bbox[0]).reshape(1, 1)
                v = np.array(bbox[1]).reshape(1, 1)
                depth = np.array(input_dict['dataset_cfg'].WEAK_LABEL.FRUSTUM_DIS).reshape(1, 1)
                pt = input_dict['calib'].img_to_rect(u, v, depth, input_dict['bbox_where'][i], input_dict['pose'], input_dict['image_shape']).squeeze()
                
                theta = np.arctan2(pt[1], pt[0])
                phi = gt_boxes[i][6]
                
                pts = input_dict['calib'].veh_to_rect(pts)
                pts = (self.rotz(-theta) @ pts.T).T
                pts = input_dict['calib'].rect_to_veh(pts)

                gt_boxes[i][:3] = input_dict['calib'].veh_to_rect(gt_boxes[i][:3].reshape(1, 3)).squeeze()
                gt_boxes[i][:3] = np.squeeze(self.rotz(-theta) @ gt_boxes[i][:3].reshape(3, 1))
                gt_boxes[i][:3] = input_dict['calib'].rect_to_veh(gt_boxes[i][:3].reshape(1, 3)).squeeze()
                gt_boxes[i][6] = phi - theta

                theta_wl.append(theta)

            pts_wl.append(pts)
            rgb_wl.append(pts_rgb)
            box_wl.append(gt_boxes[i])

            fru = []
            border = [
                (bbox[0] - bbox[3] / 2, bbox[1] - bbox[2] / 2),
                (bbox[0] + bbox[3] / 2, bbox[1] - bbox[2] / 2),
                (bbox[0] - bbox[3] / 2, bbox[1] + bbox[2] / 2),
                (bbox[0] + bbox[3] / 2, bbox[1] + bbox[2] / 2),
            ]
            for b in border:
                u = np.array(b[0]).reshape(1, 1)
                v = np.array(b[1]).reshape(1, 1)
                depth = np.array(input_dict['dataset_cfg'].WEAK_LABEL.FRUSTUM_DIS).reshape(1, 1)
                pt = input_dict['calib'].img_to_veh(u, v, depth, input_dict['bbox_where'][i], input_dict['pose'], input_dict['image_shape']).squeeze()
                if input_dict['dataset_cfg'].WEAK_LABEL.TRANSFER_TO_CENTER:
                    pt = input_dict['calib'].veh_to_rect(pt.reshape(1, 3)).squeeze()
                    pt = (self.rotz(-theta) @ pt.reshape(3, 1)).squeeze()
                    pt = input_dict['calib'].rect_to_veh(pt.reshape(1, 3)).squeeze()
                fru.append(pt)
            
            pt = input_dict['calib'].rect_to_veh(np.array([0, 0, 0]).reshape(1, 3)).squeeze()
            fru.append(pt)
            fru_wl.append(fru)

        input_dict['pts_wl'] = pts_wl
        input_dict['rgb_wl'] = rgb_wl
        input_dict['box_wl'] = np.array(box_wl)
        input_dict['seg_wl'] = seg_wl
        input_dict['fru_wl'] = np.array(fru_wl)
        input_dict['lab_wl'] = lab_wl
        input_dict['theta_wl'] = np.array(theta_wl)
        input_dict['image_wl'] = np.array(image_wl, dtype=object)

        input_dict.pop('bbox', None)
        input_dict.pop('image', None)
        input_dict.pop('calib', None)
        input_dict.pop('bbox_where', None)
        input_dict.pop('image_shape', None)
        input_dict.pop('dataset_cfg', None)
        return input_dict

    def post_process_coords(self, corner_coords: List, imsize: Tuple[int, int]=(1600, 900)) -> Union[Tuple[float, float, float, float], None]:
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

    def lidar_to_camera_box(self, box, lidar_cs_rec, lidar_pose_rec, camera_cs_rec, camera_pose_rec):
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

    def camera_to_lidar_box(self, box, lidar_cs_rec, lidar_pose_rec, camera_cs_rec, camera_pose_rec):
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

    def lidar_to_camera_pc(self, pc, lidar_cs_rec, lidar_pose_rec, camera_cs_rec, camera_pose_rec):
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

    def camera_to_lidar_pc(self, pc, lidar_cs_rec, lidar_pose_rec, camera_cs_rec, camera_pose_rec):
        # Move to camera ego vehicle coord system
        pc.rotate(Quaternion(camera_cs_rec['rotation']).rotation_matrix)
        pc.translate(np.array(camera_cs_rec['translation']))

        # Move to global coord system
        pc.rotate(Quaternion(camera_pose_rec['rotation']).rotation_matrix)
        pc.translate(np.array(camera_pose_rec['translation']))

        # Move to lidar ego vehicle coord system
        pc.translate(-np.array(lidar_pose_rec['translation']))
        pc.rotate(Quaternion(lidar_pose_rec['rotation']).rotation_matrix.T)
        
        # Move to lidar coord system
        pc.translate(-np.array(lidar_cs_rec['translation']))
        pc.rotate(Quaternion(lidar_cs_rec['rotation']).rotation_matrix.T)
        return pc

    def get_corners(self, box):
        x, y, z = box.center
        w, l, h = box.wlh
        r = box.orientation.radians

        x_corners = w / 2 * np.array([1,  1,  1,  1, -1, -1, -1, -1])
        y_corners = h / 2 * np.array([1, -1, -1,  1,  1, -1, -1,  1])
        z_corners = l / 2 * np.array([1,  1, -1, -1,  1,  1, -1, -1])
        corners = np.vstack((x_corners, y_corners, z_corners))

        corners = np.dot(self.roty(-r), corners)
        corners[0, :] = corners[0, :] + x
        corners[1, :] = corners[1, :] + y
        corners[2, :] = corners[2, :] + z
        return corners

    def prepare_data_wl_nuscenes(self, input_dict, nusc):
        pts_lidar = np.copy(input_dict['points'][:, :3])
        gt_boxes = np.copy(input_dict['gt_boxes'])
        if input_dict['dataset_cfg'].get('SHIFT_COOR', None):
            pts_lidar -= np.array(input_dict['dataset_cfg'].SHIFT_COOR, dtype=np.float32)
            gt_boxes[:, :3] -= np.array(input_dict['dataset_cfg'].SHIFT_COOR, dtype=np.float32)
        
        sample_token = input_dict['metadata']['token']
        s_record = nusc.get('sample', sample_token)
        
        lidar_token = s_record['data']['LIDAR_TOP']
        camera_name = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
        camera_token = [s_record['data'][n] for n in camera_name]
        len_c = len(camera_name)

        image_path = [nusc.get_sample_data(camera_token[i])[0] for i in range(len_c)]
        image_path = [Path(image_path[i]).relative_to(Path(self.dataset_cfg.DATA_PATH) / self.dataset_cfg.VERSION).__str__() \
            for i in range(len_c)]
        image = [io.imread(self.root_path / image_path[i]) for i in range(len_c)]

        lidar_record = nusc.get('sample_data', lidar_token)
        camera_record = [nusc.get('sample_data', t) for t in camera_token]

        lidar_cs_rec = nusc.get('calibrated_sensor', lidar_record['calibrated_sensor_token'])
        lidar_pose_rec = nusc.get('ego_pose', lidar_record['ego_pose_token'])

        camera_cs_rec = [nusc.get('calibrated_sensor', r['calibrated_sensor_token']) for r in camera_record]
        camera_pose_rec = [nusc.get('ego_pose', r['ego_pose_token']) for r in camera_record]

        pts_wl, rgb_wl, box_wl, seg_wl, fru_wl, lab_wl, theta_wl, image_wl, where_wl = [], [], [], [], [], [], [], [], []
        pts_lidar = LidarPointCloud.from_points(pts_lidar.T)
        pts_camera = [copy.deepcopy(pts_lidar) for _ in range(len_c)]
        pts_camera = [self.lidar_to_camera_pc(pts_camera[i], lidar_cs_rec, lidar_pose_rec, camera_cs_rec[i], camera_pose_rec[i]) \
            for i in range(len_c)]
        pts_depth = [pts_camera[i].points[2, :] for i in range(len_c)]
        
        camera_intrinsic = [np.array(camera_cs_rec[i]['camera_intrinsic']) for i in range(len_c)]
        pts_img = [np.copy(view_points(pts_camera[i].points, camera_intrinsic[i], normalize=True)) for i in range(len_c)]
        pts_img = [pts_img[i][:2, :].T for i in range(len_c)] # 6 * (N, 2)

        # For cam_intrinsic
        fu = [camera_intrinsic[i][0, 0] for i in range(len_c)]
        fv = [camera_intrinsic[i][1, 1] for i in range(len_c)]
        cu = [camera_intrinsic[i][0, 2] for i in range(len_c)]
        cv = [camera_intrinsic[i][1, 2] for i in range(len_c)]

        angles = [90, 145, 35, 270, 200, 340]
        angles = [a / 180 * np.pi for a in angles]

        for i, box in enumerate(gt_boxes):
            if input_dict['gt_names'][i] not in input_dict['dataset_cfg'].CLASS_NAMES:
                continue

            box2d = [Box(box[:3], box[[4, 3, 5]], Quaternion(axis=[0, 0, 1], radians=box[6])) for _ in range(len_c)]
            box2d = [self.lidar_to_camera_box(box2d[j], lidar_cs_rec, lidar_pose_rec, camera_cs_rec[j], camera_pose_rec[j]) \
                for j in range(len_c)]

            # Filter out the corners that are not in front of the calibrated sensor
            # corners_3d = self.get_corners(box2d)
            corners_3d = [box2d[j].corners() for j in range(len_c)]
            in_front = [np.argwhere(corners_3d[j][2, :] > 0).flatten() for j in range(len_c)]
            corners_3d = [corners_3d[j][:, in_front[j]] for j in range(len_c)]

            # Project 3d box to 2d
            corner_coords = [view_points(corners_3d[j], camera_intrinsic[j], True).T[:, :2].tolist() for j in range(len_c)]

            # Keep only corners that fall within the image
            final_coords = [self.post_process_coords(corner_coords[j], (image[j].shape[1], image[j].shape[0])) for j in range(len_c)]

            # Skip if the convex hull of the re-projected corners does not intersect the image canvas
            for j in range(len_c):
                if final_coords[j] is None:
                    continue
                min_x, min_y, max_x, max_y = final_coords[j]

                val_flag_1 = np.logical_and(pts_img[j][:, 0] >= min_x, pts_img[j][:, 0] <= max_x)
                val_flag_2 = np.logical_and(pts_img[j][:, 1] >= min_y, pts_img[j][:, 1] <= max_y)
                val_flag_merge = np.logical_and(val_flag_1, val_flag_2)
                
                pts_valid_flag = np.logical_and(val_flag_merge, pts_depth[j] >= 0)
                pts = np.copy(pts_lidar.points.T[pts_valid_flag]) # (N, 3)
                if len(pts) == 0:
                    continue
                
                seg = roiaware_pool3d_utils.points_in_boxes_cpu(pts, box[np.newaxis, :7]).squeeze(axis=0)
                if np.count_nonzero(seg) == 0:
                    continue

                img_idx = pts_img[j][pts_valid_flag].astype(np.int) # (N, 2)
                pts_rgb = image[j][img_idx[:, 1], img_idx[:, 0], :] # (N, 3)
                image_wl.append([
                    str(input_dict['image_path']),
                    str(min_y),
                    str(max_y),
                    str(min_x),
                    str(max_x)
                ])
                seg_wl.append(seg)
                lab_wl.append(input_dict['gt_names'][i])
                where_wl.append(j)

                pts = LidarPointCloud.from_points(pts.T)
                if input_dict['dataset_cfg'].WEAK_LABEL.TRANSFER_TO_CENTER:
                    u = (min_x + max_x) / 2
                    v = (min_y + max_y) / 2
                    depth_rect = input_dict['dataset_cfg'].WEAK_LABEL.FRUSTUM_DIS

                    x = ((u - cu[j]) * depth_rect) / fu[j]
                    y = ((v - cv[j]) * depth_rect) / fv[j]
                    pt = np.array([x, y, depth_rect])
                    theta = np.arctan2(-pt[0], pt[2])

                    pts = self.lidar_to_camera_pc(pts, lidar_cs_rec, lidar_pose_rec, camera_cs_rec[j], camera_pose_rec[j])
                    pts.rotate(Quaternion(axis=[0, 1, 0], radians=theta + angles[j]).rotation_matrix)
                    pts = self.camera_to_lidar_pc(pts, lidar_cs_rec, lidar_pose_rec, camera_cs_rec[j], camera_pose_rec[j])
                    
                    box2d[j].rotate(Quaternion(axis=[0, 1, 0], radians=theta + angles[j]))
                    theta_wl.append(theta + angles[j])

                box2d[j] = self.camera_to_lidar_box(box2d[j], lidar_cs_rec, lidar_pose_rec, camera_cs_rec[j], camera_pose_rec[j])
                if input_dict['dataset_cfg'].get('SHIFT_COOR', None):
                    pts.points += np.array(input_dict['dataset_cfg'].SHIFT_COOR, dtype=np.float32).reshape((3, 1))
                    box2d[j].center += np.array(input_dict['dataset_cfg'].SHIFT_COOR, dtype=np.float32)

                box2d[j] = np.concatenate((box2d[j].center.reshape((1, 3)), box2d[j].wlh.reshape((1, 3)), np.array(box2d[j].orientation.yaw_pitch_roll[0]).reshape((1, 1))), axis=1)
                box2d[j] = np.squeeze(box2d[j])[[0, 1, 2, 4, 3, 5, 6]]

                pts_wl.append(np.copy(pts.points.T))
                rgb_wl.append(np.copy(pts_rgb))
                box_wl.append(box2d[j])

                fru = []
                border = [(min_x, min_y), (min_x, max_y), (max_x, min_y), (max_x, max_y)]
                for b in border:
                    u = b[0]
                    v = b[1]
                    depth_rect = input_dict['dataset_cfg'].WEAK_LABEL.FRUSTUM_DIS

                    x = ((u - cu[j]) * depth_rect) / fu[j]
                    y = ((v - cv[j]) * depth_rect) / fv[j]
                    pt = np.array([x, y, depth_rect]).reshape((3, -1))
                    pt = LidarPointCloud.from_points(pt)
                    pt = self.camera_to_lidar_pc(pt, lidar_cs_rec, lidar_pose_rec, camera_cs_rec[j], camera_pose_rec[j])

                    if input_dict['dataset_cfg'].WEAK_LABEL.TRANSFER_TO_CENTER:
                        pt = self.lidar_to_camera_pc(pt, lidar_cs_rec, lidar_pose_rec, camera_cs_rec[j], camera_pose_rec[j])
                        pt.rotate(Quaternion(axis=[0, 1, 0], radians=theta + angles[j]).rotation_matrix)
                        pt = self.camera_to_lidar_pc(pt, lidar_cs_rec, lidar_pose_rec, camera_cs_rec[j], camera_pose_rec[j])
                    if input_dict['dataset_cfg'].get('SHIFT_COOR', None):
                        pt.points += np.array(input_dict['dataset_cfg'].SHIFT_COOR, dtype=np.float32).reshape((3, 1))
                    fru.append(pt.points.squeeze())

                pt = np.array([0.0, 0.0, 0.0]).reshape((3, -1))
                pt = LidarPointCloud.from_points(pt)
                pt = self.camera_to_lidar_pc(pt, lidar_cs_rec, lidar_pose_rec, camera_cs_rec[j], camera_pose_rec[j])
                if input_dict['dataset_cfg'].get('SHIFT_COOR', None):
                    pt.points += np.array(input_dict['dataset_cfg'].SHIFT_COOR, dtype=np.float32).reshape((3, 1))
                fru.append(pt.points.squeeze())
                fru_wl.append(fru)

        input_dict['pts_wl'] = pts_wl
        input_dict['rgb_wl'] = rgb_wl
        input_dict['box_wl'] = np.array(box_wl)
        input_dict['seg_wl'] = seg_wl
        input_dict['fru_wl'] = np.array(fru_wl)
        input_dict['lab_wl'] = lab_wl
        input_dict['theta_wl'] = np.array(theta_wl)
        input_dict['image_wl'] = np.array(image_wl, dtype=object)
        input_dict['where_wl'] = np.array(where_wl)

        input_dict.pop('image', None)
        input_dict.pop('image_path', None)
        input_dict.pop('dataset_cfg', None)
        input_dict.pop('cam_intrinsic', None)
        return input_dict

    def prepare_data(self, data_dict):
        """
        Args:
            data_dict:
                points: (N, 3 + C_in)
                gt_boxes: optional, (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
                gt_names: optional, (N), string
                ...

        Returns:
            data_dict:
                frame_id: string
                points: (N, 3 + C_in)
                gt_boxes: optional, (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
                gt_names: optional, (N), string
                use_lead_xyz: bool
                voxels: optional (num_voxels, max_points_per_voxel, 3 + C)
                voxel_coords: optional (num_voxels, 3)
                voxel_num_points: optional (num_voxels)
                ...
        """
        if self.training:
            # filter gt_boxes without points
            num_points_in_gt = data_dict.get('num_points_in_gt', None)
            if num_points_in_gt is None:
                num_points_in_gt = roiaware_pool3d_utils.points_in_boxes_cpu(
                    torch.from_numpy(data_dict['points'][:, :3]),
                    torch.from_numpy(data_dict['gt_boxes'][:, :7])).numpy().sum(axis=1)

            mask = (num_points_in_gt >= self.dataset_cfg.get('MIN_POINTS_OF_GT', 1))
            data_dict['gt_boxes'] = data_dict['gt_boxes'][mask]
            data_dict['gt_names'] = data_dict['gt_names'][mask]
            if 'gt_classes' in data_dict:
                data_dict['gt_classes'] = data_dict['gt_classes'][mask]
                data_dict['gt_scores'] = data_dict['gt_scores'][mask]

            assert 'gt_boxes' in data_dict, 'gt_boxes should be provided for training'
            gt_boxes_mask = np.array([n in self.class_names for n in data_dict['gt_names']], dtype=np.bool_)

            data_dict = self.data_augmentor.forward(
                data_dict={
                    **data_dict,
                    'gt_boxes_mask': gt_boxes_mask
                }
            )
            if len(data_dict['gt_boxes']) == 0:
                new_index = np.random.randint(self.__len__())
                return self.__getitem__(new_index)

        if data_dict.get('gt_boxes', None) is not None:
            selected = common_utils.keep_arrays_by_name(data_dict['gt_names'], self.class_names)
            data_dict['gt_boxes'] = data_dict['gt_boxes'][selected]
            data_dict['gt_names'] = data_dict['gt_names'][selected]
            # for pseudo label has ignore labels.
            if 'gt_classes' not in data_dict:
                gt_classes = np.array([self.class_names.index(n) + 1 for n in data_dict['gt_names']], dtype=np.int32)
            else:
                gt_classes = data_dict['gt_classes'][selected]
                data_dict['gt_scores'] = data_dict['gt_scores'][selected]
            gt_boxes = np.concatenate((data_dict['gt_boxes'], gt_classes.reshape(-1, 1).astype(np.float32)), axis=1)
            data_dict['gt_boxes'] = gt_boxes

        data_dict = self.point_feature_encoder.forward(data_dict)

        data_dict = self.data_processor.forward(
            data_dict=data_dict
        )
        data_dict.pop('gt_names', None)
        data_dict.pop('gt_classes', None)

        return data_dict

    @staticmethod
    def collate_batch(batch_list, _unused=False):
        data_dict = defaultdict(list)
        for cur_sample in batch_list:
            for key, val in cur_sample.items():
                data_dict[key].append(val)
        batch_size = len(batch_list)
        ret = {}

        for key, val in data_dict.items():
            try:
                if key in ['voxels', 'voxel_num_points']:
                    ret[key] = np.concatenate(val, axis=0)
                elif key in ['points', 'voxel_coords']:
                    coors = []
                    for i, coor in enumerate(val):
                        coor_pad = np.pad(coor, ((0, 0), (1, 0)), mode='constant', constant_values=i)
                        coors.append(coor_pad)
                    ret[key] = np.concatenate(coors, axis=0)
                elif key in ['gt_boxes']:
                    max_gt = max([len(x) for x in val])
                    batch_gt_boxes3d = np.zeros((batch_size, max_gt, val[0].shape[-1]), dtype=np.float32)
                    for k in range(batch_size):
                        batch_gt_boxes3d[k, :val[k].__len__(), :] = val[k]
                    ret[key] = batch_gt_boxes3d
                elif key in ['gt_scores']:
                    max_gt = max([len(x) for x in val])
                    batch_scores = np.zeros((batch_size, max_gt), dtype=np.float32)
                    for k in range(batch_size):
                        batch_scores[k, :val[k].__len__()] = val[k]
                    ret[key] = batch_scores
                elif key in ['pts_wl', 'rgb_wl']:
                    coors = []
                    for i, batch in enumerate(val):
                        for j, p in enumerate(batch):
                            p_pad = np.pad(p, ((0, 0), (1, 0)), mode='constant', constant_values=j)
                            p_pad = np.pad(p_pad, ((0, 0), (1, 0)), mode='constant', constant_values=i)
                            coors.append(p_pad)
                    if len(coors) > 0:
                        ret[key] = np.concatenate(coors, axis=0)
                    else:
                        ret[key] = np.array([])
                elif key in ['box_wl']:
                    max_gt = max([len(x) for x in val])
                    batch_gt_boxes3d = np.zeros((batch_size, max_gt, 7), dtype=np.float32)
                    for k in range(batch_size):
                        if val[k].shape[0] > 0:
                            batch_gt_boxes3d[k, :val[k].__len__(), :] = val[k]
                    ret[key] = batch_gt_boxes3d
                elif key in ['fru_wl']:
                    max_gt = max([len(x) for x in val])
                    batch_fru = np.zeros((batch_size, max_gt, 5, 3), dtype=np.float32)
                    for k in range(batch_size):
                        if val[k].shape[0] > 0:
                            batch_fru[k, :val[k].__len__(), ...] = val[k]
                    ret[key] = batch_fru
                elif key in ['image_wl']:
                    max_gt = max([len(x) for x in val])
                    batch_img = np.zeros((batch_size, max_gt, 5), dtype=object)
                    for k in range(batch_size):
                        if val[k].shape[0] > 0:
                            batch_img[k, :val[k].__len__(), ...] = val[k]
                    ret[key] = batch_img
                elif key in ['seg_wl']:
                    segs = []
                    for i, batch in enumerate(val):
                        for j, s in enumerate(batch):
                            s_pad = np.pad(s[:, np.newaxis], ((0, 0), (1, 0)), mode='constant', constant_values=j)
                            s_pad = np.pad(s_pad, ((0, 0), (1, 0)), mode='constant', constant_values=i)
                            segs.append(s_pad)
                    if len(segs) > 0:
                        ret[key] = np.concatenate(segs, axis=0)
                    else:
                        ret[key] = np.array([])
                elif key in ['theta_wl']:
                    max_gt = max([len(x) for x in val])
                    batch_theta = np.zeros((batch_size, max_gt), dtype=np.float32)
                    for k in range(batch_size):
                        if len(val[k]) > 0:
                            batch_theta[k, :val[k].__len__()] = val[k]
                    ret[key] = batch_theta
                elif key in ['where_wl']:
                    max_gt = max([len(x) for x in val])
                    batch_where = np.zeros((batch_size, max_gt), dtype=np.int)
                    for k in range(batch_size):
                        batch_where[k, :val[k].__len__()] = val[k]
                    ret[key] = batch_where
                elif key in ['lab_wl']:
                    pass
                else:
                    ret[key] = np.stack(val, axis=0)
            except:
                print('Error in collate_batch: key=%s' % key)
                raise TypeError

        ret['batch_size'] = batch_size
        return ret

    def eval(self):
        self.training = False
        self.data_processor.eval()

    def train(self):
        self.training = True
        self.data_processor.train()