import os
import cv2
import time
import yaml
import pickle
import pathlib
import argparse
import traceback
import numpy as np

from easydict import EasyDict
from multiprocessing import Pool
from utils import kitti_utils_official

def seg(s, seg_path, dataset_path, config):
    thresh_ransac = config.THRESH_RANSAC
    thresh_seg_max = config.THRESH_SEG_MAX
    ratio = config.REGION_GROWTH_RATIO

    total_object_number = 0
    iou_collection = []

    # Path to dataset
    image_path = os.path.join(dataset_path, f'data_object_image_2/training/image_2/{s}.png')
    lidar_path = os.path.join(dataset_path, f'data_object_velodyne/training/velodyne/{s}.bin')
    calib_path = os.path.join(dataset_path, f'data_object_calib/training/calib/{s}.txt')
    label_path = os.path.join(dataset_path, f'data_object_label_2/training/label_2/{s}.txt')

    img = cv2.imread(image_path)
    calib = kitti_utils_official.read_obj_calibration(calib_path)
    objects = kitti_utils_official.read_obj_data(label_path, calib, img.shape)

    dic = {
        'calib': calib,
        'shape': img.shape,
        'sample': {},
    }

    if len(objects) == 0:
        with open(os.path.join(seg_path, f'{s}.pickle'), 'wb') as f:
            pickle.dump(dic, f)
        print(f'{s}: none')
        return

    pc_all, object_filter_all = kitti_utils_official.get_point_cloud_my_version(lidar_path, calib, img.shape, back_cut=False)
    mask_ground_all, ground_sample_points = kitti_utils_official.calculate_ground(lidar_path, calib, img.shape, 0.2, back_cut=False)

    z_list = []
    index_list = []
    valid_list = []
    valid_index = []

    for i in range(len(objects)):
        total_object_number += 1
        flag = 1

        _, object_filter = kitti_utils_official.get_point_cloud_my_version(lidar_path, calib, img.shape, [objects[i].boxes[0]], back_cut=False)
        pc = pc_all[object_filter == 1]
        
        filter_sample = kitti_utils_official.calculate_gt_point_number(pc, objects[i].corners)
        pc_in_box_3d = pc[filter_sample]
        if len(pc_in_box_3d) < 30:
            flag = 0

        if flag == 1:
            valid_list.append(i)

        z_list.append(np.median(pc[:, 2]))
        index_list.append(i)

    sort = np.argsort(np.array(z_list))
    object_list = list(np.array(index_list)[sort])
    mask_object = np.ones((pc_all.shape[0]))

    dic = {
        'calib': calib,
        'shape': img.shape,
        'ground_sample': ground_sample_points,
        'sample': {}
    }

    for i in object_list:
        result = np.zeros((7, 2))
        count = 0
        mask_seg_list = []

        for j in range(thresh_seg_max):
            thresh = (j + 1) * 0.1
            _, object_filter = kitti_utils_official.get_point_cloud_my_version(
                lidar_path, calib, img.shape, [objects[i].boxes[0]], back_cut=False)
            
            filter_z = pc_all[:, 2] > 0
            mask_search = mask_ground_all * object_filter_all * mask_object * filter_z
            mask_origin = mask_ground_all * object_filter * mask_object * filter_z
            mask_seg = kitti_utils_official.region_grow_my_version(pc_all.copy(), 
                                                                   mask_search, mask_origin, thresh, ratio)
            if mask_seg.sum() == 0:
                continue

            if j >= 1:
                mask_seg_old = mask_seg_list[-1]
                if mask_seg_old.sum() != (mask_seg * mask_seg_old).sum():
                    count += 1
            result[count, 0] = j  
            result[count, 1] = mask_seg.sum()
            mask_seg_list.append(mask_seg)
            
        best_j = result[np.argmax(result[:, 1]), 0]
        try:
            mask_seg_best = mask_seg_list[int(best_j)]
            mask_object *= (1 - mask_seg_best)
            pc = pc_all[mask_seg_best == 1].copy()
        except IndexError:
            # print('bad region grow result! deprecated')
            continue
        if i not in valid_list:
            continue

        if kitti_utils_official.check_truncate(img.shape, objects[i].boxes_origin[0].box):
            # print('object %d truncates in %s, with bbox %s' % (i, s, str(objects[i].boxes_origin[0].box)))
            mask_origin_new = mask_seg_best
            mask_search_new = mask_ground_all
            thresh_new      = (best_j + 1) * 0.1

            mask_seg_for_truncate = kitti_utils_official.region_grow_my_version(
                pc_all.copy(),
                mask_search_new,
                mask_origin_new,
                thresh_new,
                ratio=None,
            )
            pc_truncate = pc_all[mask_seg_for_truncate == 1].copy()

            dic['sample'][i] = {
                'truncate': True,
                'object': objects[i],
                'pc': pc_truncate
            }
        
        else:
            dic['sample'][i] = {
                'truncate': False,
                'object': objects[i],
                'pc': pc
            }

    with open(os.path.join(seg_path, f'{s}.pickle'), 'wb') as f:
        pickle.dump(dic, f)
    
    print(f'{s}: done')
    return

if __name__ == '__main__':
    # python3 segmentation.py --dataset_path /home/master/10/jacky1212/dataset/kitti_fgr --seg_path ./kitti/segmentation --split_path ../../../data/kitti/ImageSets/val.txt
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', help='Path to 3D object detection dataset.')
    parser.add_argument('--seg_path', help='Path to segmentation results.')
    parser.add_argument('--split_path', help='Path to split file.')
    parser.add_argument('--processes', default=32, type=int, help='Number of parallel processes.')
    args = parser.parse_args()

    if not os.path.isdir(args.seg_path):
        os.makedirs(args.seg_path)

    # Load config file
    config_path = './configs/config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        config = EasyDict(config)
    
    with open(args.split_path, 'r') as f:
        split = f.readlines()

    for i in range(len(split)):
        split[i] = split[i].strip()

    print(f'Find {len(split)} samples.')
    print(f'Result will be stored in {args.seg_path}.')

    start = time.time()
    pool = Pool(processes=args.processes)
    for s in split:
        pool.apply_async(seg, (s, args.seg_path, args.dataset_path, config))

    pool.close()
    pool.join()

    print(f'Runtime: {time.time() - start:.4f}(s).')