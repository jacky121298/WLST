import os
import pickle
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path
from pcdet.utils import box_utils
from pcdet.utils import calibration_kitti

if __name__ == '__main__':
    # python ./utils/ps2kitti.py --ps ../output/da-waymo-kitti_models/pvrcnn_st3d/wl_labeler_ros_st/consistency/ps_label/ps_label_e40_M3D.pkl
    parser = argparse.ArgumentParser()
    parser.add_argument('--ps', help='Path to pseudo labels.')
    args = parser.parse_args()

    ps = pickle.load(open(args.ps, 'rb'))
    with open('../data/kitti/ImageSets/train.txt', 'r') as f:
        split = f.readlines()

    output_dir = Path('../output/da-waymo-kitti_models/pvrcnn_st3d/wl_labeler_ros_st/consistency/ps_label/kitti')
    output_dir.mkdir(parents=True, exist_ok=True)

    final_output_dir = output_dir / args.ps.split('_')[-2]
    if final_output_dir.exists():
        print(f'Removing (existed) label path {final_output_dir}.')
        os.system(f'rm -rf {final_output_dir}')
    final_output_dir.mkdir(parents=True, exist_ok=True)

    for s in tqdm(split):
        label = ''
        s = s.strip()
        label_path = final_output_dir / f'{s}.txt'

        if s not in ps.keys():
            with open(label_path, 'a+') as f:
                f.write(label)
            continue

        ps_boxes = ps[s]['gt_boxes'] # LCS: (x, y, z, l, w, h, r), shape: (N, 7)
        if len(ps_boxes) == 0:
            with open(label_path, 'a+') as f:
                f.write(label)
            continue

        ps_boxes[:, :3] -= np.array([0.0, 0.0, 1.6], dtype=np.float32).reshape(1, 3)
        calib = calibration_kitti.Calibration('/tmp2/jacky1212/kitti/training/calib/' + s + '.txt')
        ps_boxes_camera = box_utils.boxes3d_lidar_to_kitti_camera(ps_boxes, calib) # CCS: (x, y, z, l, h, w, r), shape: (N, 7)

        for box in ps_boxes_camera:
            label += 'Car 0.00 0 0.00 0.00 0.00 0.00 0.00 '
            label += '%.2f %.2f %.2f ' % (box[4], box[5], box[3]) # (h, w, l)
            label += '%.2f %.2f %.2f ' % (box[0], box[1], box[2]) # (x, y, z)
            label += '%.2f\n' % (box[6]) # r

        with open(label_path, 'a+') as f:
            f.write(label)