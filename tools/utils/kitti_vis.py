import os
import pickle
import argparse
import numpy as np
import open3d as o3d
import skimage.io as io
import numpy.matlib as matlib
from math import sin, cos
from pcdet.utils import calibration_kitti

class bcolors:
    FAIL = '\033[91m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    ENDC = '\033[0m'

def get_fov_flag(pts_rect, img_shape, calib, margin=0):
    pts_img, pts_rect_depth = calib.rect_to_img(pts_rect)
    val_flag_1 = np.logical_and(pts_img[:, 0] >= 0 - margin, pts_img[:, 0] < img_shape[1] + margin)
    val_flag_2 = np.logical_and(pts_img[:, 1] >= 0 - margin, pts_img[:, 1] < img_shape[0] + margin)
    val_flag_merge = np.logical_and(val_flag_1, val_flag_2)
    pts_valid_flag = np.logical_and(val_flag_merge, pts_rect_depth >= 0)
    return pts_valid_flag

def text_3d(text, pos, direction=None, degrees=0.0, font='DejaVu Sans Mono for Powerline.ttf', font_size=20):
    """
        Generate a 3D text point cloud used for visualization.
        :param text: content of the text
        :param pos: 3D xyz position of the text upper left corner
        :param direction: 3D normalized direction of where the text faces
        :param degrees: in plane rotation of text
        :param font: Name of the font - change it according to your system
        :param font_size: size of the font
        :return: o3d.geoemtry.PointCloud object
    """
    if direction is None:
        direction = (0., 0., 1.)

    from PIL import Image, ImageFont, ImageDraw
    from pyquaternion import Quaternion

    font_obj = ImageFont.truetype(font, font_size)
    font_dim = font_obj.getsize(text)

    img = Image.new('RGB', font_dim, color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((0, 0), text, font=font_obj, fill=(0, 0, 0))
    img = np.asarray(img)
    img_mask = img[:, :, 0] < 128
    indices = np.indices([*img.shape[0:2], 1])[:, img_mask, 0].reshape(3, -1).T

    pcd = o3d.geometry.PointCloud()
    # pcd.colors = o3d.utility.Vector3dVector(img[img_mask, :].astype(float) / 255.0)
    pcd.colors = o3d.utility.Vector3dVector(np.array([[0, 0, 0] for _ in range(img.shape[0])]))
    pcd.points = o3d.utility.Vector3dVector(indices / 100.0)

    raxis = np.cross([0.0, 0.0, 1.0], direction)
    if np.linalg.norm(raxis) < 1e-6:
        raxis = (0.0, 0.0, 1.0)
    
    trans = (
        Quaternion(axis=raxis, radians=np.arccos(direction[2])) *
        Quaternion(axis=direction, degrees=degrees)
    ).transformation_matrix
    
    trans[0:3, 3] = np.asarray(pos)
    pcd.transform(trans)
    return pcd

def get_lineset(points: np.ndarray, color: list):
    lines = [
        [0, 1], [0, 2], [1, 3], [2, 3],
        [4, 5], [4, 6], [5, 7], [6, 7],
        [0, 4], [1, 5], [2, 6], [3, 7],
    ]
    colors = [color for i in range(len(lines))]
    
    lineset = o3d.geometry.LineSet()
    lineset.points = o3d.utility.Vector3dVector(points)
    lineset.lines = o3d.utility.Vector2iVector(lines)
    lineset.colors = o3d.utility.Vector3dVector(colors)
    
    return lineset

def get_points(xmin: np.float, xmax: np.float, ymin: np.float, ymax: np.float, zmin: np.float, zmax: np.float):
    points = [
        [xmin, ymin, zmin], [xmin, ymin, zmax], [xmin, ymax, zmin], [xmin, ymax, zmax],
        [xmax, ymin, zmin], [xmax, ymin, zmax], [xmax, ymax, zmin], [xmax, ymax, zmax]
    ]
    return np.asarray(points)

def rotz(angle: np.float):
    c = np.cos(angle)
    s = np.sin(angle)
    rotz = np.array([
        [c, -s, 0],
        [s,  c, 0],
        [0,  0, 1],
    ])
    return rotz

def roty(angle: np.float):
    c = np.cos(angle)
    s = np.sin(angle)
    roty = np.array([
        [ c, 0, s],
        [ 0, 1, 0],
        [-s, 0, c],
    ])
    return roty

def draw_3dbbox(bboxs: np.ndarray, vis: o3d.visualization.Visualizer, color: list, score: np.ndarray=np.array([]), scale: float=1.0, disp: bool=False, pref: str='Box'):
    for i, bbox in enumerate(bboxs):
        x, y, z, l, w, h, heading = bbox
        points = get_points(-w / 2, w / 2, -h, 0, -l / 2, l / 2)
        print(f'[{bcolors.OKBLUE}Info{bcolors.ENDC}] {pref}: ({x:6.2f}, {y:6.2f}, {z:6.2f}, {l:5.2f}, {w:5.2f}, {h:5.2f}, {heading:5.2f})')

        heading = heading + np.pi / 2
        points = roty(heading) @ points.T + bbox[:3, np.newaxis]
        points = points.T
        
        lineset = get_lineset(points=points, color=color)
        vis.add_geometry(lineset)
        if len(score) > 0 and disp:
            vis.add_geometry(text_3d(f'{score[i]:.2f}', [x, y - scale * h, z], direction=(0, 1, 0), degrees=270))

def euler_to_so3(rpy: list):
    R_x = np.matrix([
        [1,           0,            0],
        [0, cos(rpy[0]), -sin(rpy[0])],
        [0, sin(rpy[0]),  cos(rpy[0])],
    ])
    R_y = np.matrix([
        [ cos(rpy[1]), 0, sin(rpy[1])],
        [           0, 1,           0],
        [-sin(rpy[1]), 0, cos(rpy[1])],
    ])
    R_z = np.matrix([
        [cos(rpy[2]), -sin(rpy[2]), 0],
        [sin(rpy[2]),  cos(rpy[2]), 0],
        [          0,            0, 1],
    ])
    return R_z * R_y * R_x

def build_se3_transform(xyzrpy: list):
    se3 = matlib.identity(4)
    se3[0:3, 0:3] = euler_to_so3(xyzrpy[3:6])
    se3[0:3, 3] = np.matrix(xyzrpy[0:3]).transpose()
    return se3

def boxes3d_lidar_to_kitti_camera(boxes3d_lidar, calib):
    """
    :param boxes3d_lidar: (N, 7) [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center
    :param calib:
    :return:
        boxes3d_camera: (N, 7) [x, y, z, l, h, w, r] in rect camera coords
    """
    xyz_lidar = boxes3d_lidar[:, 0:3]
    l, w, h, r = boxes3d_lidar[:, 3:4], boxes3d_lidar[:, 4:5], boxes3d_lidar[:, 5:6], boxes3d_lidar[:, 6:7]

    xyz_lidar[:, 2] -= h.reshape(-1) / 2
    xyz_cam = calib.lidar_to_rect(xyz_lidar)
    # xyz_cam[:, 1] += h.reshape(-1) / 2
    r = -r - np.pi / 2
    return np.concatenate([xyz_cam, l, h, w, r], axis=-1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seq', help='Sequence name.')
    parser.add_argument('--lidar', help='Path to lidar.')
    parser.add_argument('--gt_label', help='Path to groundtruth labels.')
    parser.add_argument('--ps_label_wlst', default=None, help='Path to WLST pseudo labels.')
    parser.add_argument('--ps_label_st3d', default=None, help='Path to ST3D pseudo labels.')
    parser.add_argument('--disp', action='store_true', default=False, help='display scores.')
    args = parser.parse_args()

    lidar_path = os.path.join(args.lidar, args.seq + '.bin')
    points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 4)
    points = points[..., :3]

    calib_path = os.path.join(args.lidar.replace('velodyne', 'calib'), args.seq + '.txt')
    calib = calibration_kitti.Calibration(calib_path)
    points = calib.lidar_to_rect(points)
    print(f'[{bcolors.OKBLUE}Info{bcolors.ENDC}] Point cloud shape: {points.shape}')
    
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=args.seq, left=0, top=40)

    render_option = vis.get_render_option()
    render_option.point_color_option = o3d.visualization.PointColorOption.Default
    
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
    vis.add_geometry(coordinate_frame)
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.ascontiguousarray(points.astype(np.float64)))
    pcd.colors = o3d.utility.Vector3dVector(np.array([[0, 0, 0] for _ in range(points.shape[0])]))
    vis.add_geometry(pcd)

    # Draw groundtruth labels (red)
    annos_path = os.path.join(args.gt_label, args.seq + '.txt')
    with open(annos_path, 'r') as f:
        gt = f.readlines()
    
    labels = []
    used_class = ['Car']
    for i in range(len(gt)):
        g = gt[i].split()
        if g[0] not in used_class:
            continue
        # (x, y, z, l, w, h, rot)
        labels.append(np.array([g[11], g[12], g[13], g[10], g[9], g[8], g[14]]).astype(np.float))

    labels = np.array(labels)
    draw_3dbbox(labels, vis, color=[255, 0, 0], pref='GroundTruth')

    # Draw WLST pseudo labels (green)
    ps = pickle.load(open(args.ps_label_wlst, 'rb'))
    if ps.get(args.seq):
        pseudo = ps[args.seq]['gt_boxes'].reshape(-1, 7)
        scores = ps[args.seq]['iou_pred']

        pseudo_lidar_center = pseudo[:, :3]
        pts_rect = calib.lidar_to_rect(pseudo_lidar_center)
        img = io.imread(os.path.join(args.lidar.replace('velodyne', 'image_2'), args.seq + '.png'))
        fov_flag = get_fov_flag(pts_rect, img.shape[:2], calib, margin=5)

        pseudo = pseudo[fov_flag]
        scores = scores[fov_flag]
        
        pseudo[:, :3] -= np.array([0.0, 0.0, 1.6])
        pseudo = boxes3d_lidar_to_kitti_camera(pseudo, calib)
        draw_3dbbox(pseudo, vis, color=[0, 255, 0], score=scores, scale=0.0, disp=args.disp, pref='WLST')

    # Draw ST3D pseudo labels (blue)
    ps = pickle.load(open(args.ps_label_st3d, 'rb'))
    if ps.get(args.seq):
        pseudo = ps[args.seq]['gt_boxes'].reshape(-1, 9)
        scores = ps[args.seq]['iou_scores']

        pseudo_lidar_center = pseudo[:, :3]
        pts_rect = calib.lidar_to_rect(pseudo_lidar_center)
        img = io.imread(os.path.join(args.lidar.replace('velodyne', 'image_2'), args.seq + '.png'))
        fov_flag = get_fov_flag(pts_rect, img.shape[:2], calib, margin=5)

        pseudo = pseudo[fov_flag]
        scores = scores[fov_flag]
        
        pseudo = pseudo[pseudo[:, 7] > 0]
        pseudo = pseudo[:, :7]
        pseudo[:, :3] -= np.array([0.0, 0.0, 1.6])

        pseudo = boxes3d_lidar_to_kitti_camera(pseudo, calib)
        draw_3dbbox(pseudo, vis, color=[0, 0, 255], score=scores, scale=1.0, disp=args.disp, pref='ST3D')
    
    # Run Visualizer
    view_control = vis.get_view_control()
    params = view_control.convert_to_pinhole_camera_parameters()
    params.extrinsic = build_se3_transform([0, 10, 20, np.pi / 5, 0, 0])
    view_control.convert_from_pinhole_camera_parameters(params)
    vis.run()