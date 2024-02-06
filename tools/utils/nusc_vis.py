import os
import pickle
import argparse
import numpy as np
import open3d as o3d
import numpy.matlib as matlib
from math import sin, cos

class bcolors:
    FAIL = '\033[91m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    ENDC = '\033[0m'

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
    pcd.colors = o3d.utility.Vector3dVector(np.asarray([[1, 0, 0]] * img.shape[0]))
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

def draw_3dbbox(bboxs: np.ndarray, vis: o3d.visualization.Visualizer, color: list, score: np.ndarray=np.array([]), scale: float=1.0, disp: bool=False, pref: str='Box'):
    for i, bbox in enumerate(bboxs):
        x, y, z, l, w, h, heading = bbox
        if np.isclose(np.sum(bbox), 0.0):
            continue
        points = get_points(-l / 2, l / 2, -w / 2, w / 2, -h / 2, h / 2)
        print(f'[{bcolors.OKBLUE}Info{bcolors.ENDC}] {pref}: ({x:6.2f}, {y:6.2f}, {z:6.2f}, {l:5.2f}, {w:5.2f}, {h:5.2f}, {heading:5.2f})')

        points = rotz(heading) @ points.T + bbox[:3, np.newaxis]
        points = points.T
        
        lineset = get_lineset(points=points, color=color)
        vis.add_geometry(lineset)
        if len(score) > 0 and disp:
            vis.add_geometry(text_3d(f'{score[i]:.2f}', [x, y, z + scale * h], direction=(1, 0, 0), degrees=270))

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lidar', help='Path to lidar.')
    parser.add_argument('--ps_label_wlst', default=None, help='Path to WLST pseudo labels.')
    parser.add_argument('--ps_label_st3d', default=None, help='Path to ST3D pseudo labels.')
    parser.add_argument('--infos', help='Path to infos.')
    parser.add_argument('--pos_thres', type=float, default=0.0, help='Positive IOU threshold.')
    parser.add_argument('--disp', action='store_true', default=False, help='display scores.')
    args = parser.parse_args()

    points = np.fromfile(args.lidar, dtype=np.float32, count=-1).reshape([-1, 5])[:, :3]
    print(f'[{bcolors.OKBLUE}Info{bcolors.ENDC}] Point cloud shape: {points.shape}')

    frame_id = args.lidar.split('/')[-1].rstrip('.bin')
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=frame_id, left=0, top=40)

    render_option = vis.get_render_option()
    render_option.point_color_option = o3d.visualization.PointColorOption.Default
    
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
    vis.add_geometry(coordinate_frame)
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.ascontiguousarray(points.astype(np.float64)))
    pcd.colors = o3d.utility.Vector3dVector(np.array([[0, 0, 0] for _ in range(points.shape[0])]))
    vis.add_geometry(pcd)

    # Draw groundtruth labels (red)
    with open(args.infos, 'rb') as f:
        infos = pickle.load(f)
    for info in infos:
        if args.lidar.split('/')[-1] in info['lidar_path']:
            gt_boxes = info['gt_boxes'][:, :7]
            gt_names = info['gt_names']
            break
    gt_boxes = gt_boxes[gt_names == 'car']
    draw_3dbbox(gt_boxes, vis, color=[255, 0, 0], pref='GroundTruth')

    # Draw WLST pseudo labels (green)
    ps_wlst = pickle.load(open(args.ps_label_wlst, 'rb'))
    frame_M3D = ps_wlst[frame_id]
    pred_box = frame_M3D['gt_boxes']
    pred_box[:, :3] -= np.array([0.0, 0.0, 1.8])
    iou_pred = frame_M3D['iou_pred']
    draw_3dbbox(pred_box, vis, color=[0, 255, 0], score=iou_pred, scale=0.0, disp=args.disp, pref='WLST')

    # Draw ST3D pseudo labels (blue)
    ps_st3d = pickle.load(open(args.ps_label_st3d, 'rb'))
    if ps_st3d.get(frame_id):
        pseudo = ps_st3d[frame_id]['gt_boxes'].reshape(-1, 9)
        scores = ps_st3d[frame_id]['iou_scores']

        pseudo = pseudo[pseudo[:, 7] > 0]
        pseudo = pseudo[:, :7]
        pseudo[:, :3] -= np.array([0.0, 0.0, 1.6])

        draw_3dbbox(pseudo, vis, color=[0, 0, 255], score=scores, scale=1.0, disp=args.disp, pref='ST3D')
    
    # Run Visualizer
    view_control = vis.get_view_control()
    params = view_control.convert_to_pinhole_camera_parameters()
    params.extrinsic = build_se3_transform([0, 0, 30, np.pi / 1.5, 0, 0])
    # params.extrinsic = build_se3_transform([0, 10, 20, np.pi / 5, 0, 0])
    view_control.convert_from_pinhole_camera_parameters(params)
    vis.run()