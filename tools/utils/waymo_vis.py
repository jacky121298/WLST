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

def draw_3dbbox(bboxs: np.ndarray, vis: o3d.visualization.Visualizer, color: list, score: np.ndarray=np.array([]), scale: float=0.0, prefix: str=None, disp: bool=False):
    for i, bbox in enumerate(bboxs):
        x, y, z, l, w, h, heading = bbox
        if np.isclose(np.sum(bbox), 0.0):
            continue
        points = get_points(-l / 2, l / 2, -w / 2, w / 2, -h / 2, h / 2)
        print(f'[{bcolors.OKBLUE}Info{bcolors.ENDC}] Box: ({x:6.2f}, {y:6.2f}, {z:6.2f}, {l:5.2f}, {w:5.2f}, {h:5.2f}, {heading:5.2f})')

        points = rotz(heading) @ points.T + bbox[:3, np.newaxis]
        points = points.T
        
        lineset = get_lineset(points=points, color=color)
        vis.add_geometry(lineset)
        if len(score) > 0 and disp:
            vis.add_geometry(text_3d(f'{prefix}: {score[i]:.2f}', [x, y, z + scale * h], direction=(1, 0, 0), degrees=270))

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
    parser.add_argument('--index', type=int, help='Frame index.')
    parser.add_argument('--lidar', help='Path to lidar.')
    parser.add_argument('--M3D', help='Path to M3D.')
    parser.add_argument('--M2D3D', help='Path to M2D3D.')
    parser.add_argument('--pos_thres', type=float, default=0.0, help='Positive IOU threshold.')
    parser.add_argument('--vis_pcd', action='store_true', default=False, help='visualize point clouds.')
    args = parser.parse_args()

    point_features = np.load(args.lidar) # (N, 7): [x, y, z, intensity, elongation, NLZ_flag]
    points, NLZ_flag = point_features[:, 0:5], point_features[:, 5]
    points = points[NLZ_flag == -1]
    points[:, 3] = np.tanh(points[:, 3])
    points = points[:, :3]
    print(f'[{bcolors.OKBLUE}Info{bcolors.ENDC}] Point cloud shape: {points.shape}')
    
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='Frame:' + str(args.index), left=0, top=40)

    # render_option = vis.get_render_option()
    # render_option.background_color = np.array([0, 0, 0], np.float32)
    # render_option.point_color_option = o3d.visualization.PointColorOption.Default
    
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
    vis.add_geometry(coordinate_frame)
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.ascontiguousarray(points.astype(np.float64)))
    if args.vis_pcd:
        vis.add_geometry(pcd)

    ps_M3D = pickle.load(open(args.M3D, 'rb'))
    ps_M2D3D = pickle.load(open(args.M2D3D, 'rb'))
    
    frame_id = list(ps_M3D.keys())[args.index]
    frame_M3D = ps_M3D[frame_id]
    frame_M2D3D = ps_M2D3D[frame_id]

    gt_boxes = frame_M3D['gt_boxes']
    pred_box = frame_M3D['pred_box']
    iou_pred = frame_M3D['iou_pred']

    # Draw groundtruth labels (red)
    draw_3dbbox(gt_boxes, vis, color=[255, 0, 0])
    # Draw pseudo labels M3D (blue)
    draw_3dbbox(pred_box, vis, color=[0, 0, 255], score=iou_pred, scale=-0.5, prefix='M3D', disp=True)
    # Draw pseudo labels M2D3D (green)
    for sample in frame_M2D3D.values():
        pred_box = sample['pred_box']
        iou_pred = sample['iou_pred']
        draw_3dbbox(pred_box[np.newaxis, ...], vis, color=[0, 255, 0], score=[iou_pred], scale=0.5, prefix='M2D3D', disp=True)
    
    # Run Visualizer
    view_control = vis.get_view_control()
    params = view_control.convert_to_pinhole_camera_parameters()
    params.extrinsic = build_se3_transform([0, 0, 10, np.radians(90), np.radians(-90), 0])
    view_control.convert_from_pinhole_camera_parameters(params)
    vis.run()