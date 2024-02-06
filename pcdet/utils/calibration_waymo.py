import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import tensorflow as tf
from waymo_open_dataset import dataset_pb2
from waymo_open_dataset.camera.ops import py_camera_model_ops

try:
    tf.compat.v1.enable_eager_execution()
except:
    pass

class Calibration(object):
    def __init__(self, calib):
        self.CAMERA_EX = {
            'CAMERA_FRONT': calib['CAMERA_FRONT_EX'],
            'CAMERA_FRONT_LEFT': calib['CAMERA_FRONT_LEFT_EX'],
            'CAMERA_FRONT_RIGHT': calib['CAMERA_FRONT_RIGHT_EX'],
            'CAMERA_SIDE_LEFT': calib['CAMERA_SIDE_LEFT_EX'],
            'CAMERA_SIDE_RIGHT': calib['CAMERA_SIDE_RIGHT_EX'],
        }
        self.CAMERA_IN = {
            'CAMERA_FRONT': calib['CAMERA_FRONT_IN'],
            'CAMERA_FRONT_LEFT': calib['CAMERA_FRONT_LEFT_IN'],
            'CAMERA_FRONT_RIGHT': calib['CAMERA_FRONT_RIGHT_IN'],
            'CAMERA_SIDE_LEFT': calib['CAMERA_SIDE_LEFT_IN'],
            'CAMERA_SIDE_RIGHT': calib['CAMERA_SIDE_RIGHT_IN'],
        }
        self.IMG_SHAPE = {
            'CAMERA_FRONT': 'image_shape_0',
            'CAMERA_FRONT_LEFT': 'image_shape_1',
            'CAMERA_FRONT_RIGHT': 'image_shape_2',
            'CAMERA_SIDE_LEFT': 'image_shape_3',
            'CAMERA_SIDE_RIGHT': 'image_shape_4',
        }

    def cart_to_hom(self, pts):
        pts_hom = np.hstack((pts, np.ones((pts.shape[0], 1), dtype=np.float32)))
        return pts_hom

    def veh_to_img(self, pts, where, pose, image):
        pts_hom = self.cart_to_hom(pts) # (N, 4)
        pts_world = pose @ pts_hom.T    # (4, N)
        pts_world = pts_world[:3].T     # (N, 3)

        extrinsic = tf.reshape(tf.constant(list(self.CAMERA_EX[where].flatten()), dtype=tf.float64), [4, 4])
        intrinsic = tf.constant(list(self.CAMERA_IN[where].flatten()), dtype=tf.float64)

        height, width = image[self.IMG_SHAPE[where]]
        metadata = tf.constant([width, height, dataset_pb2.CameraCalibration.GLOBAL_SHUTTER], dtype=tf.int32)
        camera_image_metadata = list(pose.flatten()) + [0.0] * 10

        pts_img = py_camera_model_ops.world_to_image(extrinsic, intrinsic, metadata, camera_image_metadata, pts_world).numpy()
        return pts_img[:, :2], pts_img[:, 2]

    def veh_to_rect(self, pts):
        pts_hom = self.cart_to_hom(pts)                                      # (N, 4)
        pts_rect = np.linalg.inv(self.CAMERA_EX['CAMERA_FRONT']) @ pts_hom.T # (4, N)
        return pts_rect[:3].T

    def rect_to_veh(self, pts):
        pts_hom = self.cart_to_hom(pts)                      # (N, 4)
        pts_veh = self.CAMERA_EX['CAMERA_FRONT'] @ pts_hom.T # (4, N)
        return pts_veh[:3].T

    def img_to_veh(self, u, v, depth, where, pose, image):
        pts_img = np.hstack([u, v, depth]) # (N, 3)

        extrinsic = tf.reshape(tf.constant(list(self.CAMERA_EX[where].flatten()), dtype=tf.float64), [4, 4])
        intrinsic = tf.constant(list(self.CAMERA_IN[where].flatten()), dtype=tf.float64)

        height, width = image[self.IMG_SHAPE[where]]
        metadata = tf.constant([width, height, dataset_pb2.CameraCalibration.GLOBAL_SHUTTER], dtype=tf.int32)
        camera_image_metadata = list(pose.flatten()) + [0.0] * 10

        pts_world = py_camera_model_ops.image_to_world(extrinsic, intrinsic, metadata, camera_image_metadata, pts_img).numpy() # (N, 3)
        pts_world = self.cart_to_hom(pts_world)     # (N, 4)
        pts_veh = np.linalg.inv(pose) @ pts_world.T # (4, N)
        return pts_veh[:3].T

    def img_to_rect(self, u, v, depth, where, pose, image):
        pts_veh = self.img_to_veh(u, v, depth, where, pose, image)            # (N, 3)
        pts_veh = self.cart_to_hom(pts_veh)                                   # (N, 4)
        pts_rect = np.linalg.inv(self.CAMERA_EX['CAMERA_FRONT']) @ pts_veh.T  # (4, N)
        return pts_rect[:3].T