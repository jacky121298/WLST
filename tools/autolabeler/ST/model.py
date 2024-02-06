import os
import copy
import torch
import pickle
import numpy as np
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from tqdm import tqdm
from skimage import io
from collections import defaultdict
from torch.autograd import Function
from torch.utils.data import Dataset
from skimage.transform import resize
from pcdet.utils import self_training_utils_wl
from train_utils.train_utils_wl import compute_box3d_iou
from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils
from pcdet.datasets.augmentor.data_augmentor import DataAugmentorWL

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

def angle2class(angle, num_class):
    angle = angle % (2 * np.pi)
    assert (angle >= 0 and angle <= 2 * np.pi)
    angle_per_class = 2 * np.pi / float(num_class)
    shifted_angle = (angle + angle_per_class / 2) % (2 * np.pi)
    class_id = int(shifted_angle / angle_per_class)
    residual_angle = shifted_angle - (class_id * angle_per_class + angle_per_class / 2)
    return class_id, residual_angle

def rotz(angle: torch.float):
    c = torch.cos(angle)
    s = torch.sin(angle)
    rotz = torch.tensor([
        [c, -s, 0],
        [s,  c, 0],
        [0,  0, 1],
    ])
    return rotz

def gather_object_pts(pts, mask, n_pts):
    '''
        :param pts: (bs, 3, n)
        :param mask: (bs, n)
        :param n_pts: max number of points of an object
        :return:
            object_pts: (bs, 3, n_pts)
            indices: (bs, n_pts)
    '''
    bs = pts.shape[0]
    object_pts = torch.zeros((bs, pts.shape[1], n_pts))
    indices = torch.zeros((bs, n_pts), dtype=torch.int64)

    for i in range(bs):
        pos_indices = torch.nonzero(mask[i, :] > 0.5).squeeze(1)
        if len(pos_indices) > 0:
            if len(pos_indices) >= n_pts:
                choice = np.random.choice(len(pos_indices), n_pts, replace=False)
            else:
                choice = np.random.choice(len(pos_indices), n_pts - len(pos_indices), replace=True)
                choice = np.concatenate((np.arange(len(pos_indices)), choice))
            
            np.random.shuffle(choice)
            indices[i, :] = pos_indices[choice]
            object_pts[i, :, :] = pts[i, :, indices[i, :]]

    return object_pts, indices

def point_cloud_masking(pts, logits, NUM_OBJECT_POINT):
    '''
        :param pts: (bs, 3, n) in frustum
        :param logits: (bs, n, 2)
    '''
    bs = pts.shape[0]
    n_pts = pts.shape[2]
    # Binary Classification for each point
    mask = logits[:, :, 0] < logits[:, :, 1] # (bs, n)
    pts_xyz = pts[:, :3, :] # (bs, 3, n)
    object_pts, _ = gather_object_pts(pts_xyz, mask, NUM_OBJECT_POINT)
    return object_pts.float(), mask

def parse_output_to_tensors(box_pred, NUM_HEADING_BIN, NUM_SIZE_CLUSTER, MEAN_SIZE_ARR):
    '''
        :param box_pred: (bs, 59)
        :return:
            center: (bs, 3)
            heading_scores: (bs, 12)
            heading_residuals_normalized: (bs, 12), -1 to 1
            heading_residuals: (bs, 12)
            size_scores: (bs, 8)
            size_residuals_normalized: (bs, 8)
            size_residuals: (bs, 8)
    '''
    bs = box_pred.shape[0]
    # Center
    c = 3
    center = box_pred[:, :c]

    # Heading
    heading_scores = box_pred[:, c:c + NUM_HEADING_BIN]
    c += NUM_HEADING_BIN
    heading_residuals_normalized = box_pred[:, c:c + NUM_HEADING_BIN]
    heading_residuals = heading_residuals_normalized * (np.pi / NUM_HEADING_BIN)
    c += NUM_HEADING_BIN

    # Size
    size_scores = box_pred[:, c:c + NUM_SIZE_CLUSTER]
    c += NUM_SIZE_CLUSTER
    size_residuals_normalized = box_pred[:, c:c + 3 * NUM_SIZE_CLUSTER].contiguous()
    size_residuals_normalized = size_residuals_normalized.view(bs, NUM_SIZE_CLUSTER, 3)
    size_residuals = size_residuals_normalized * torch.from_numpy(MEAN_SIZE_ARR).unsqueeze(0).repeat(bs, 1, 1).float().cuda()
    return center, heading_scores, heading_residuals_normalized, heading_residuals, size_scores, size_residuals_normalized, size_residuals

class GradientReverse(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

class WeakLabeler(nn.Module):
    def __init__(self, cfg, n_channel=3, seg_model=None, track_running_stats=True):
        super(WeakLabeler, self).__init__()
        self.cfg = cfg
        self.ins_seg = PointNetInstanceSeg(n_channel=n_channel, track_running_stats=track_running_stats)
        self.box_est = PointNetEstimation(cfg=cfg, n_channel=3)

        if seg_model != None:
            self.load_seg(seg_model)
            self.fix_seg()
    
    def load_seg(self, seg_model):
        if seg_model != None:
            self.ins_seg.load_state_dict(torch.load(seg_model)['model_state_dict'])
    
    def fix_seg(self):
        self.ins_seg.eval()
        for param in self.ins_seg.parameters():
            param.requires_grad = False

    def free_seg(self):
        self.ins_seg.train()
        for param in self.ins_seg.parameters():
            param.requires_grad = True
    
    def forward(self, img, pts, box):
        '''
            :param img: (bs, 128, 128, 3)
            :param pts: (bs, 3, n)
            :param box: (bs, 7)
        '''
        # 3D Instance Segmentation
        with torch.no_grad():
            logits = self.ins_seg(pts) # (bs, n, 2)
        object_pts_xyz, mask = point_cloud_masking(pts, logits, self.cfg.WEAK_LABEL.NUM_OBJECT_POINT)
        object_pts_xyz = object_pts_xyz.cuda() # (bs, 3, NUM_OBJECT_POINT)

        object_pts_xyz_mean = None
        if self.cfg.WEAK_LABEL.get('SEG_TO_CENTER', False):
            object_pts_xyz_mean = torch.mean(object_pts_xyz, dim=2, keepdim=True) # (bs, 3, 1)
            object_pts_xyz -= object_pts_xyz_mean
            
        # 3D Box Estimation
        box_pred, iou_pred = self.box_est(object_pts_xyz) # (bs, 59), (bs, 1)
        center, heading_scores, heading_residuals_normalized, heading_residuals, \
        size_scores, size_residuals_normalized, size_residuals = parse_output_to_tensors(
            box_pred,
            self.cfg.WEAK_LABEL.NUM_HEADING_BIN,
            self.cfg.WEAK_LABEL.NUM_SIZE_CLUSTER,
            np.array(self.cfg.WEAK_LABEL.MEAN_SIZE_ARR)
        )
        
        if self.cfg.WEAK_LABEL.get('SEG_TO_CENTER', False):
            center += object_pts_xyz_mean.squeeze(dim=2)
        
        output = {
            'logits': logits,
            'mask': mask,
            'center': center,
            'heading_scores': heading_scores,
            'heading_residuals_normalized': heading_residuals_normalized,
            'heading_residuals': heading_residuals,
            'size_scores': size_scores,
            'size_residuals_normalized': size_residuals_normalized,
            'size_residuals': size_residuals,
            'iou_pred': iou_pred,
        }
        return output

class WeakLabelerWithImage(nn.Module):
    def __init__(self, cfg, n_channel=3, seg_model=None, track_running_stats=True, dann=False):
        super(WeakLabelerWithImage, self).__init__()
        self.cfg = cfg
        self.dann = dann
        self.ins_seg = PointNetInstanceSeg(n_channel=n_channel, track_running_stats=track_running_stats)
        self.point_enc = PointFeatureEncoder(cfg=cfg, n_channel=3)
        self.image_enc = ImageFeatureEncoder()
        self.box_est = BoxEstimation(cfg=cfg)
        if self.dann:
            self.domain_classifier = nn.Sequential(
                nn.Linear(1024, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Linear(128, 2),
            )

        if seg_model != None:
            self.load_seg(seg_model)
            self.fix_seg()
    
    def load_seg(self, seg_model):
        if seg_model != None:
            self.ins_seg.load_state_dict(torch.load(seg_model)['model_state_dict'])
    
    def fix_seg(self):
        self.ins_seg.eval()
        for param in self.ins_seg.parameters():
            param.requires_grad = False

    def free_seg(self):
        self.ins_seg.train()
        for param in self.ins_seg.parameters():
            param.requires_grad = True
    
    def forward(self, img, pts, box, alpha=0.0):
        '''
            :param img: (bs, 128, 128, 3)
            :param pts: (bs, 3, n)
            :param box: (bs, 7)
        '''
        # 3D Instance Segmentation
        with torch.no_grad():
            logits = self.ins_seg(pts) # (bs, n, 2)
        object_pts_xyz, mask = point_cloud_masking(pts, logits, self.cfg.WEAK_LABEL.NUM_OBJECT_POINT)
        object_pts_xyz = object_pts_xyz.cuda() # (bs, 3, NUM_OBJECT_POINT)

        object_pts_xyz_mean = None
        if self.cfg.WEAK_LABEL.get('SEG_TO_CENTER', False):
            object_pts_xyz_mean = torch.mean(object_pts_xyz, dim=2, keepdim=True) # (bs, 3, 1)
            object_pts_xyz -= object_pts_xyz_mean
            
        # 3D Box Estimation
        point_feat = self.point_enc(object_pts_xyz)              # (bs, 512)
        image_feat = self.image_enc(img)                         # (bs, 512)
        global_feat = torch.cat((point_feat, image_feat), dim=1) # (bs, 1024)
        if self.dann:
            reverse_feat = GradientReverse.apply(global_feat, alpha)
            domain_output = self.domain_classifier(reverse_feat)
        
        box_pred, iou_pred = self.box_est(global_feat) # (bs, 59), (bs, 1)
        center, heading_scores, heading_residuals_normalized, heading_residuals, \
        size_scores, size_residuals_normalized, size_residuals = parse_output_to_tensors(
            box_pred,
            self.cfg.WEAK_LABEL.NUM_HEADING_BIN,
            self.cfg.WEAK_LABEL.NUM_SIZE_CLUSTER,
            np.array(self.cfg.WEAK_LABEL.MEAN_SIZE_ARR)
        )
        
        if self.cfg.WEAK_LABEL.get('SEG_TO_CENTER', False):
            center += object_pts_xyz_mean.squeeze(dim=2)
        
        output = {
            'logits': logits,
            'mask': mask,
            'center': center,
            'heading_scores': heading_scores,
            'heading_residuals_normalized': heading_residuals_normalized,
            'heading_residuals': heading_residuals,
            'size_scores': size_scores,
            'size_residuals_normalized': size_residuals_normalized,
            'size_residuals': size_residuals,
            'iou_pred': iou_pred,
            'domain_output': domain_output if self.dann else None,
        }
        return output

class WeakLabelerTwoStage(nn.Module):
    def __init__(self, cfg, n_channel=3, seg_model=None, track_running_stats=True):
        super(WeakLabelerTwoStage, self).__init__()
        self.cfg = cfg
        self.ins_seg = PointNetInstanceSeg(n_channel=n_channel, track_running_stats=track_running_stats)
        self.box_est_one = PointNetEstimation(cfg=cfg, n_channel=3)
        self.box_est_two = PointNetEstimation(cfg=cfg, n_channel=3)

        if seg_model != None:
            self.load_seg(seg_model)
            self.fix_seg()
    
    def load_seg(self, seg_model):
        if seg_model != None:
            self.ins_seg.load_state_dict(torch.load(seg_model)['model_state_dict'])
    
    def fix_seg(self):
        self.ins_seg.eval()
        for param in self.ins_seg.parameters():
            param.requires_grad = False

    def free_seg(self):
        self.ins_seg.train()
        for param in self.ins_seg.parameters():
            param.requires_grad = True
    
    def forward(self, img, pts, box):
        '''
            :param img: (bs, 128, 128, 3)
            :param pts: (bs, 3, n)
            :param box: (bs, 7)
        '''
        # 3D Instance Segmentation
        with torch.no_grad():
            logits = self.ins_seg(pts) # (bs, n, 2)
        object_pts_xyz, mask = point_cloud_masking(pts, logits, self.cfg.WEAK_LABEL.NUM_OBJECT_POINT)
        object_pts_xyz = object_pts_xyz.cuda() # (bs, 3, NUM_OBJECT_POINT)
        object_pts_xyz_two = torch.clone(object_pts_xyz) # (bs, 3, NUM_OBJECT_POINT)

        object_pts_xyz_mean = None
        if self.cfg.WEAK_LABEL.get('SEG_TO_CENTER', False):
            object_pts_xyz_mean = torch.mean(object_pts_xyz, dim=2, keepdim=True) # (bs, 3, 1)
            object_pts_xyz -= object_pts_xyz_mean
        
        # 3D Box Estimation One
        box_pred_one, iou_pred_one = self.box_est_one(object_pts_xyz) # (bs, 59), (bs, 1)
        center_one, heading_scores_one, heading_residuals_normalized_one, heading_residuals_one, \
        size_scores_one, size_residuals_normalized_one, size_residuals_one = parse_output_to_tensors(
            box_pred_one,
            self.cfg.WEAK_LABEL.NUM_HEADING_BIN,
            self.cfg.WEAK_LABEL.NUM_SIZE_CLUSTER,
            np.array(self.cfg.WEAK_LABEL.MEAN_SIZE_ARR)
        )
        if self.cfg.WEAK_LABEL.get('SEG_TO_CENTER', False):
            center_one += object_pts_xyz_mean.squeeze(dim=2)

        # Coordinate transform & Generate labels
        bs = center_one.shape[0]
        heading_class = np.argmax(heading_scores_one.cpu().detach().numpy(), 1) # (bs,)
        heading_residual = np.array([heading_residuals_one.cpu().detach().numpy()[i, heading_class[i]] for i in range(bs)]) # (bs,)
        size_class = np.argmax(size_scores_one.cpu().detach().numpy(), 1) # (bs,)
        size_residual = np.vstack([size_residuals_one.cpu().detach().numpy()[i, size_class[i], :] for i in range(bs)]) # (bs, 3)
        
        box_size = np.zeros((bs, 3))
        heading_angle = np.zeros((bs, 1))
        for i in range(bs):
            box_size[i, :] = class2size(self.cfg, size_class[i], size_residual[i])
            heading_angle[i] = class2angle(self.cfg, heading_class[i], heading_residual[i])
        box_one = np.concatenate((center_one.cpu().detach().numpy(), box_size, heading_angle), axis=1) # (bs, 7)
        box_one = torch.from_numpy(box_one).float().cuda()
        
        heading_class_label_two = np.zeros((bs,))
        heading_residuals_label_two = np.zeros((bs,))
        object_pts_xyz_two -= box_one[:, :3, np.newaxis]

        for i in range(bs):
            object_pts_xyz_two[i] = rotz(-box_one[i, -1]).float().cuda() @ object_pts_xyz_two[i]
            heading_class_label_two[i], heading_residuals_label_two[i] = angle2class(box[i, -1] - box_one[i, -1], self.cfg.WEAK_LABEL.NUM_HEADING_BIN)

        heading_class_label_two = torch.from_numpy(heading_class_label_two).long().cuda()
        heading_residuals_label_two = torch.from_numpy(heading_residuals_label_two).float().cuda()

        # 3D Box Estimation Two
        box_pred_two, iou_pred_two = self.box_est_two(object_pts_xyz_two) # (bs, 59), (bs, 1)
        center_two, heading_scores_two, heading_residuals_normalized_two, heading_residuals_two, \
        size_scores_two, size_residuals_normalized_two, size_residuals_two = parse_output_to_tensors(
            box_pred_two,
            self.cfg.WEAK_LABEL.NUM_HEADING_BIN,
            self.cfg.WEAK_LABEL.NUM_SIZE_CLUSTER,
            np.array(self.cfg.WEAK_LABEL.MEAN_SIZE_ARR)
        )
        center_two += center_one
        
        # Postprocessing
        heading_class = np.argmax(heading_scores_two.cpu().detach().numpy(), 1) # (bs,)
        heading_residual = np.array([heading_residuals_two.cpu().detach().numpy()[i, heading_class[i]] for i in range(bs)]) # (bs,)
        heading_scores = np.zeros((bs, self.cfg.WEAK_LABEL.NUM_HEADING_BIN))
        heading_residuals = np.zeros((bs, self.cfg.WEAK_LABEL.NUM_HEADING_BIN))

        for i in range(bs):
            heading_angle = class2angle(self.cfg, heading_class[i], heading_residual[i])
            heading_angle += box_one[i, -1]
            heading_class_label, heading_residuals_label = angle2class(heading_angle, self.cfg.WEAK_LABEL.NUM_HEADING_BIN)
            
            heading_scores[i, heading_class_label] = 1
            heading_residuals[i, heading_class_label] = heading_residuals_label

        heading_scores = torch.from_numpy(heading_scores).float().cuda()
        heading_residuals = torch.from_numpy(heading_residuals).float().cuda()
        
        output = {
            'logits': logits,
            'mask': mask,
            'center_one': center_one,
            'heading_scores_one': heading_scores_one,
            'heading_residuals_normalized_one': heading_residuals_normalized_one,
            'heading_residuals_one': heading_residuals_one,
            'size_scores_one': size_scores_one,
            'size_residuals_normalized_one': size_residuals_normalized_one,
            'size_residuals_one': size_residuals_one,
            'iou_pred_one': iou_pred_one,
            'center_two': center_two,
            'heading_scores_two': heading_scores_two,
            'heading_residuals_normalized_two': heading_residuals_normalized_two,
            'heading_residuals_two': heading_residuals_two,
            'size_scores_two': size_scores_two,
            'size_residuals_normalized_two': size_residuals_normalized_two,
            'size_residuals_two': size_residuals_two,
            'iou_pred_two': iou_pred_two,
            'heading_class_label_two': heading_class_label_two,
            'heading_residuals_label_two': heading_residuals_label_two,
            'center': center_two,
            'heading_scores': heading_scores,
            'heading_residuals': heading_residuals,
            'size_scores': size_scores_two,
            'size_residuals': size_residuals_two,
            'iou_pred': iou_pred_two,
        }
        return output

class WeakLabelerTwoStageWithImage(nn.Module):
    def __init__(self, cfg, n_channel=3, seg_model=None, track_running_stats=True, dann=False):
        super(WeakLabelerTwoStageWithImage, self).__init__()
        self.cfg = cfg
        self.dann = dann
        self.ins_seg = PointNetInstanceSeg(n_channel=n_channel, track_running_stats=track_running_stats)
        self.point_enc_one = PointFeatureEncoder(cfg=cfg, n_channel=3)
        self.point_enc_two = PointFeatureEncoder(cfg=cfg, n_channel=3)
        self.image_enc = ImageFeatureEncoder()
        self.box_est_one = BoxEstimation(cfg=cfg)
        self.box_est_two = BoxEstimation(cfg=cfg)
        if self.dann:
            self.domain_classifier = nn.Sequential(
                nn.Linear(1024, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Linear(128, 2),
            )

        if seg_model != None:
            self.load_seg(seg_model)
            self.fix_seg()
    
    def load_seg(self, seg_model):
        if seg_model != None:
            self.ins_seg.load_state_dict(torch.load(seg_model)['model_state_dict'])
    
    def fix_seg(self):
        self.ins_seg.eval()
        for param in self.ins_seg.parameters():
            param.requires_grad = False

    def free_seg(self):
        self.ins_seg.train()
        for param in self.ins_seg.parameters():
            param.requires_grad = True
    
    def forward(self, img, pts, box, alpha=0.0):
        '''
            :param img: (bs, 128, 128, 3)
            :param pts: (bs, 3, n)
            :param box: (bs, 7)
        '''
        # 3D Instance Segmentation
        with torch.no_grad():
            logits = self.ins_seg(pts) # (bs, n, 2)
        object_pts_xyz, mask = point_cloud_masking(pts, logits, self.cfg.WEAK_LABEL.NUM_OBJECT_POINT)
        object_pts_xyz = object_pts_xyz.cuda() # (bs, 3, NUM_OBJECT_POINT)
        object_pts_xyz_two = torch.clone(object_pts_xyz) # (bs, 3, NUM_OBJECT_POINT)

        object_pts_xyz_mean = None
        if self.cfg.WEAK_LABEL.get('SEG_TO_CENTER', False):
            object_pts_xyz_mean = torch.mean(object_pts_xyz, dim=2, keepdim=True) # (bs, 3, 1)
            object_pts_xyz -= object_pts_xyz_mean
            
        # 3D Box Estimation
        point_feat_one = self.point_enc_one(object_pts_xyz)              # (bs, 512)
        image_feat = self.image_enc(img)                                 # (bs, 512)
        global_feat_one = torch.cat((point_feat_one, image_feat), dim=1) # (bs, 1024)
        if self.dann:
            reverse_feat = GradientReverse.apply(global_feat_one, alpha)
            domain_output = self.domain_classifier(reverse_feat)
        
        box_pred_one, iou_pred_one = self.box_est_one(global_feat_one)   # (bs, 59), (bs, 1)
        center_one, heading_scores_one, heading_residuals_normalized_one, heading_residuals_one, \
        size_scores_one, size_residuals_normalized_one, size_residuals_one = parse_output_to_tensors(
            box_pred_one,
            self.cfg.WEAK_LABEL.NUM_HEADING_BIN,
            self.cfg.WEAK_LABEL.NUM_SIZE_CLUSTER,
            np.array(self.cfg.WEAK_LABEL.MEAN_SIZE_ARR)
        )
        if self.cfg.WEAK_LABEL.get('SEG_TO_CENTER', False):
            center_one += object_pts_xyz_mean.squeeze(dim=2)
        
        # Coordinate transform & Generate labels
        bs = center_one.shape[0]
        heading_class = np.argmax(heading_scores_one.cpu().detach().numpy(), 1) # (bs,)
        heading_residual = np.array([heading_residuals_one.cpu().detach().numpy()[i, heading_class[i]] for i in range(bs)]) # (bs,)
        size_class = np.argmax(size_scores_one.cpu().detach().numpy(), 1) # (bs,)
        size_residual = np.vstack([size_residuals_one.cpu().detach().numpy()[i, size_class[i], :] for i in range(bs)]) # (bs, 3)
        
        box_size = np.zeros((bs, 3))
        heading_angle = np.zeros((bs, 1))
        for i in range(bs):
            box_size[i, :] = class2size(self.cfg, size_class[i], size_residual[i])
            heading_angle[i] = class2angle(self.cfg, heading_class[i], heading_residual[i])
        box_one = np.concatenate((center_one.cpu().detach().numpy(), box_size, heading_angle), axis=1) # (bs, 7)
        box_one = torch.from_numpy(box_one).float().cuda()
        
        heading_class_label_two = np.zeros((bs,))
        heading_residuals_label_two = np.zeros((bs,))
        object_pts_xyz_two -= box_one[:, :3, np.newaxis]

        for i in range(bs):
            object_pts_xyz_two[i] = rotz(-box_one[i, -1]).float().cuda() @ object_pts_xyz_two[i]
            heading_class_label_two[i], heading_residuals_label_two[i] = angle2class(box[i, -1] - box_one[i, -1], self.cfg.WEAK_LABEL.NUM_HEADING_BIN)

        heading_class_label_two = torch.from_numpy(heading_class_label_two).long().cuda()
        heading_residuals_label_two = torch.from_numpy(heading_residuals_label_two).float().cuda()

        # 3D Box Estimation Two
        point_feat_two = self.point_enc_two(object_pts_xyz_two)          # (bs, 512)
        global_feat_two = torch.cat((point_feat_two, image_feat), dim=1) # (bs, 1024)
        
        box_pred_two, iou_pred_two = self.box_est_two(global_feat_two)   # (bs, 59), (bs, 1)
        center_two, heading_scores_two, heading_residuals_normalized_two, heading_residuals_two, \
        size_scores_two, size_residuals_normalized_two, size_residuals_two = parse_output_to_tensors(
            box_pred_two,
            self.cfg.WEAK_LABEL.NUM_HEADING_BIN,
            self.cfg.WEAK_LABEL.NUM_SIZE_CLUSTER,
            np.array(self.cfg.WEAK_LABEL.MEAN_SIZE_ARR)
        )
        center_two += center_one
        
        # Postprocessing
        heading_class = np.argmax(heading_scores_two.cpu().detach().numpy(), 1) # (bs,)
        heading_residual = np.array([heading_residuals_two.cpu().detach().numpy()[i, heading_class[i]] for i in range(bs)]) # (bs,)
        heading_scores = np.zeros((bs, self.cfg.WEAK_LABEL.NUM_HEADING_BIN))
        heading_residuals = np.zeros((bs, self.cfg.WEAK_LABEL.NUM_HEADING_BIN))

        for i in range(bs):
            heading_angle = class2angle(self.cfg, heading_class[i], heading_residual[i])
            heading_angle += box_one[i, -1]
            heading_class_label, heading_residuals_label = angle2class(heading_angle, self.cfg.WEAK_LABEL.NUM_HEADING_BIN)
            
            heading_scores[i, heading_class_label] = 1
            heading_residuals[i, heading_class_label] = heading_residuals_label

        heading_scores = torch.from_numpy(heading_scores).float().cuda()
        heading_residuals = torch.from_numpy(heading_residuals).float().cuda()

        output = {
            'logits': logits,
            'mask': mask,
            'center_one': center_one,
            'heading_scores_one': heading_scores_one,
            'heading_residuals_normalized_one': heading_residuals_normalized_one,
            'heading_residuals_one': heading_residuals_one,
            'size_scores_one': size_scores_one,
            'size_residuals_normalized_one': size_residuals_normalized_one,
            'size_residuals_one': size_residuals_one,
            'iou_pred_one': iou_pred_one,
            'center_two': center_two,
            'heading_scores_two': heading_scores_two,
            'heading_residuals_normalized_two': heading_residuals_normalized_two,
            'heading_residuals_two': heading_residuals_two,
            'size_scores_two': size_scores_two,
            'size_residuals_normalized_two': size_residuals_normalized_two,
            'size_residuals_two': size_residuals_two,
            'iou_pred_two': iou_pred_two,
            'heading_class_label_two': heading_class_label_two,
            'heading_residuals_label_two': heading_residuals_label_two,
            'center': center_two,
            'heading_scores': heading_scores,
            'heading_residuals': heading_residuals,
            'size_scores': size_scores_two,
            'size_residuals': size_residuals_two,
            'iou_pred': iou_pred_two,
            'domain_output': domain_output if self.dann else None,
        }
        return output

class PointNetInstanceSeg(nn.Module):
    def __init__(self, n_channel=3, track_running_stats=True):
        '''
            3D Instance Segmentation PointNet
            :param n_channel: 3
        '''
        super(PointNetInstanceSeg, self).__init__()
        self.conv1 = nn.Conv1d(n_channel, 64, 1)
        self.conv2 = nn.Conv1d(64, 64, 1)
        self.conv3 = nn.Conv1d(64, 64, 1)
        self.conv4 = nn.Conv1d(64, 128, 1)
        self.conv5 = nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64, track_running_stats=track_running_stats)
        self.bn2 = nn.BatchNorm1d(64, track_running_stats=track_running_stats)
        self.bn3 = nn.BatchNorm1d(64, track_running_stats=track_running_stats)
        self.bn4 = nn.BatchNorm1d(128, track_running_stats=track_running_stats)
        self.bn5 = nn.BatchNorm1d(1024, track_running_stats=track_running_stats)

        self.dconv1 = nn.Conv1d(1088, 512, 1)
        self.dconv2 = nn.Conv1d(512, 256, 1)
        self.dconv3 = nn.Conv1d(256, 128, 1)
        self.dconv4 = nn.Conv1d(128, 128, 1)
        self.dropout = nn.Dropout(p=0.5)
        self.dconv5 = nn.Conv1d(128, 2, 1)
        self.dbn1 = nn.BatchNorm1d(512, track_running_stats=track_running_stats)
        self.dbn2 = nn.BatchNorm1d(256, track_running_stats=track_running_stats)
        self.dbn3 = nn.BatchNorm1d(128, track_running_stats=track_running_stats)
        self.dbn4 = nn.BatchNorm1d(128, track_running_stats=track_running_stats)

    def forward(self, pts):
        '''
            :param pts: [bs, 3, n]: x, y, z, intensity
            :return: logits: [bs, n, 2], scores for bkg/clutter and object
        '''
        bs = pts.size()[0]
        n_pts = pts.size()[2]

        out1 = F.relu(self.bn1(self.conv1(pts)))          # (bs, 64, n)
        out2 = F.relu(self.bn2(self.conv2(out1)))         # (bs, 64, n)
        out3 = F.relu(self.bn3(self.conv3(out2)))         # (bs, 64, n)
        out4 = F.relu(self.bn4(self.conv4(out3)))         # (bs, 128, n)
        out5 = F.relu(self.bn5(self.conv5(out4)))         # (bs, 1024, n)
        global_feat = torch.max(out5, 2, keepdim=True)[0] # (bs, 1024, 1)

        global_feat_repeat = global_feat.view(bs, -1, 1).repeat(1, 1, n_pts) # (bs, 1024, n)
        concat_feat = torch.cat([out2, global_feat_repeat], 1)               # (bs, 1088, n)

        x = F.relu(self.dbn1(self.dconv1(concat_feat)))   # (bs, 512, n)
        x = F.relu(self.dbn2(self.dconv2(x)))             # (bs, 256, n)
        x = F.relu(self.dbn3(self.dconv3(x)))             # (bs, 128, n)
        x = F.relu(self.dbn4(self.dconv4(x)))             # (bs, 128, n)
        x = self.dropout(x)
        x = self.dconv5(x)                                # (bs, 2, n)
        seg_pred = x.transpose(2, 1).contiguous()         # (bs, n, 2)
        return seg_pred

class PointNetEstimation(nn.Module):
    def __init__(self, cfg, n_channel=3):
        '''
            3D Box Estimation Pointnet
            :param cfg: cfg
            :param n_channel: 3
        '''
        super(PointNetEstimation, self).__init__()
        self.NUM_HEADING_BIN = cfg.WEAK_LABEL.NUM_HEADING_BIN
        self.NUM_SIZE_CLUSTER = cfg.WEAK_LABEL.NUM_SIZE_CLUSTER

        self.conv1 = nn.Conv1d(n_channel, 128, 1)
        self.conv2 = nn.Conv1d(128, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)
        self.conv4 = nn.Conv1d(256, 512, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(512)

        self.fc1 = nn.Linear(512, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 3 + self.NUM_HEADING_BIN*2 + self.NUM_SIZE_CLUSTER*4)
        self.fcbn1 = nn.BatchNorm1d(512)
        self.fcbn2 = nn.BatchNorm1d(256)

        self.iou_fc1 = nn.Linear(512, 256)
        self.iou_fc2 = nn.Linear(256, 256)
        self.iou_fc3 = nn.Linear(256, 1)
        self.iou_fcbn1 = nn.BatchNorm1d(256)
        self.iou_fcbn2 = nn.BatchNorm1d(256)

    def forward(self, pts):
        '''
            :param pts: [bs, 3, m]: x, y, z after InstanceSeg
            :return: box_pred: [bs, 3 + NUM_HEADING_BIN*2 + NUM_SIZE_CLUSTER*4]
                including box centers, heading bin class scores and residual,
                and size cluster scores and residual
        '''
        bs = pts.size()[0]
        n_pts = pts.size()[2]

        out1 = F.relu(self.bn1(self.conv1(pts)))              # (bs, 128, n)
        out2 = F.relu(self.bn2(self.conv2(out1)))             # (bs, 128, n)
        out3 = F.relu(self.bn3(self.conv3(out2)))             # (bs, 256, n)
        out4 = F.relu(self.bn4(self.conv4(out3)))             # (bs, 512, n)
        global_feat = torch.max(out4, 2, keepdim=False)[0]    # (bs, 512)

        feat = torch.clone(global_feat)
        x = F.relu(self.fcbn1(self.fc1(feat)))                # (bs, 512)
        x = F.relu(self.fcbn2(self.fc2(x)))                   # (bs, 256)
        box_pred = self.fc3(x)                                # (bs, 3 + NUM_HEADING_BIN*2 + NUM_SIZE_CLUSTER*4)

        feat = torch.clone(global_feat)
        y = F.relu(self.iou_fcbn1(self.iou_fc1(feat)))        # (bs, 256)
        y = F.relu(self.iou_fcbn2(self.iou_fc2(y)))           # (bs, 256)
        iou_pred = self.iou_fc3(y)                            # (bs, 1)
        return box_pred, iou_pred

class PointFeatureEncoder(nn.Module):
    def __init__(self, cfg, n_channel=3):
        super(PointFeatureEncoder, self).__init__()
        self.NUM_HEADING_BIN = cfg.WEAK_LABEL.NUM_HEADING_BIN
        self.NUM_SIZE_CLUSTER = cfg.WEAK_LABEL.NUM_SIZE_CLUSTER

        self.conv1 = nn.Conv1d(n_channel, 128, 1)
        self.conv2 = nn.Conv1d(128, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)
        self.conv4 = nn.Conv1d(256, 512, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(512)

    def forward(self, pts):
        out1 = F.relu(self.bn1(self.conv1(pts)))          # (bs, 128, n)
        out2 = F.relu(self.bn2(self.conv2(out1)))         # (bs, 128, n)
        out3 = F.relu(self.bn3(self.conv3(out2)))         # (bs, 256, n)
        out4 = F.relu(self.bn4(self.conv4(out3)))         # (bs, 512, n)
        point_feat = torch.max(out4, 2, keepdim=False)[0] # (bs, 512)
        return point_feat

class ImageFeatureEncoder(nn.Module):
    def __init__(self):
        super(ImageFeatureEncoder, self).__init__()
        self.resnext = torchvision.models.resnext101_32x8d(pretrained=True)
        self.fc = nn.Linear(2048, 512)
        self.fcbn = nn.BatchNorm1d(512)

    def forward(self, img):
        out = self.resnext.conv1(img)
        out = self.resnext.bn1(out)
        out = self.resnext.relu(out)
        out = self.resnext.maxpool(out)

        out = self.resnext.layer1(out)
        out = self.resnext.layer2(out)
        out = self.resnext.layer3(out)
        out = self.resnext.layer4(out)

        out = self.resnext.avgpool(out)
        out = out.reshape(out.size(0), -1)           # (bs, 2048)
        image_feat = F.relu(self.fcbn(self.fc(out))) # (bs, 512)
        return image_feat

class BoxEstimation(nn.Module):
    def __init__(self, cfg):
        super(BoxEstimation, self).__init__()
        self.NUM_HEADING_BIN = cfg.WEAK_LABEL.NUM_HEADING_BIN
        self.NUM_SIZE_CLUSTER = cfg.WEAK_LABEL.NUM_SIZE_CLUSTER

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 3 + self.NUM_HEADING_BIN*2 + self.NUM_SIZE_CLUSTER*4)
        self.fcbn1 = nn.BatchNorm1d(512)
        self.fcbn2 = nn.BatchNorm1d(256)

        self.iou_fc1 = nn.Linear(1024, 512)
        self.iou_fc2 = nn.Linear(512, 256)
        self.iou_fc3 = nn.Linear(256, 1)
        self.iou_fcbn1 = nn.BatchNorm1d(512)
        self.iou_fcbn2 = nn.BatchNorm1d(256)

    def forward(self, global_feat):
        feat = torch.clone(global_feat)
        x = F.relu(self.fcbn1(self.fc1(feat)))                # (bs, 512)
        x = F.relu(self.fcbn2(self.fc2(x)))                   # (bs, 256)
        box_pred = self.fc3(x)                                # (bs, 3 + NUM_HEADING_BIN*2 + NUM_SIZE_CLUSTER*4)

        feat = torch.clone(global_feat)
        y = F.relu(self.iou_fcbn1(self.iou_fc1(feat)))        # (bs, 512)
        y = F.relu(self.iou_fcbn2(self.iou_fc2(y)))           # (bs, 256)
        iou_pred = self.iou_fc3(y)                            # (bs, 1)
        return box_pred, iou_pred

class LateFusionNetwork(nn.Module):
    def __init__(self, n_channel):
        super(LateFusionNetwork, self).__init__()
        self.fusion_net = nn.Sequential(
            nn.Conv1d(n_channel, 18, 1),
            nn.BatchNorm1d(18),
            nn.ReLU(),
            
            nn.Conv1d(18, 36, 1),
            nn.BatchNorm1d(36),
            nn.ReLU(),

            nn.Conv1d(36, 36, 1),
            nn.BatchNorm1d(36),
            nn.ReLU(),
            nn.Conv1d(36, 1, 1),
        )

    def forward(self, x):
        out = self.fusion_net(x)
        return out

def huber_loss(error, delta=1.0):
    abs_error = torch.abs(error)
    quadratic = torch.clamp(abs_error, max=delta)
    linear = (abs_error - quadratic)
    losses = 0.5 * quadratic ** 2 + delta * linear
    return torch.mean(losses)

class WeakLabelerLoss(nn.Module):
    def __init__(self, cfg):
        super(WeakLabelerLoss, self).__init__()
        self.cfg = cfg
        self.NUM_HEADING_BIN = cfg.WEAK_LABEL.NUM_HEADING_BIN
        self.NUM_SIZE_CLUSTER = cfg.WEAK_LABEL.NUM_SIZE_CLUSTER
        self.MEAN_SIZE_ARR = np.array(cfg.WEAK_LABEL.MEAN_SIZE_ARR)

    def forward(self, output, mask_label, center_label, heading_class_label, \
                heading_residuals_label, size_class_label, size_residuals_label, w_iou=1.0, w_box=2.0):
        '''
        1. Center Regression Loss
            center: torch.Size([bs, 3]) torch.float32
            center_label: [bs, 3]
        2. Heading Loss
            heading_scores: torch.Size([bs, 12]) torch.float32
            heading_residuals_normalized: torch.Size([bs, 12]) torch.float32
            heading_residuals: torch.Size([bs, 12]) torch.float32
            heading_class_label: (bs)
            heading_residuals_label: (bs)
        3. Size Loss
            size_scores: torch.Size([bs, 8]) torch.float32
            size_residuals_normalized: torch.Size([bs, 8, 3]) torch.float32
            size_residuals: torch.Size([bs, 8, 3]) torch.float32
            size_class_label: (bs)
            size_residuals_label: (bs, 3)
        4. IOU Loss
            iou_pred: (bs, 1)
        5. Weighted sum of all losses
            w_iou: float scalar
            w_box: float scalar
        '''
        logits = output['logits']
        center = output['center']
        heading_scores = output['heading_scores']
        heading_residuals_normalized = output['heading_residuals_normalized']
        heading_residuals = output['heading_residuals']
        size_scores = output['size_scores']
        size_residuals_normalized = output['size_residuals_normalized']
        size_residuals = output['size_residuals']
        iou_pred = output['iou_pred']

        bs = logits.shape[0]
        # Center Regression Loss
        center_dist = torch.norm(center - center_label, dim=1) # (bs,)
        center_loss = huber_loss(center_dist, delta=2.0)

        # Heading Loss
        heading_class_loss = F.nll_loss(F.log_softmax(heading_scores, dim=1), heading_class_label.long())
        hcls_onehot = torch.eye(self.NUM_HEADING_BIN)[heading_class_label.long()].cuda() # (bs, 12)
        heading_residuals_normalized_label = heading_residuals_label / (np.pi / self.NUM_HEADING_BIN) # (bs,)
        heading_residuals_normalized_dist = torch.sum(heading_residuals_normalized * hcls_onehot.float(), dim=1) # (bs,)
        heading_residuals_normalized_loss = huber_loss(heading_residuals_normalized_dist - heading_residuals_normalized_label, delta=1.0)
        
        # Size loss
        size_class_loss = F.nll_loss(F.log_softmax(size_scores, dim=1), size_class_label.long())
        scls_onehot = torch.eye(self.NUM_SIZE_CLUSTER)[size_class_label.long()].cuda() # (bs, 3)
        scls_onehot_repeat = scls_onehot.view(-1, self.NUM_SIZE_CLUSTER, 1).repeat(1, 1, 3) # (bs, 3, 3)
        predicted_size_residuals_normalized_dist = torch.sum(size_residuals_normalized * scls_onehot_repeat.cuda(), dim=1) # (bs, 3)
        mean_size_arr_expand = torch.from_numpy(self.MEAN_SIZE_ARR).float().cuda().view(1, self.NUM_SIZE_CLUSTER, 3) # (1, 3, 3)
        mean_size_label = torch.sum(scls_onehot_repeat * mean_size_arr_expand, dim=1) # (bs, 3)
        size_residuals_label_normalized = size_residuals_label / mean_size_label.cuda()
        size_normalized_dist = torch.norm(size_residuals_label_normalized - predicted_size_residuals_normalized_dist, dim=1) # (bs,)
        size_residuals_normalized_loss = huber_loss(size_normalized_dist, delta=1.0)

        # IOU Loss
        _, iou3ds = compute_box3d_iou(
            self.cfg,
            center.cpu().detach().numpy(),
            heading_scores.cpu().detach().numpy(),
            heading_residuals.cpu().detach().numpy(),
            size_scores.cpu().detach().numpy(),
            size_residuals.cpu().detach().numpy(),
            center_label.cpu().detach().numpy(),
            heading_class_label.cpu().detach().numpy(),
            heading_residuals_label.cpu().detach().numpy(),
            size_class_label.cpu().detach().numpy(),
            size_residuals_label.cpu().detach().numpy(),
        )
        iou_loss = F.binary_cross_entropy_with_logits(iou_pred.squeeze(dim=1), torch.from_numpy(iou3ds).float().cuda())

        # Weighted sum of all losses
        total_loss = w_iou * iou_loss + w_box * ( \
            center_loss + \
            heading_class_loss + \
            size_class_loss + \
            heading_residuals_normalized_loss + \
            size_residuals_normalized_loss \
        )

        losses = {
            'total_loss': total_loss,
            'iou_loss': w_iou * iou_loss,
            'center_loss': w_box * center_loss,
            'heading_class_loss': w_box * heading_class_loss,
            'size_class_loss': w_box * size_class_loss,
            'heading_residuals_normalized_loss': w_box * heading_residuals_normalized_loss,
            'size_residuals_normalized_loss': w_box * size_residuals_normalized_loss,
        }
        return losses

class WeakLabelerLossTwoStage(nn.Module):
    def __init__(self, cfg):
        super(WeakLabelerLossTwoStage, self).__init__()
        self.cfg = cfg
        self.NUM_HEADING_BIN = cfg.WEAK_LABEL.NUM_HEADING_BIN
        self.NUM_SIZE_CLUSTER = cfg.WEAK_LABEL.NUM_SIZE_CLUSTER
        self.MEAN_SIZE_ARR = np.array(cfg.WEAK_LABEL.MEAN_SIZE_ARR)

    def forward(self, output, mask_label, center_label, heading_class_label, \
                heading_residuals_label, size_class_label, size_residuals_label, w_iou=1.0, w_box=2.0):
        '''
        1. Center Regression Loss
            center: torch.Size([bs, 3]) torch.float32
            center_label: [bs, 3]
        2. Heading Loss
            heading_scores: torch.Size([bs, 12]) torch.float32
            heading_residuals_normalized: torch.Size([bs, 12]) torch.float32
            heading_residuals: torch.Size([bs, 12]) torch.float32
            heading_class_label: (bs)
            heading_residuals_label: (bs)
        3. Size Loss
            size_scores: torch.Size([bs, 8]) torch.float32
            size_residuals_normalized: torch.Size([bs, 8, 3]) torch.float32
            size_residuals: torch.Size([bs, 8, 3]) torch.float32
            size_class_label: (bs)
            size_residuals_label: (bs, 3)
        4. IOU Loss
            iou_pred: (bs, 1)
        5. Weighted sum of all losses
            w_iou: float scalar
            w_box: float scalar
        '''
        logits = output['logits']
        center_one = output['center_one']
        heading_scores_one = output['heading_scores_one']
        heading_residuals_normalized_one = output['heading_residuals_normalized_one']
        heading_residuals_one = output['heading_residuals_one']
        size_scores_one = output['size_scores_one']
        size_residuals_normalized_one = output['size_residuals_normalized_one']
        size_residuals_one = output['size_residuals_one']
        iou_pred_one = output['iou_pred_one']
        center_two = output['center_two']
        heading_scores_two = output['heading_scores_two']
        heading_residuals_normalized_two = output['heading_residuals_normalized_two']
        heading_residuals_two = output['heading_residuals_two']
        size_scores_two = output['size_scores_two']
        size_residuals_normalized_two = output['size_residuals_normalized_two']
        size_residuals_two = output['size_residuals_two']
        iou_pred_two = output['iou_pred_two']
        heading_class_label_two = output['heading_class_label_two']
        heading_residuals_label_two = output['heading_residuals_label_two']

        bs = logits.shape[0]
        # Center Regression Loss
        center_dist_one = torch.norm(center_one - center_label, dim=1) # (bs,)
        center_loss_one = huber_loss(center_dist_one, delta=2.0)

        center_dist_two = torch.norm(center_two - center_label, dim=1) # (bs,)
        center_loss_two = huber_loss(center_dist_two, delta=2.0)

        # Heading Loss
        heading_class_loss_one = F.nll_loss(F.log_softmax(heading_scores_one, dim=1), heading_class_label.long())
        hcls_onehot_one = torch.eye(self.NUM_HEADING_BIN)[heading_class_label.long()].cuda() # (bs, 12)
        heading_residuals_normalized_label_one = heading_residuals_label / (np.pi / self.NUM_HEADING_BIN) # (bs,)
        heading_residuals_normalized_dist_one = torch.sum(heading_residuals_normalized_one * hcls_onehot_one.float(), dim=1) # (bs,)
        heading_residuals_normalized_loss_one = huber_loss(heading_residuals_normalized_dist_one - heading_residuals_normalized_label_one, delta=1.0)

        heading_class_loss_two = F.nll_loss(F.log_softmax(heading_scores_two, dim=1), heading_class_label_two.long())
        hcls_onehot_two = torch.eye(self.NUM_HEADING_BIN)[heading_class_label_two.long()].cuda() # (bs, 12)
        heading_residuals_normalized_label_two = heading_residuals_label_two / (np.pi / self.NUM_HEADING_BIN) # (bs,)
        heading_residuals_normalized_dist_two = torch.sum(heading_residuals_normalized_two * hcls_onehot_two.float(), dim=1) # (bs,)
        heading_residuals_normalized_loss_two = huber_loss(heading_residuals_normalized_dist_two - heading_residuals_normalized_label_two, delta=1.0)
        
        # Size loss
        size_class_loss_one = F.nll_loss(F.log_softmax(size_scores_one, dim=1), size_class_label.long())
        scls_onehot_one = torch.eye(self.NUM_SIZE_CLUSTER)[size_class_label.long()].cuda() # (bs, 3)
        scls_onehot_repeat_one = scls_onehot_one.view(-1, self.NUM_SIZE_CLUSTER, 1).repeat(1, 1, 3) # (bs, 3, 3)
        predicted_size_residuals_normalized_dist_one = torch.sum(size_residuals_normalized_one * scls_onehot_repeat_one.cuda(), dim=1) # (bs, 3)
        mean_size_arr_expand_one = torch.from_numpy(self.MEAN_SIZE_ARR).float().cuda().view(1, self.NUM_SIZE_CLUSTER, 3) # (1, 3, 3)
        mean_size_label_one = torch.sum(scls_onehot_repeat_one * mean_size_arr_expand_one, dim=1) # (bs, 3)
        size_residuals_label_normalized_one = size_residuals_label / mean_size_label_one.cuda()
        size_normalized_dist_one = torch.norm(size_residuals_label_normalized_one - predicted_size_residuals_normalized_dist_one, dim=1) # (bs,)
        size_residuals_normalized_loss_one = huber_loss(size_normalized_dist_one, delta=1.0)

        size_class_loss_two = F.nll_loss(F.log_softmax(size_scores_two, dim=1), size_class_label.long())
        scls_onehot_two = torch.eye(self.NUM_SIZE_CLUSTER)[size_class_label.long()].cuda() # (bs, 3)
        scls_onehot_repeat_two = scls_onehot_two.view(-1, self.NUM_SIZE_CLUSTER, 1).repeat(1, 1, 3) # (bs, 3, 3)
        predicted_size_residuals_normalized_dist_two = torch.sum(size_residuals_normalized_two * scls_onehot_repeat_two.cuda(), dim=1) # (bs, 3)
        mean_size_arr_expand_two = torch.from_numpy(self.MEAN_SIZE_ARR).float().cuda().view(1, self.NUM_SIZE_CLUSTER, 3) # (1, 3, 3)
        mean_size_label_two = torch.sum(scls_onehot_repeat_two * mean_size_arr_expand_two, dim=1) # (bs, 3)
        size_residuals_label_normalized_two = size_residuals_label / mean_size_label_two.cuda()
        size_normalized_dist_two = torch.norm(size_residuals_label_normalized_two - predicted_size_residuals_normalized_dist_two, dim=1) # (bs,)
        size_residuals_normalized_loss_two = huber_loss(size_normalized_dist_two, delta=1.0)

        # IOU Loss
        _, iou3ds_one = compute_box3d_iou(
            self.cfg,
            center_one.cpu().detach().numpy(),
            heading_scores_one.cpu().detach().numpy(),
            heading_residuals_one.cpu().detach().numpy(),
            size_scores_one.cpu().detach().numpy(),
            size_residuals_one.cpu().detach().numpy(),
            center_label.cpu().detach().numpy(),
            heading_class_label.cpu().detach().numpy(),
            heading_residuals_label.cpu().detach().numpy(),
            size_class_label.cpu().detach().numpy(),
            size_residuals_label.cpu().detach().numpy(),
        )
        iou_loss_one = F.binary_cross_entropy_with_logits(iou_pred_one.squeeze(dim=1), torch.from_numpy(iou3ds_one).float().cuda())

        _, iou3ds_two = compute_box3d_iou(
            self.cfg,
            center_two.cpu().detach().numpy(),
            heading_scores_two.cpu().detach().numpy(),
            heading_residuals_two.cpu().detach().numpy(),
            size_scores_two.cpu().detach().numpy(),
            size_residuals_two.cpu().detach().numpy(),
            center_label.cpu().detach().numpy(),
            heading_class_label_two.cpu().detach().numpy(),
            heading_residuals_label_two.cpu().detach().numpy(),
            size_class_label.cpu().detach().numpy(),
            size_residuals_label.cpu().detach().numpy(),
        )
        iou_loss_two = F.binary_cross_entropy_with_logits(iou_pred_two.squeeze(dim=1), torch.from_numpy(iou3ds_two).float().cuda())

        # Weighted sum of all losses
        total_loss = w_iou * iou_loss_one + w_iou * iou_loss_two + w_box * ( \
            center_loss_one + \
            heading_class_loss_one + \
            size_class_loss_one + \
            heading_residuals_normalized_loss_one + \
            size_residuals_normalized_loss_one + \
            center_loss_two + \
            heading_class_loss_two * 0.1 + \
            size_class_loss_two * 0.1 + \
            heading_residuals_normalized_loss_two + \
            size_residuals_normalized_loss_two \
        )

        losses = {
            'total_loss': total_loss,
            'iou_loss_one': w_iou * iou_loss_one,
            'iou_loss_two': w_iou * iou_loss_two,
            'center_loss_one': w_box * center_loss_one,
            'heading_class_loss_one': w_box * heading_class_loss_one,
            'size_class_loss_one': w_box * size_class_loss_one,
            'heading_residuals_normalized_loss_one': w_box * heading_residuals_normalized_loss_one,
            'size_residuals_normalized_loss_one': w_box * size_residuals_normalized_loss_one,
            'center_loss_two': w_box * center_loss_two,
            'heading_class_loss_two': w_box * heading_class_loss_two * 0.1,
            'size_class_loss_two': w_box * size_class_loss_two * 0.1,
            'heading_residuals_normalized_loss_two': w_box * heading_residuals_normalized_loss_two,
            'size_residuals_normalized_loss_two': w_box * size_residuals_normalized_loss_two,
        }
        return losses

class WeakLabelerDataset(Dataset):
    def __init__(self, cfg, dataloader, use_rgb=False, training=True, fusion_dataset=False):
        self.cfg = cfg
        self.use_rgb = use_rgb
        self.training = training
        self.fusion_dataset = fusion_dataset
        
        self.NUM_POINT = cfg.WEAK_LABEL.NUM_POINT
        self.NUM_HEADING_BIN = cfg.WEAK_LABEL.NUM_HEADING_BIN
        self.NUM_SIZE_CLUSTER = cfg.WEAK_LABEL.NUM_SIZE_CLUSTER
        self.MEAN_SIZE_ARR = np.array(cfg.WEAK_LABEL.MEAN_SIZE_ARR)

        self.data_augmentor_wl = DataAugmentorWL(
            self.cfg.WEAK_LABEL.DATA_AUGMENTOR, self.cfg.CLASS_NAMES
        ) if self.cfg.get('WEAK_LABEL', None) and self.training else None

        self.img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std =[0.229, 0.224, 0.225],
            )
        ])

        dataset_name = cfg.DATA_PATH.split('/')[-1]
        data_path = f'/home/master/10/jacky1212/WL-ST3D/output/wl_data/{dataset_name}_WeakLabelerDataset.pkl'
        if os.path.exists(data_path):
            wl_data = pickle.load(open(data_path, 'rb'))
            self.pts_wl = wl_data['pts_wl']
            self.rgb_wl = wl_data['rgb_wl']
            self.seg_wl = wl_data['seg_wl']
            self.box_wl = wl_data['box_wl']
            self.fru_wl = wl_data['fru_wl']

            self.pose = wl_data['pose']
            self.bbox_wl = wl_data['bbox_wl']
            self.frame_id = wl_data['frame_id']
            self.theta_wl = wl_data['theta_wl']
            self.image_wl = wl_data['image_wl']
            self.calib_file = wl_data['calib_file']
            self.image_shape = wl_data['image_shape']
            return
        
        self.pts_wl = []
        self.rgb_wl = []
        self.seg_wl = []
        self.box_wl = []
        self.fru_wl = []
        
        self.pose = []
        self.bbox_wl = []
        self.frame_id = []
        self.theta_wl = []
        self.image_wl = []
        self.where_wl = []
        self.calib_file = []
        self.image_shape = []
        self.sample_token = []

        dataloader.dataset.eval()
        for data in tqdm(dataloader):
            if len(data['pts_wl']) == 0:
                continue
            batch_size = np.max(data['pts_wl'][:, 0]).astype(np.int) + 1
            for b in range(batch_size):
                batch_pts = data['pts_wl'][data['pts_wl'][:, 0] == b]
                batch_rgb = data['rgb_wl'][data['rgb_wl'][:, 0] == b]
                batch_seg = data['seg_wl'][data['seg_wl'][:, 0] == b]
                if len(batch_pts) == 0:
                    continue

                sample_size = np.max(batch_pts[:, 1]).astype(np.int) + 1
                for s in range(sample_size):
                    pts = batch_pts[batch_pts[:, 1] == s]
                    rgb = batch_rgb[batch_rgb[:, 1] == s]
                    seg = batch_seg[batch_seg[:, 1] == s]
                    
                    pts = pts[:, 2:]
                    rgb = rgb[:, 2:]
                    seg = seg[:, 2:]
                    if len(pts) == 0:
                        continue

                    self.pts_wl.append(pts)
                    self.rgb_wl.append(rgb)
                    self.seg_wl.append(seg)
                    self.box_wl.append(data['box_wl'][b, s])
                    self.fru_wl.append(data['fru_wl'][b, s])
                    
                    self.frame_id.append(data['frame_id'][b])
                    if cfg.WEAK_LABEL.TRANSFER_TO_CENTER:
                        self.theta_wl.append(data['theta_wl'][b, s])
                    else:
                        self.theta_wl.append(-1)
                    img = data['image_wl'][b, s]
                    self.bbox_wl.append([int(float(img[1])), int(float(img[2])), int(float(img[3])), int(float(img[4]))])
                    image_wl = io.imread(img[0])
                    self.image_shape.append(image_wl.shape[:2])
                    image_wl = resize(image_wl[int(float(img[1])): int(float(img[2])), int(float(img[3])): int(float(img[4]))], (64, 64))
                    self.image_wl.append(image_wl)
                    self.calib_file.append(data['calib_file'][b])
                    if 'pose' in data.keys():
                        self.pose.append(data['pose'][b])
                    if dataset_name == 'nuscenes':
                        self.where_wl.append(data['where_wl'][b, s])
                        self.sample_token.append(data['metadata'][b]['token'])

        # wl_data = {
        #     'pts_wl': self.pts_wl,
        #     'rgb_wl': self.rgb_wl,
        #     'seg_wl': self.seg_wl,
        #     'box_wl': self.box_wl,
        #     'fru_wl': self.fru_wl,
        #     'pose': self.pose,
        #     'bbox_wl': self.bbox_wl,
        #     'frame_id': self.frame_id,
        #     'theta_wl': self.theta_wl,
        #     'image_wl': self.image_wl,
        #     'calib_file': self.calib_file,
        #     'image_shape': self.image_shape,
        # }
        # with open(f'/home/master/10/jacky1212/WL-ST3D/output/wl_data/{dataset_name}_WeakLabelerDataset.pkl', 'wb') as f:
        #     pickle.dump(wl_data, f)

    def __len__(self):
        return len(self.pts_wl)
    
    def __getitem__(self, index):
        pts = np.copy(self.pts_wl[index])
        rgb = np.copy(self.rgb_wl[index])
        seg = np.copy(self.seg_wl[index])
        box = np.copy(self.box_wl[index])
        fru = np.copy(self.fru_wl[index])

        if len(self.pose) > 0:
            pose = self.pose[index]
        else:
            pose = np.array([])
        
        bbox_wl = self.bbox_wl[index]
        frame_id = self.frame_id[index]
        theta_wl = self.theta_wl[index]
        image_wl = self.image_wl[index]
        calib_file = self.calib_file[index]
        image_shape = self.image_shape[index]
        if len(self.where_wl) > 0:
            where_wl = self.where_wl[index]
            sample_token = self.sample_token[index]

        if self.cfg.WEAK_LABEL.get('USE_PSEUDO_LABEL', None) and self.training:
            box = self_training_utils_wl.load_ps_label_M2D3D(frame_id, key=f'{theta_wl}')
            if len(box) == 0:
                new_index = np.random.randint(self.__len__())
                return self.__getitem__(new_index)

            seg = roiaware_pool3d_utils.points_in_boxes_cpu(pts, box[np.newaxis, ...]).squeeze(axis=0)
            if np.count_nonzero(seg) == 0:
                new_index = np.random.randint(self.__len__())
                return self.__getitem__(new_index)
            seg = seg[..., np.newaxis]

        if self.use_rgb:
            pts = np.hstack([pts, rgb, seg])
        else:
            pts = np.hstack([pts, seg])

        if self.cfg.WEAK_LABEL.get('RESAMPLE_POINT_CLOUDS', None):
            num_samples = np.count_nonzero(seg)
            front = np.argwhere(seg.squeeze(axis=1)).squeeze(axis=1)
            choice = np.random.choice(np.argwhere(seg.squeeze(axis=1) == 0).squeeze(axis=1), num_samples, replace=(pts.shape[0] - len(front) < num_samples))
            choice = np.concatenate((front, choice))
            pts = pts[choice, :]

        if self.data_augmentor_wl != None and self.training:
            data_dict = self.data_augmentor_wl.forward(
                data_dict = {
                    'box_wl': [box],
                    'pts_wl': [pts],
                    'gt_boxes_mask': np.array([True], dtype=np.bool_),
                }
            )
            box = data_dict['box_wl'][0]
            pts = data_dict['pts_wl'][0]

        if len(pts) == 0:
            new_index = np.random.randint(self.__len__())
            return self.__getitem__(new_index)
        
        if len(pts) >= self.NUM_POINT:
            choice = np.random.choice(len(pts), self.NUM_POINT, replace=False)
        else:
            choice = np.random.choice(len(pts), self.NUM_POINT - len(pts), replace=True)
            choice = np.concatenate((np.arange(len(pts)), choice))
        
        np.random.shuffle(choice)
        _pts = pts[choice, :-1] # (N, 3 or 6)
        _seg = pts[choice,  -1] # (N,)

        _pts = _pts.astype(np.float32)
        box = box.astype(np.float32)

        ########### Frustum visualization ###########
        # io.imsave('img_fru.png', image_wl)
        # np.save('pts.npy', _pts)
        # np.save('seg.npy', _seg)
        # np.save('box.npy', box)
        # np.save('fru.npy', fru)
        # exit()

        ########### Generate labels ###########
        # mask_label
        mask_label = _seg.astype(np.float)
        # center_label
        center_label = box[:3]
        # heading_class_label & heading_residual_label
        heading_class_label, heading_residual_label = angle2class(box[-1], self.NUM_HEADING_BIN)
        # size_class_label & size_residual_label
        size_class_label, size_residual_label = self.size2class(box[3:6])
        if self.fusion_dataset:
            if len(self.sample_token) > 0:
                return self.img_transform(image_wl), torch.from_numpy(_pts), torch.from_numpy(box), theta_wl, calib_file, frame_id, np.array(image_shape), np.array(pose), torch.from_numpy(np.array(bbox_wl)), sample_token, where_wl
            return self.img_transform(image_wl), torch.from_numpy(_pts), torch.from_numpy(box), theta_wl, calib_file, frame_id, np.array(image_shape), np.array(pose), torch.from_numpy(np.array(bbox_wl))
        elif not self.training:
            return self.img_transform(image_wl), torch.from_numpy(_pts), torch.from_numpy(box), theta_wl, calib_file, frame_id, mask_label, center_label, heading_class_label, heading_residual_label, size_class_label, size_residual_label
        else:
            return self.img_transform(image_wl), torch.from_numpy(_pts), torch.from_numpy(box), mask_label, center_label, heading_class_label, heading_residual_label, size_class_label, size_residual_label

    def size2class(self, lwh):
        diff = lwh[np.newaxis, ...] - self.MEAN_SIZE_ARR
        diff = np.linalg.norm(diff, axis=1)
        class_id = np.argmin(diff)
        residual_size = lwh - self.MEAN_SIZE_ARR[class_id]
        return class_id, residual_size

    def eval(self):
        self.training = False

    def train(self):
        self.training = True