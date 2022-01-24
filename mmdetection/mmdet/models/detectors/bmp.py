import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .base import BaseDetector
from ..registry import DETECTORS
from .. import builder
from ..utils.smpl_utils import rot6d_to_rotmat, batch_rodrigues, rotation_matrix_to_angle_axis
from mmdet.datasets.utils import flip_pose_batch


@DETECTORS.register_module
class BMP(BaseDetector):

    def __init__(self,
                 backbone,
                 neck,
                 smpl_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):

        super(BMP, self).__init__()
        self.backbone = builder.build_backbone(backbone)

        if neck is not None:
            self.neck = builder.build_neck(neck)
        
        self.smpl_head = builder.build_head(smpl_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.init_weights(pretrained=pretrained)

    def init_weights(self, pretrained=None):
        super(BMP, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        if self.with_neck:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()
        self.smpl_head.init_weights()
    
    def extract_feat(self, img):
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_dummy(self, img):
        x = self.extract_feat(img)
        outs = self.smpl_head(x)
        return outs

    def forward_train(self,
                      imgs,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_kpts3d=None,
                      gt_kpts2d=None,
                      gt_shapes=None,
                      gt_poses=None,
                      gt_trans=None,
                      gt_depths=None,
                      has_smpl=None,
                      disable_smpl=False,
                      return_pred=False,
                      scene=None,
                      log_depth=None,
                      **kwargs):
        x = self.extract_feat(imgs)
        outs = self.smpl_head(x)
        img_sizes = torch.zeros(imgs.size(0), 2).to(imgs.device)
        img_sizes += torch.tensor(imgs.shape[:-3:-1], dtype=img_sizes.dtype).to(img_sizes.device) 
        loss_inputs = outs + (gt_bboxes, gt_labels, gt_kpts2d, gt_kpts3d, gt_shapes, gt_poses, gt_trans,
                              gt_bboxes_ignore, has_smpl, imgs, img_sizes, img_metas, scene, log_depth, self.train_cfg)
        losses, smpl_pred = self.smpl_head.loss(
            *loss_inputs, **kwargs)
        
        if return_pred:
            return losses, smpl_pred
        else:
            return losses

    def simple_test(self, img, img_meta, rescale=False, use_gt_bboxes=False, gt_bboxes=None, **kwargs):
        x = self.extract_feat(img)
        outs = self.smpl_head(x, eval=True)
        pose_inputs = outs + (img_meta, self.test_cfg, rescale, use_gt_bboxes, gt_bboxes, img)
        pred_results = self.smpl_head.get_pose(*pose_inputs, **kwargs)[0]
        return pred_results

    def aug_test(self, imgs, img_metas, rescale=False, use_gt_bboxes=False, gt_bboxes=None, **kwargs):
        imgs_per_gpu = imgs.size(0)
        assert imgs_per_gpu == 2  # flip augmentation
        x = self.extract_feat(imgs)
        smpl_preds, cate_preds = self.smpl_head(x, eval=True) # smpl_pred, cate_pred
        smpl_preds_pose = []
        w = img_metas[0]['img_shape'][1]
        for i in range(len(cate_preds)):
            # Flip results
            cate_preds[i][1] = torch.flip(cate_preds[i][1], dims=[1])
            smpl_preds[i][1] = torch.flip(smpl_preds[i][1], dims=[1])
            # Flip pose
            pred_rotmat = rot6d_to_rotmat(smpl_preds[i][1, :, :, :144].contiguous().view(-1, 144))
            bs = pred_rotmat.size(0) // 24
            grid = int(np.sqrt(bs))
            pred_rotmat_hom = torch.cat([pred_rotmat, torch.tensor([0, 0, 1], 
                device=pred_rotmat.device).float().view(1, 3, 1).expand(bs * 24, -1, -1)], dim=-1)
            pred_pose = rotation_matrix_to_angle_axis(pred_rotmat_hom)
            pred_pose[torch.isnan(pred_pose)] = 0.0
            pred_pose = flip_pose_batch(pred_pose.contiguous().view(-1, 72))
            pred_rotmat = batch_rodrigues(pred_pose.contiguous().view(-1, 3))
            pred_rotmat = pred_rotmat.contiguous().view(grid, grid, -1)
            pred_6d = pred_rotmat.view(grid, grid, -1, 3, 3)[..., :2].contiguous().view(grid, grid, -1)
            smpl_preds[i][1, :, :, :144] = pred_6d
            # Flip camera
            smpl_preds[i][1, :, :, 144+10+1: 144+10+2] = -smpl_preds[i][1, :, :, 144+10+1: 144+10+2]
            # Fuse results
            cate_preds[i] = torch.mean(cate_preds[i], dim=0, keepdim=True)
            smpl_preds[i] = torch.mean(smpl_preds[i], dim=0, keepdim=True)

        outs = (smpl_preds, cate_preds)
        # Fuse results
        pose_inputs = outs + (img_metas, self.test_cfg, rescale, use_gt_bboxes, gt_bboxes, imgs)
        pred_results = self.smpl_head.get_pose(*pose_inputs, **kwargs)[0]
        return pred_results

    def forward_test(self, imgs, img_metas, **kwargs):
        num_augs = len(imgs)
        if num_augs != len(img_metas):
            img_metas = [img_metas] # Hack for demo code
    
        imgs_per_gpu = imgs.size(0)

        if num_augs == 1:
            return self.simple_test(imgs, img_metas, **kwargs)
        else:
            assert imgs_per_gpu == 2  # Support flip-aug test only
            return self.aug_test(imgs, img_metas, **kwargs)