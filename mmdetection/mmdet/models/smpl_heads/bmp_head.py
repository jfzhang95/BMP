import mmcv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import h5py
from mmcv.cnn import normal_init, kaiming_init
from mmdet.ops import DeformConv, roi_align, nms
from mmdet.core import multi_apply, bbox2roi
from ..builder import build_loss
from ..registry import HEADS
from ..utils import bias_init_with_prob, ConvModule
from ..utils.smpl.smpl import SMPL
from ..utils.smpl_utils import rot6d_to_rotmat, batch_rodrigues, perspective_projection
from scipy import ndimage

INF = 1e8


def _neg_loss(pred, gt):
    ''' Modified focal loss. Exactly the same as CornerNet.
        Runs faster and costs a little bit more memory
    Arguments:
        pred (batch x c x h x w)
        gt_regr (batch x c x h x w)
    '''
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()

    neg_weights = torch.pow(1 - gt, 4).float()

    loss = 0

    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

    num_pos  = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos
    return loss


class SimpleFocalLoss(nn.Module):
    '''nn.Module warpper for focal loss'''
    def __init__(self):
        super(SimpleFocalLoss, self).__init__()
        self.neg_loss = _neg_loss

    def forward(self, out, target):
        return self.neg_loss(out, target)


def points_nms(heat, kernel=2):
    assert kernel == 2
    # kernel must be 2
    hmax = nn.functional.max_pool2d(
        heat, (kernel, kernel), stride=1, padding=1)
    keep = (hmax[:, :, :-1, :-1] == heat).float()
    return heat * keep


@HEADS.register_module
class BMPHead(nn.Module):

    def __init__(self,
                 num_classes,
                 in_channels,
                 smpl_feat_channels=256,
                 stacked_convs=4,
                 strides=(4, 8, 16, 32, 64),
                 base_edge_list=(16, 32, 64, 128, 256),
                 scale_ranges=((8, 32), (16, 64), (32, 128), (64, 256), (128, 512)),
                 sigma=0.4,
                 num_grids=None,
                 cate_down_pos=0,
                 use_1x1=True,
                 pred_conf=False,
                 use_centernet_loss=False,
                 init_param_file='data/neutral_smpl_mean_params.h5',
                 joint_names=None,
                 joint_map=None,
                 joint_regressor_extra=None,
                 FOCAL_LENGTH=1000,
                 loss_cate=dict(type='FocalLoss', use_sigmoid=True),
                 loss_cfg=dict(type='BMPLoss'),
                 with_deform=False,
                 conv_cfg=None,
                 norm_cfg=None):

        super(BMPHead, self).__init__()
        self.num_classes = num_classes
        self.smpl_num_grids = num_grids
        self.cate_out_channels = self.num_classes - 1
        self.in_channels = in_channels
        self.smpl_feat_channels = smpl_feat_channels
        self.stacked_convs = stacked_convs
        self.strides = strides
        self.sigma = sigma
        self.cate_down_pos = cate_down_pos
        self.base_edge_list = base_edge_list
        self.scale_ranges = scale_ranges
        self.with_deform = with_deform        
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.use_1x1 = use_1x1
        self.pred_conf = pred_conf
        self.use_centernet_loss = use_centernet_loss

        # Load SMPL mean parameters
        f = h5py.File(init_param_file, 'r')
        init_grot = np.array([np.pi, 0., 0.])
        init_pose = np.hstack([init_grot, f['pose'][3:]])
        init_pose = torch.tensor(init_pose.astype('float32'))
        init_rotmat = batch_rodrigues(init_pose.contiguous().view(-1, 3))
        init_contrep = init_rotmat.view(-1, 3, 3)[:, :, :2].contiguous().view(-1)
        init_shape = torch.tensor(f['shape'][:].astype('float32'))
        init_cam = torch.tensor([0.5, 0, 0])

        # Multiply by 6 as we need to estimate two matrixes on each joins
        self.npose = init_rotmat.shape[0] * 6
        self.smpl_out_channels = self.npose + 10 + 3 + 1 if self.pred_conf else self.npose + 10 + 3 # Pose, Shape, Trans

        self.register_buffer('init_contrep', init_contrep)
        self.register_buffer('init_shape', init_shape)
        self.register_buffer('init_cam', init_cam)

        # Build loss
        self.loss_cate = SimpleFocalLoss() if self.use_centernet_loss else build_loss(loss_cate)
        self.loss_smpl = build_loss(loss_cfg)
        
        # Initialize SMPL model
        self.smpl = SMPL('data/smpl')
        self.FOCAL_LENGTH = FOCAL_LENGTH
        
        # Initialize model parameters
        self._init_layers()

    def _init_layers(self):
        norm_cfg = self.norm_cfg
        self.smpl_convs = nn.ModuleList()
        self.cate_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            if i == 0:
                chn = self.in_channels + 2  # 2 is coord dim
            else:
                chn = self.smpl_feat_channels
            self.smpl_convs.append(
                ConvModule(
                    chn,
                    self.smpl_feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    norm_cfg=norm_cfg,
                    bias=norm_cfg is None))
            
            chn = self.in_channels if i == 0 else self.smpl_feat_channels
            self.cate_convs.append(
                ConvModule(
                    chn,
                    self.smpl_feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    norm_cfg=norm_cfg,
                    bias=norm_cfg is None))
        
        self.cate_convs = nn.Sequential(*self.cate_convs)
        self.smpl_convs = nn.Sequential(*self.smpl_convs)

        self.solo_cate = nn.Conv2d(
            self.smpl_feat_channels, self.cate_out_channels, 3, padding=1)
        
        kernel_size, padding = (1, 0) if self.use_1x1 else (3, 1)
        self.solo_smpl = nn.Conv2d(
            self.smpl_feat_channels, self.smpl_out_channels, kernel_size, padding=padding)  # Pose, Shape, Trans

    def init_weights(self):
        for m in self.cate_convs:
            normal_init(m.conv, std=0.01)
        bias_cate = bias_init_with_prob(0.01)
        normal_init(self.solo_cate, std=0.01, bias=bias_cate)

        for m in self.smpl_convs:
            kaiming_init(m.conv)
        kaiming_init(self.solo_smpl)

    def forward(self, feats, eval=False):
        new_feats = self.split_feats(feats)
        featmap_size = [featmap.size()[-2:] for featmap in new_feats]
        smpl_pred, cate_pred = multi_apply(self.forward_single, new_feats, 
                                           list(range(len(self.smpl_num_grids))), eval=eval)
        return smpl_pred, cate_pred        

    def split_feats(self, feats):
        return (F.interpolate(feats[0], scale_factor=0.5, mode='bilinear', align_corners=True), 
                feats[1], 
                feats[2], 
                feats[3], 
                F.interpolate(feats[4], size=feats[3].shape[-2:], mode='bilinear', align_corners=True))

    def forward_single(self, x, idx, eval=False):
        smpl_feat = x
        cate_feat = x
        batch_size = x.shape[0]
        width = x.shape[-1]
        height = x.shape[-2]
        
        # Concate coordinate
        x_range = torch.linspace(-1, 1, smpl_feat.shape[-1], device=smpl_feat.device)
        y_range = torch.linspace(-1, 1, smpl_feat.shape[-2], device=smpl_feat.device)

        y, x = torch.meshgrid(y_range, x_range)
        y = y.expand([smpl_feat.shape[0], 1, -1, -1])
        x = x.expand([smpl_feat.shape[0], 1, -1, -1])
        coord_feat = torch.cat([x, y], 1)
        smpl_feat = torch.cat([smpl_feat, coord_feat], 1)

        smpl_num_grid = self.smpl_num_grids[idx]
        cate_feat = F.interpolate(cate_feat, size=smpl_num_grid, mode='bilinear', align_corners=True)
        smpl_feat = F.interpolate(smpl_feat, size=smpl_num_grid, mode='bilinear', align_corners=True)

        cate_feat = self.cate_convs(cate_feat)
        smpl_feat = self.smpl_convs(smpl_feat)

        # B, cate_out, grid, grid
        cate_pred = self.solo_cate(cate_feat)
        # B, smpl_out, grid, grid
        init_pose = self.init_contrep.view(1, -1, 1, 1).expand(batch_size, -1, smpl_num_grid, smpl_num_grid)
        init_shape = self.init_shape.view(1, -1, 1, 1).expand(batch_size, -1, smpl_num_grid, smpl_num_grid)        
        init_cam = self.init_cam.view(1, -1, 1, 1).expand(batch_size, -1, smpl_num_grid, smpl_num_grid)

        smpl_init = torch.cat([init_pose, init_shape, init_cam], 1)
        if self.pred_conf:
            smpl_pred = self.solo_smpl(smpl_feat)
            smpl_pred[:, :-1] = smpl_pred[:, :-1] + smpl_init
        else:
            smpl_pred = self.solo_smpl(smpl_feat) + smpl_init

        if eval:
            cate_pred = points_nms(cate_pred.sigmoid(), kernel=2).permute(0, 2, 3, 1)
            smpl_pred = smpl_pred.permute(0, 2, 3, 1)
            return smpl_pred, cate_pred

        # reshape to B, grid^2, smpl_out
        cate_pred = cate_pred.permute(0, 2, 3, 1).reshape(cate_pred.size(0), -1, self.cate_out_channels)
        # reshape to B, grid^2, smpl_out
        smpl_pred = smpl_pred.permute(0, 2, 3, 1).reshape(smpl_pred.size(0), -1, self.smpl_out_channels)        
        return smpl_pred, cate_pred

    def loss(self,
             smpl_preds,
             cate_preds,
             gt_bbox_list,
             gt_label_list,
             gt_kpts2d_list=None,             
             gt_kpts3d_list=None,
             gt_shapes_list=None,
             gt_poses_list=None,
             gt_trans_list=None,
             gt_bboxes_ignore=None,
             has_smpls=None,
             imgs=None,
             img_sizes=None,
             img_metas=None,
             scene=None, 
             log_depth=None,
             cfg=None,
             **kwargs):
        """
        smpl_preds: NxR^2x144+10+3 for pose, shape and trans
        cate_preds: NxR^2x1
        """
        pose_label_list, shape_label_list, trans_label_list, kpts2d_label_list, kpts3d_label_list, bbox_label_list, \
                cate_label_list, pose_ind_label_list, has_smpl_label_list, idxs_in_batch_label, pose_idx_label_list = multi_apply(
            self.solo_target_single,
            gt_bbox_list,
            gt_label_list,
            gt_kpts3d_list,
            gt_kpts2d_list,
            gt_shapes_list,
            gt_poses_list,
            gt_trans_list,
            has_smpls,
            img_sizes,
            range(len(gt_bbox_list))
        )

        # pose
        poses_targets = torch.cat([torch.cat([pose_labels_level_img[pose_ind_labels_level_img, ...]
                                 for pose_labels_level_img, pose_ind_labels_level_img in
                                 zip(pose_labels_level, pose_ind_labels_level)], 0)
                      for pose_labels_level, pose_ind_labels_level in zip(zip(*pose_label_list), zip(*pose_ind_label_list))], 0)
                      
        shapes_targets = torch.cat([torch.cat([shape_labels_level_img[shape_ind_labels_level_img, ...]
                                 for shape_labels_level_img, shape_ind_labels_level_img in
                                 zip(shape_labels_level, shape_ind_labels_level)], 0)
                      for shape_labels_level, shape_ind_labels_level in zip(zip(*shape_label_list), zip(*pose_ind_label_list))], 0)

        trans_targets = torch.cat([torch.cat([trans_labels_level_img[trans_ind_labels_level_img, ...]
                                 for trans_labels_level_img, trans_ind_labels_level_img in
                                 zip(trans_labels_level, trans_ind_labels_level)], 0)
                      for trans_labels_level, trans_ind_labels_level in zip(zip(*trans_label_list), zip(*pose_ind_label_list))]  , 0)                                          

        kpts2d_targets = torch.cat([torch.cat([kpts2d_labels_level_img[kpts2d_ind_labels_level_img, ...]
                                 for kpts2d_labels_level_img, kpts2d_ind_labels_level_img in
                                 zip(kpts2d_labels_level, kpts2d_ind_labels_level)], 0)
                      for kpts2d_labels_level, kpts2d_ind_labels_level in zip(zip(*kpts2d_label_list), zip(*pose_ind_label_list))], 0)

        kpts3d_targets = torch.cat([torch.cat([kpts3d_labels_level_img[kpts3d_ind_labels_level_img, ...]
                                 for kpts3d_labels_level_img, kpts3d_ind_labels_level_img in
                                 zip(kpts3d_labels_level, kpts3d_ind_labels_level)], 0)
                      for kpts3d_labels_level, kpts3d_ind_labels_level in zip(zip(*kpts3d_label_list), zip(*pose_ind_label_list))], 0)

        has_smpl_targets = torch.cat([torch.cat([has_smpl_labels_level_img[has_smpl_ind_labels_level_img, ...]
                                 for has_smpl_labels_level_img, has_smpl_ind_labels_level_img in
                                 zip(has_smpl_labels_level, has_smpl_ind_labels_level)], 0)
                      for has_smpl_labels_level, has_smpl_ind_labels_level in zip(zip(*has_smpl_label_list), zip(*pose_ind_label_list))], 0)
        
        bboxes_targets = torch.cat([torch.cat([bbox_labels_level_img[bbox_ind_labels_level_img, ...]
                                 for bbox_labels_level_img, bbox_ind_labels_level_img in
                                 zip(bbox_labels_level, bbox_ind_labels_level)], 0)
                      for bbox_labels_level, bbox_ind_labels_level in zip(zip(*bbox_label_list), zip(*pose_ind_label_list))], 0)

        smpl_preds = torch.cat([torch.cat([smpl_preds_level_img[smpl_ind_labels_level_img, ...]
                                for smpl_preds_level_img, smpl_ind_labels_level_img in
                                zip(smpl_preds_level, smpl_ind_labels_level)], 0)
                    for smpl_preds_level, smpl_ind_labels_level in zip(smpl_preds, zip(*pose_ind_label_list))], 0)
        
        idxs_in_batch = torch.cat([torch.cat([pose_labels_level_img[pose_ind_labels_level_img, ...]
                                 for pose_labels_level_img, pose_ind_labels_level_img in
                                 zip(pose_labels_level, pose_ind_labels_level)], 0)
                      for pose_labels_level, pose_ind_labels_level in zip(zip(*idxs_in_batch_label), zip(*pose_ind_label_list))], 0)
        assert idxs_in_batch.min() > -1

        pose_idx = torch.cat([torch.cat([pose_labels_level_img[pose_ind_labels_level_img, ...]
                                 for pose_labels_level_img, pose_ind_labels_level_img in
                                 zip(pose_labels_level, pose_ind_labels_level)], 0)
                      for pose_labels_level, pose_ind_labels_level in zip(zip(*pose_idx_label_list), zip(*pose_ind_label_list))], 0)
        assert pose_idx.min() > -1

        pose_ind_targets = [
            torch.cat([pose_ind_labels_level_img.flatten()
                       for pose_ind_labels_level_img in pose_ind_labels_level]).int()
            for pose_ind_labels_level in zip(*pose_ind_label_list)
        ]  

        flatten_pose_ind_targets = torch.cat(pose_ind_targets)
        batch_size = flatten_pose_ind_targets.sum()
        
        if self.use_centernet_loss:
            cate_preds = [
                cate_pred.permute(0, 2, 1).view(-1, 1, smpl_num_grid, smpl_num_grid)
                for cate_pred, smpl_num_grid in zip(cate_preds, self.smpl_num_grids)
            ]
            
            cate_targets = [
                torch.cat([cate_labels_img.unsqueeze(0).unsqueeze(0) 
                        for cate_labels_img in cate_labels_level])
                for cate_labels_level in zip(*cate_label_list)
            ]
            loss_cate = []
            loss_cate = torch.cat([self.loss_cate(torch.sigmoid(cate_preds[i]), cate_targets[i]).unsqueeze(0)
                                        for i in range(len(self.smpl_num_grids))])
            loss_cate = torch.mean(loss_cate)
            cate_preds = [cate_pred.flatten().unsqueeze(-1) for cate_pred in cate_preds]
            cate_targets = [cate_target.flatten() for cate_target in cate_targets]
            flatten_cate_targets = torch.cat(cate_targets)
            flatten_cate_preds = torch.cat(cate_preds)
        else:
            # cate loss
            cate_targets = [
                torch.cat([cate_labels_level_img.flatten()
                        for cate_labels_level_img in cate_labels_level])
                for cate_labels_level in zip(*cate_label_list)
            ]
            cate_preds = [
                cate_pred.reshape(-1, self.cate_out_channels) for cate_pred in cate_preds
            ]
            flatten_cate_targets = torch.cat(cate_targets)
            flatten_cate_preds = torch.cat(cate_preds)
            
            loss_cate = self.loss_cate(flatten_cate_preds, flatten_cate_targets, avg_factor=batch_size+1)

        losses = dict()
        losses.update({'loss_cate': loss_cate})
        selected_idx = torch.nonzero(flatten_cate_targets == 1)[0]
        cate_preds_ = torch.sigmoid(flatten_cate_preds[selected_idx][:, 0]) > 0.1
        cate_targets_ = flatten_cate_targets[selected_idx]
        acc = (cate_preds_.int() == cate_targets_.int()).sum() / cate_targets_.size(0)
        losses['acc'] = acc

        bboxes_confidence = flatten_cate_preds[flatten_cate_targets==1, 0]
        # smpl loss
        # we first obtain rotmat, shape and camera from smpl_preds
        pose_preds = smpl_preds[..., :self.npose].contiguous()
        betas_preds = smpl_preds[..., self.npose: self.npose+10].contiguous()
        camera_preds = smpl_preds[..., self.npose+10: self.npose+10+3].contiguous()
        if self.pred_conf:
            conf_preds = torch.sigmoid(smpl_preds[..., -1:].contiguous())
        else:
            conf_preds = None
        # process pred data
        rotmat_preds = rot6d_to_rotmat(pose_preds).view(batch_size, 24, 3, 3)
        smpl_output = self.smpl(betas=betas_preds, body_pose=rotmat_preds[:, 1:],
                                global_orient=rotmat_preds[:, 0].unsqueeze(1), pose2rot=False)
        vertices_preds = smpl_output.vertices
        joints_preds = smpl_output.joints

        smpl_pred = {
            'pred_rotmat': rotmat_preds, 'pred_betas': betas_preds, 
            'pred_camera': camera_preds, 'pred_vertices': vertices_preds, 
            'pred_joints': joints_preds, 'pred_confs': conf_preds
        }
        
        # process target data
        smpl_output = self.smpl(betas=shapes_targets, body_pose=poses_targets[:, 1:], global_orient=poses_targets[:, None, 0], pose2rot=True)
        vertices_targets = smpl_output.vertices

        smpl_target = {
            'gt_keypoints_2d': kpts2d_targets,
            'gt_keypoints_3d': kpts3d_targets,
            'gt_rotmat': poses_targets,
            'gt_shape': shapes_targets,
            'gt_camera': trans_targets, 
            'gt_bboxes': bboxes_targets,
            'has_smpl': has_smpl_targets,
            'gt_vertices': vertices_targets,
            'raw_images': imgs.clone(),
            'img_meta': img_metas,
            'idxs_in_batch': idxs_in_batch,
            'pose_idx': pose_idx,
            'mosh': kwargs.get('mosh', None),
            'scene': scene,
            'log_depth': log_depth,
        }
        
        loss_smpl = self.loss_smpl(smpl_pred, smpl_target, bboxes_confidence=bboxes_confidence, 
                                   discriminator=kwargs.get('discriminator', None), pred_conf=self.pred_conf)
        
        losses.update(loss_smpl)

        return losses, smpl_pred

    def solo_target_single(self,
                           gt_bboxes_raw,
                           gt_labels_raw,
                           gt_kpts3d_raw,
                           gt_kpts2d_raw,
                           gt_shapes_raw,
                           gt_poses_raw,
                           gt_trans_raw,
                           has_smpl_raw,
                           img_size,
                           idx_in_batch):
        # We should use image shape of padded images
        img_width, img_height = img_size
        device = gt_labels_raw[0].device
        gt_areas = torch.sqrt((gt_bboxes_raw[:, 2] - gt_bboxes_raw[:, 0]) * (
                gt_bboxes_raw[:, 3] - gt_bboxes_raw[:, 1]))
        
        pose_label_list = []
        shape_label_list = []
        trans_label_list = []
        cate_label_list = []
        bbox_label_list = []
        kpts2d_label_list = []
        kpts3d_label_list = []
        has_smpl_label_list = []
        pose_ind_label_list = []
        # compute idxs_in_batch and pose_idx
        idxs_in_batch_label_list = []
        pose_idx_label_list = []
        
        for (lower_bound, upper_bound), stride, num_grid \
                in zip(self.scale_ranges, self.strides, self.smpl_num_grids):
            cate_label = torch.zeros([num_grid, num_grid], dtype=torch.int64, device=device)
            pose_label = torch.zeros([num_grid**2, 24, 3], dtype=torch.float32, device=device)
            shape_label = torch.zeros([num_grid**2, 10], dtype=torch.float32, device=device)
            trans_label = torch.zeros([num_grid**2, 3], dtype=torch.float32, device=device)
            bbox_label = torch.zeros([num_grid**2, 4], dtype=torch.float32, device=device)
            kpts2d_label = torch.zeros([num_grid**2, 24, 3], dtype=torch.float32, device=device)
            kpts3d_label = torch.zeros([num_grid**2, 24, 4], dtype=torch.float32, device=device)
            has_smpl_label = torch.zeros([num_grid**2], dtype=torch.int64, device=device)
            pose_ind_label = torch.zeros([num_grid**2], dtype=torch.bool, device=device)
            # initialize idxs_in_batch and pose_idx as -1
            idxs_in_batch_label = torch.zeros([num_grid**2, 1], dtype=torch.int64, device=device) - 1
            pose_idx_label = torch.zeros([num_grid**2, 1], dtype=torch.int64, device=device) - 1

            hit_indices = ((gt_areas >= lower_bound) & (gt_areas <= upper_bound)).nonzero().flatten()
            if len(hit_indices) == 0:
                pose_label_list.append(pose_label)
                shape_label_list.append(shape_label)
                trans_label_list.append(trans_label)
                bbox_label_list.append(bbox_label)
                kpts2d_label_list.append(kpts2d_label)
                kpts3d_label_list.append(kpts3d_label)
                cate_label_list.append(cate_label)
                has_smpl_label_list.append(has_smpl_label)
                pose_ind_label_list.append(pose_ind_label)
                idxs_in_batch_label_list.append(idxs_in_batch_label)
                pose_idx_label_list.append(pose_idx_label)                
                continue

            gt_bboxes = gt_bboxes_raw[hit_indices]
            gt_labels = gt_labels_raw[hit_indices]
            gt_kpts3d = gt_kpts3d_raw[hit_indices]
            gt_kpts2d = gt_kpts2d_raw[hit_indices]
            gt_poses = gt_poses_raw[hit_indices]
            gt_shapes = gt_shapes_raw[hit_indices]
            gt_trans = gt_trans_raw[hit_indices]
            gt_has_smpl = has_smpl_raw[hit_indices]

            half_ws = 0.5 * (gt_bboxes[:, 2] - gt_bboxes[:, 0]) * self.sigma
            half_hs = 0.5 * (gt_bboxes[:, 3] - gt_bboxes[:, 1]) * self.sigma    

            output_stride = stride / 2
            
            for gt_bbox, gt_label, kpts3d, kpts2d, gt_pose, gt_shape, gt_trans, has_smpl, half_h, half_w, hit_idx \
                in zip(gt_bboxes, gt_labels, gt_kpts3d, gt_kpts2d, gt_poses, gt_shapes, gt_trans, gt_has_smpl, half_ws, half_hs, hit_indices):
                # filter out too small object
                if torch.abs(gt_bbox[..., 0] - gt_bbox[..., 2]) < 5 or torch.abs(gt_bbox[..., 1] - gt_bbox[..., 3]) < 5:
                    continue

                # visible joints center
                vis_kpts2d = kpts2d[kpts2d[..., -1] == 1]             
                if kpts2d[14, -1] == 1:
                    # root joint is visible
                    center_w, center_h = kpts2d[14, 0], kpts2d[14, 1]
                elif len(vis_kpts2d) > 0:
                    center_w, center_h = torch.mean(vis_kpts2d[:, 0]), torch.mean(vis_kpts2d[:, 1])
                else:
                    # in cases where the kpts2d are all zero
                    center_w, center_h = (gt_bbox[0] + gt_bbox[2]) / 2, (gt_bbox[1] + gt_bbox[3]) / 2
                # get img shape and compute obj center in grid coordinate
                coord_w = int((center_w / img_width) // (1. / num_grid))
                coord_h = int((center_h / img_height) // (1. / num_grid))
                coord_w = max(0, min(num_grid-1, coord_w))
                coord_h = max(0, min(num_grid-1, coord_h))
                
                # left, top, right, down
                top_box = max(0, int(((center_h - half_h) / img_height) // (1. / num_grid)))
                down_box = min(num_grid - 1, int(((center_h + half_h) / img_height) // (1. / num_grid)))
                left_box = max(0, int(((center_w - half_w) / img_width) // (1. / num_grid)))
                right_box = min(num_grid - 1, int(((center_w + half_w) / img_width) // (1. / num_grid)))

                top = max(top_box, coord_h-1)
                down = min(down_box, coord_h+1)
                left = max(coord_w-1, left_box)
                right = min(right_box, coord_w+1)

                # cate
                cate_label[top:(down+1), left:(right+1)] = gt_label

                # pose
                for i in range(top, down+1):
                    for j in range(left, right+1):
                        position = int(i * num_grid + j)
                        pose_label[position] = gt_pose
                        shape_label[position] = gt_shape
                        trans_label[position] = gt_trans
                        bbox_label[position] = gt_bbox
                        kpts2d_label[position] = kpts2d
                        kpts3d_label[position] = kpts3d
                        has_smpl_label[position] = has_smpl
                        pose_ind_label[position] = True
                        idxs_in_batch_label[position] = idx_in_batch
                        pose_idx_label[position] = hit_idx

            pose_label_list.append(pose_label)
            shape_label_list.append(shape_label)
            trans_label_list.append(trans_label)
            kpts2d_label_list.append(kpts2d_label)
            kpts3d_label_list.append(kpts3d_label)
            bbox_label_list.append(bbox_label)
            cate_label_list.append(cate_label)
            has_smpl_label_list.append(has_smpl_label)
            pose_ind_label_list.append(pose_ind_label)     
            idxs_in_batch_label_list.append(idxs_in_batch_label)
            pose_idx_label_list.append(pose_idx_label)

        return pose_label_list, shape_label_list, trans_label_list, kpts2d_label_list, kpts3d_label_list, bbox_label_list, \
                cate_label_list, pose_ind_label_list, has_smpl_label_list, idxs_in_batch_label_list, pose_idx_label_list
                
    def get_pose(self, smpl_preds, cate_preds, img_metas, cfg, rescale=None, use_gt_bboxes=False, gt_bboxes=None, imgs=None, **kwargs):
        assert len(smpl_preds) == len(cate_preds)
        num_levels = len(cate_preds)
        result_list = []
        img_res = (imgs.size(3), imgs.size(2))

        for img_id in range(len(img_metas)):
            cate_pred_list = [
                cate_preds[i][img_id].view(-1, self.cate_out_channels).detach() for i in range(num_levels)]
            smpl_pred_list = [
                smpl_preds[i][img_id].view(-1, self.smpl_out_channels).detach() for i in range(num_levels)]
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']

            cate_pred_list = torch.cat(cate_pred_list, dim=0)
            smpl_pred_list = torch.cat(smpl_pred_list, dim=0)   

            result = self.get_pose_single(cate_pred_list, smpl_pred_list,
                                          img_shape, img_res, scale_factor, cfg, rescale)
            result_list.append(result)
        return result_list

    def get_pose_single(self,
                        cate_preds,
                        smpl_preds,
                        img_shape,
                        img_res,
                        scale_factor,
                        cfg,
                        rescale=False):
        assert len(cate_preds) == len(smpl_preds)
        
        # overall info.
        h, w, _ = img_shape

        # class threshold, samples with scores lower than it will not be considered.
        cls_inds = (cate_preds > cfg.score_thr)
        # category scores.
        cate_scores = cate_preds[cls_inds]
        if len(cate_scores) == 0:
            return None
        # category labels.
        cls_inds = cls_inds.nonzero()

        # poses.
        if self.pred_conf:
            conf_preds = torch.sigmoid(smpl_preds[cls_inds[:, 0], -1])
            smpl_preds = smpl_preds[cls_inds[:, 0], :-1]
        else:
            smpl_preds = smpl_preds[cls_inds[:, 0]]
            conf_preds = torch.ones(smpl_preds.size(0), device=smpl_preds.device).float()

        # pose threshold, samples with scores lower than it will not be considered.
        inds = (conf_preds > cfg.pose_thr)
        inds = inds.nonzero()
        cate_scores = cate_scores[inds[:, 0]]
        if len(cate_scores) == 0:
            return None     

        conf_preds = conf_preds[inds[:, 0]]
        smpl_preds = smpl_preds[inds[:, 0]]
        cate_scores = cate_scores * conf_preds

        # sort and keep top nms_pre
        sort_inds = torch.argsort(cate_scores, descending=True)
        if len(sort_inds) > cfg.nms_pre:
            sort_inds = sort_inds[:cfg.nms_pre]
        
        cate_scores = cate_scores[sort_inds]
        conf_preds = conf_preds[sort_inds]
        smpl_preds = smpl_preds[sort_inds]
        
        # from smpl to kps2d to bbox to bbox nms
        batch_size = cate_scores.size(0)
        pred_pose = smpl_preds[..., :self.npose].contiguous()
        pred_betas = smpl_preds[..., self.npose: self.npose+10].contiguous()
        pred_camera = smpl_preds[..., self.npose+10:].contiguous()
        pred_rotmat = rot6d_to_rotmat(pred_pose).view(batch_size, 24, 3, 3)

        smpl_output = self.smpl(betas=pred_betas, body_pose=pred_rotmat[:, 1:], 
            global_orient=pred_rotmat[:, 0].unsqueeze(1), pose2rot=False)
        pred_vertices = smpl_output.vertices
        pred_joints = smpl_output.joints                                
        img_size = torch.zeros(batch_size, 2).to(pred_joints.device)
        img_size += torch.tensor(img_res).float().to(pred_joints.device)
        rotation_Is = torch.eye(3).unsqueeze(0).repeat(batch_size, 1, 1).to(pred_joints.device)
        imgs_size = torch.max(img_size, 1)[0]
        camera_center = img_size / 2
        depth = 2 * self.FOCAL_LENGTH / (1e-8 + pred_camera[..., 0] * imgs_size)
        translation = torch.zeros((batch_size, 3), dtype=pred_camera.dtype).to(
            pred_joints.device)
        translation[:, :-1] = pred_camera[:, 1:]
        translation[:, -1] = depth
        focal_length = self.FOCAL_LENGTH * torch.ones_like(depth)

        pred_keypoints_2d_smpl = perspective_projection(
            pred_joints,
            rotation_Is,
            translation,
            focal_length,
            camera_center
        )      
        
        # Obtain bbox from 2d kpts, then do nms 
        x1 = torch.clamp(torch.min(pred_keypoints_2d_smpl[..., 0:1], dim=1)[0], 0, img_res[0])
        y1 = torch.clamp(torch.min(pred_keypoints_2d_smpl[..., 1:2], dim=1)[0], 0, img_res[1])
        x2 = torch.clamp(torch.max(pred_keypoints_2d_smpl[..., 0:1], dim=1)[0], 0, img_res[0])
        y2 = torch.clamp(torch.max(pred_keypoints_2d_smpl[..., 1:2], dim=1)[0], 0, img_res[1])

        cate_scores = cate_scores.unsqueeze(-1)
        pred_bboxes = torch.cat((x1, y1, x2, y2, cate_scores), dim=-1)
        # KPTS NMS
        pred_bboxes, inds = nms_oks(pred_keypoints_2d_smpl, pred_bboxes, cfg.oks_thr)

        pred_rotmat = pred_rotmat[inds]
        pred_betas = pred_betas[inds]
        pred_vertices = pred_vertices[inds]
        pred_joints = pred_joints[inds]
        pred_keypoints_2d_smpl = pred_keypoints_2d_smpl[inds]
        translation = translation[inds]        
        pred_scores = pred_bboxes[:, 4]
        sort_inds = torch.argsort(pred_scores, descending=True)
        
        if len(sort_inds) > cfg.max_per_img:
            sort_inds = sort_inds[:cfg.max_per_img]
            pred_bboxes = pred_bboxes[sort_inds]
            pred_rotmat = pred_rotmat[sort_inds]
            pred_betas = pred_betas[sort_inds]
            pred_vertices = pred_vertices[sort_inds]
            pred_keypoints_2d_smpl = pred_keypoints_2d_smpl[sort_inds]
            pred_joints = pred_joints[sort_inds]
            translation = translation[sort_inds]
        
        pred_smpl = {
            'pred_rotmat': pred_rotmat, 'pred_betas': pred_betas, 
            'pred_translation': translation, 'pred_vertices': pred_vertices, 
            'pred_joints': pred_joints, 'pred_kpts2d': pred_keypoints_2d_smpl, 
            'bboxes': pred_bboxes
        }

        return pred_smpl


def nms_oks(kp_predictions, rois, thresh):
    """Nms based on kp predictions."""

    scores = rois[..., -1]
    order = torch.argsort(scores, descending=True)
    keep = []
    while order.size(0) > 0:
        i = int(order[0].cpu().numpy())
        keep.append(i)
        ovr = compute_oks(
            kp_predictions[order[1:]], kp_predictions[i], rois[i])
        inds = torch.nonzero(ovr <= thresh)[:, 0]
        order = order[inds + 1]
    keep = torch.from_numpy(np.array(keep)).to(rois.device)
    return rois[keep], keep


def compute_oks(dst_keypoints_2d, src_keypoints_2d, src_bboxes):
    """
    Compute OKS for predicted and GT keypoints.
    pred_keypoints_2d: BxKx2
    gt_keypoints_2d: BxKx3 (last dim for visibility)
    gt_bboxes: Bx4
    """
    
    sigmas = np.array([
        .89, .87, 1.07, 1.07, .87, .89, .62, .72, .79, .79, .72, .62, .26, .26,
        1.07, .79, .79, .26, .26, .26, .25, .25, .25, .25], dtype=np.float32) / 10.0        
    sigmas = torch.from_numpy(sigmas).to(dst_keypoints_2d.device).unsqueeze(0)
    vars = (sigmas * 2) ** 2
    # area 
    area = (src_bboxes[..., 2] - src_bboxes[..., 0] + 1) * (src_bboxes[..., 3] - src_bboxes[..., 1] + 1)
    area = area.unsqueeze(-1)
    
    # measure the per-keypoint distance if keypoints are visible
    dx = dst_keypoints_2d[..., 0] - src_keypoints_2d[..., 0]
    dy = dst_keypoints_2d[..., 1] - src_keypoints_2d[..., 1]
    
    e = (dx**2 + dy**2) / vars / (area + 1e-8) / 2
    e = torch.sum(torch.exp(-e), dim=1) / e.shape[1]

    return e.unsqueeze(-1)