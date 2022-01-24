import torch
import torch.nn as nn

from .utils import weighted_loss
from ..registry import LOSSES
from ..utils.smpl_utils import batch_rodrigues, perspective_projection, J24_TO_J14, H36M_TO_J14
from ..utils.pose_utils import reconstruction_error, reconstruction_error_vis
from ..utils.smpl.smpl import SMPL
import random
import numpy as np
torch.autograd.set_detect_anomaly(True)


def batch_adv_disc_l2_loss(real_disc_value, fake_disc_value):
    ka = real_disc_value.shape[0]
    kb = fake_disc_value.shape[0]
    lb, la = torch.sum(fake_disc_value ** 2) / kb, torch.sum((real_disc_value - 1) ** 2) / ka
    return la, lb, la + lb


def batch_encoder_disc_l2_loss(disc_value):
    k = disc_value.shape[0]
    return torch.sum((disc_value - 1.0) ** 2) * 1.0 / k


@weighted_loss
def smpl_loss(pred, target, beta=1.0):
    assert beta > 0
    assert pred.size() == target.size() and target.numel() > 0
    diff = torch.abs(pred - target)
    loss = torch.where(diff < beta, 0.5 * diff * diff / beta,
                       diff - 0.5 * beta)
    return loss


def compute_similarity_transform(S1, S2):
    '''
    Computes a similarity transform (sR, t) that takes
    a set of 3D points S1 (3 x N) closest to a set of 3D points S2,
    where R is an 3x3 rotation matrix, t 3x1 translation, s scale.
    i.e. solves the orthogonal Procrutes problem.
    '''
    transposed = False
    if S1.shape[1] != 3 and S1.shape[1] != 2:
        S1 = S1.permute(0, 2, 1)
        S2 = S2.permute(0, 2, 1)
        transposed = True
    assert(S2.shape[1] == S1.shape[1])

    # 1. Remove mean.
    mu1 = torch.mean(S1, dim=-1, keepdim=True)
    mu2 = torch.mean(S2, dim=-1, keepdim=True)

    X1 = S1 - mu1
    X2 = S2 - mu2

    # 2. Compute variance of X1 used for scale.
    var1 = torch.sum(X1**2, dim=1).sum(dim=1)

    # 3. The outer product of X1 and X2.
    K = X1.bmm(X2.permute(0, 2, 1)) + 1e-8

    # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are
    # singular vectors of K.
    U, s, V = batch_svd(K)

    # Construct Z that fixes the orientation of R to get det(R)=1.
    Z = torch.eye(U.shape[1], device=S1.device).unsqueeze(0)
    Z = Z.repeat(U.shape[0], 1, 1)
    Z[:, -1, -1] *= torch.sign(batch_det(U.bmm(V.permute(0, 2, 1))))

    # Construct R.
    R = V.bmm(Z.bmm(U.permute(0, 2, 1)))

    # 5. Recover scale.
    scale = torch.cat([torch.trace(x).unsqueeze(0) for x in R.bmm(K)]) / var1

    # 6. Recover translation.
    t = mu2 - (scale.unsqueeze(-1).unsqueeze(-1) * (R.bmm(mu1)))

    # 7. Error:
    S1_hat = scale.unsqueeze(-1).unsqueeze(-1) * R.bmm(S1) + t

    if transposed:
        S1_hat = S1_hat.permute(0, 2, 1)

    return S1_hat


@torch.jit.script
def batch_svd(H):
    batch_size = H.shape[0]
    U_batch, s_batch, V_batch = [],[],[]
    for i in range(batch_size):
        U, s, V = H[i].svd()
        U_batch.append(U.unsqueeze(0))
        s_batch.append(s.unsqueeze(0))
        V_batch.append(V.unsqueeze(0))
    return torch.cat(U_batch, 0), torch.cat(s_batch, 0), torch.cat(V_batch, 0)   


@torch.jit.script
def batch_det(A):
    batch_size = A.shape[0]
    A_det = []
    for i in range(batch_size):
        A_det.append(A[i].det().unsqueeze(0))
    return torch.cat(A_det, 0)


@torch.no_grad()
def select_index(im_id, pids, metric, invalid_mask=None):
    im_id = im_id.clone().int()[:, 0]
    num_imgs = im_id.max().item()
    selected_idxs = list()
    full_idxs = torch.arange(im_id.shape[0], device=im_id.device)
    for bid in set(im_id.tolist()):
        batch_mask = bid == im_id
        cur_pids = pids[batch_mask]
        cur_select = list()
        for pid in set(cur_pids.tolist()):
            person_mask = (pid == cur_pids)
            idx_to_select = full_idxs[batch_mask][person_mask][metric[batch_mask][person_mask].argmax()]
            if invalid_mask and invalid_mask[idx_to_select]:
                continue
            cur_select.append(idx_to_select)
        selected_idxs.append(cur_select)
    return selected_idxs


def adversarial_loss(discriminator, pred_pose_shape, real_pose_shape):
    loss_disc = batch_encoder_disc_l2_loss(discriminator(pred_pose_shape))
    fake_pose_shape = pred_pose_shape.detach()
    fake_disc_value, real_disc_value = discriminator(fake_pose_shape), discriminator(real_pose_shape)
    d_disc_real, d_disc_fake, d_disc_loss = batch_adv_disc_l2_loss(real_disc_value, fake_disc_value)
    return loss_disc, d_disc_fake, d_disc_real


@LOSSES.register_module
class BMPLoss(nn.Module):

    def __init__(self, beta=1.0, reduction='mean', loss_weight=1.0, eval_pose=False, re_weight=None,
                 normalize_kpts=False, pad_size=False, JOINT_REGRESSOR_H36M='data/J_regressor_h36m.npy',
                 adversarial_cfg=None, FOCAL_LENGTH=1000, kpts_loss_type='L1Loss', 
                 kpts_3d_loss_type=None, img_size=None, **kwargs):
        super(BMPLoss, self).__init__()
        self.beta = beta
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.criterion_shape = nn.L1Loss()
        self.criterion_keypoints = getattr(nn, kpts_loss_type)(reduction='none')  # nn.L1Loss(reduction='none')
        if kpts_3d_loss_type is not None:
            self.criterion_3d_keypoints = getattr(nn, kpts_3d_loss_type)(reduction='none')
        self.criterion_regr = nn.MSELoss()
        self.eval_pose = eval_pose
        self.re_weight = re_weight
        self.normalize_kpts = normalize_kpts
        self.pad_size = pad_size
        self.FOCAL_LENGTH = FOCAL_LENGTH
        # Initialize SMPL model
        self.smpl = SMPL('data/smpl')
        self.J_regressor = torch.from_numpy(np.load(JOINT_REGRESSOR_H36M)).float()
        self.adversarial_cfg = adversarial_cfg

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                bboxes_confidence=None,
                discriminator=None,
                pred_conf=False,
                **kwargs):
        """
        :param pred: SMPL parameters with 24*6+10+3
        :param target: same as pred
        :param weight:
        :param avg_factor:
        :param reduction_override:
        :param kwargs:
        :param bboxes_confidence:
        :return: loss: dict. All the value whose keys contain 'loss' will be summed up.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        pred_rotmat = pred['pred_rotmat']
        pred_camera = pred['pred_camera']
        pred_joints = pred['pred_joints']
        pred_vertices = pred['pred_vertices']
        pred_betas = pred['pred_betas']
        pred_confs = pred['pred_confs']

        gt_rotmat = target['gt_rotmat']  # It's not rotmat actually. This is a (B, 24, 3) tensor.
        gt_shape = target['gt_shape']
        gt_camera = target['gt_camera']
        gt_keypoints_2d = target['gt_keypoints_2d']
        gt_keypoints_3d = target['gt_keypoints_3d']
        has_smpl = target['has_smpl']
        gt_vertices = target['gt_vertices']
        gt_bboxes = target['gt_bboxes']
        raw_images = target['raw_images']
        img_meta = target['img_meta']
        ori_shape = [i['ori_shape'] for i in img_meta]
        idxs_in_batch = target['idxs_in_batch']
        pose_idx = target['pose_idx']
        scene = target['scene']
        batch_size = pred_joints.shape[0]

        if self.pad_size:
            img_pad_shape = torch.tensor([i['pad_shape'][:2] for i in img_meta], dtype=torch.float32).to(
                pred_joints.device)
            img_size = img_pad_shape[idxs_in_batch[:, 0].long()]
        else:
            img_size = torch.zeros(batch_size, 2).to(pred_joints.device)
            img_size += torch.tensor(raw_images.shape[:-3:-1], dtype=img_size.dtype).to(img_size.device)
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
        gt_keypoints_2d_orig = gt_keypoints_2d.clone()
        pred_keypoints_2d_smpl_orig = pred_keypoints_2d_smpl.clone()
        if pred_conf:
            oks = self.compute_oks(pred_keypoints_2d_smpl_orig, gt_keypoints_2d_orig, gt_bboxes)
            loss_conf = self.confidence_loss(pred_confs, oks)
        else:
            loss_conf = torch.tensor(0).float().cuda()
        
        if self.normalize_kpts:
            scaled_img_shape = torch.tensor([(i['img_shape'][1], i['img_shape'][0]) for i in img_meta], dtype=torch.float32).to(
                pred_joints.device)
            scaled_img_size = scaled_img_shape[idxs_in_batch[:, 0].long()]
            center_pts = scaled_img_size / 2
            pred_keypoints_2d_smpl = (pred_keypoints_2d_smpl - center_pts.unsqueeze(1)) / scaled_img_size.unsqueeze(1)
            gt_keypoints_2d[..., :2] = (gt_keypoints_2d[..., :2] - center_pts.unsqueeze(1)) / scaled_img_size.unsqueeze(1)
        else:
            scaled_img_shape = torch.tensor([(i['img_shape'][1], i['img_shape'][0]) for i in img_meta], dtype=torch.float32).to(
                pred_joints.device)
            scaled_img_size = scaled_img_shape[idxs_in_batch[:, 0].long()]
            pred_keypoints_2d_smpl = pred_keypoints_2d_smpl / scaled_img_size.unsqueeze(1)
            gt_keypoints_2d[..., :2] = gt_keypoints_2d[..., :2] / scaled_img_size.unsqueeze(1)     
        loss_keypoints_smpl, error_ranks = self.keypoint_loss(pred_keypoints_2d_smpl, gt_keypoints_2d)
        loss_keypoints_3d_smpl = self.keypoint_3d_loss(pred_joints, gt_keypoints_3d)
        loss_shape_smpl = self.shape_loss(pred_vertices, gt_vertices, has_smpl)
        loss_regr_pose, loss_regr_betas = self.smpl_losses(pred_rotmat, pred_betas, gt_rotmat, gt_shape, has_smpl)

        loss_dict = {
            'loss_keypoints_smpl': loss_keypoints_smpl*4, 'loss_keypoints_3d_smpl': loss_keypoints_3d_smpl*4,
            'loss_shape_smpl': loss_shape_smpl, 'loss_regr_pose': loss_regr_pose, 
            'loss_regr_betas': loss_regr_betas * 0.01, 'loss_conf': loss_conf,
            'img$raw_images': raw_images.detach(), 'img$idxs_in_batch': idxs_in_batch.detach(),
            'img$pose_idx': pose_idx.detach(),
            'img$pred_vertices': pred_vertices.detach(),
            'img$translation': translation.detach(), 'img$error_rank': -bboxes_confidence.detach(),
            'img$pred_bboxes': gt_bboxes.detach(),
            'img$pred_keypoints_2d_smpl': (pred_keypoints_2d_smpl_orig[:, -24:, :]).detach().clone(),
            'img$gt_keypoints_2d': gt_keypoints_2d_orig.detach().clone()}

        if self.adversarial_cfg:
            valid_batch_size = pred_rotmat.shape[0]
            pred_pose_shape = torch.cat([pred_rotmat.view(valid_batch_size, -1), pred_betas], dim=1)
            loss_dict.update(
                {'pred_pose_shape': pred_pose_shape})

        if self.re_weight:
            for k, v in self.re_weight.items():
                if k.startswith('adv_loss'):
                    loss_dict[k] *= v
                else:
                    loss_dict[f'loss_{k}'] *= v

        if self.eval_pose:
            # 3D pose evaluation
            with torch.no_grad():
                # Regressor broadcasting
                # Get 14 ground truth joints
                # If we want to evaluate on MuPoTS-3D, the strategy for finding most confidence
                # indexes should be changed as we are assuming there are only one person for each images here.

                idxs_in_batch = idxs_in_batch.clone().int()[:, 0]
                num_imgs = idxs_in_batch.max().item()
                selected_idx = list()
                full_idxs = torch.arange(idxs_in_batch.shape[0])
                
                for idx in set(idxs_in_batch.tolist()):
                    arg_max_conf = bboxes_confidence[idx == idxs_in_batch].argmax().item()
                    idx_to_select = full_idxs[idxs_in_batch == idx][arg_max_conf]
                    if gt_keypoints_3d[idx_to_select][:, -1].sum() < 1:
                        continue
                    selected_idx.append(idx_to_select)

                if selected_idx:
                    selected_idx = torch.tensor(selected_idx).long()
                    # To evaluate on bbox with highest confidence value.
                    gt_keypoints_3d = gt_keypoints_3d[selected_idx]
                    pred_keypoints_3d_smpl = pred_joints[selected_idx]
                    pred_vertices = pred_vertices[selected_idx]

                    visible_kpts = gt_keypoints_3d[:, J24_TO_J14, -1].clone()
                    visible_kpts[visible_kpts > 0.1] = 1
                    visible_kpts[visible_kpts <= 0.1] = 0
                    gt_pelvis_smpl = gt_keypoints_3d[:, [14], :-1].clone()
                    gt_keypoints_3d = gt_keypoints_3d[:, J24_TO_J14, :-1].clone()
                    gt_keypoints_3d = gt_keypoints_3d - gt_pelvis_smpl

                    J_regressor_batch = self.J_regressor[None, :].expand(pred_vertices.shape[0], -1, -1).to(
                        pred_vertices.device)
                    # Get 14 predicted joints from the SMPL mesh
                    pred_keypoints_3d_smpl = torch.matmul(J_regressor_batch, pred_vertices)
                    pred_pelvis_smpl = pred_keypoints_3d_smpl[:, [0], :].clone()
                    pred_keypoints_3d_smpl = pred_keypoints_3d_smpl[:, H36M_TO_J14, :]
                    pred_keypoints_3d_smpl = pred_keypoints_3d_smpl - pred_pelvis_smpl

                    # Absolute error (MPJPE)
                    error_smpl = (torch.sqrt(
                        ((pred_keypoints_3d_smpl - gt_keypoints_3d) ** 2).sum(dim=-1)) * visible_kpts).sum(
                        -1) / visible_kpts.sum(-1)

                    # Reconstuction_error
                    r_error_smpl = reconstruction_error(pred_keypoints_3d_smpl.cpu().numpy(),
                                    gt_keypoints_3d.cpu().numpy(),
                                    reduction=None, visible_kpts=visible_kpts.cpu().numpy())
                    loss_dict['MPJPE'] = error_smpl * 1000  # m to mm
                    loss_dict['r_error'] = (torch.tensor(r_error_smpl) * 1000).to(
                        pred_keypoints_3d_smpl.device)  # m to mm

                else:
                    loss_dict['MPJPE'] = torch.FloatTensor(1).fill_(100.).to(pred_joints.device)
                    loss_dict['r_error'] = torch.FloatTensor(1).fill_(100.).to(pred_joints.device)
        return loss_dict

    def compute_oks(self, pred_keypoints_2d, gt_keypoints_2d, gt_bboxes):
        """
        Compute OKS for predicted and GT keypoints.
        pred_keypoints_2d: BxKx2
        gt_keypoints_2d: BxKx3 (last dim for visibility)
        gt_bboxes: Bx4
        """
        
        sigmas = np.array([
            .89, .87, 1.07, 1.07, .87, .89, .62, .72, .79, .79, .72, .62, .26, .26,
            1.07, .79, .79, .26, .26, .26, .25, .25, .25, .25], dtype=np.float32) / 10.0        
        sigmas = torch.from_numpy(sigmas).to(pred_keypoints_2d.device).unsqueeze(0)
        vars = (sigmas * 2) ** 2
        # area 
        area = (gt_bboxes[..., 2] - gt_bboxes[..., 0] + 1) * (gt_bboxes[..., 3] - gt_bboxes[..., 1] + 1)
        area = area.unsqueeze(-1)
        
        # measure the per-keypoint distance if keypoints are visible
        dx = (pred_keypoints_2d[..., 0] - gt_keypoints_2d[..., 0]) * gt_keypoints_2d[..., -1]
        dy = (pred_keypoints_2d[..., 1] - gt_keypoints_2d[..., 1]) * gt_keypoints_2d[..., -1]
        
        e = (dx**2 + dy**2) / vars / (area + 1e-8) / 2
        # only compute keypoints that are visible
        invisible_mask = gt_keypoints_2d[..., -1] == 0
        e[invisible_mask] = float('Inf')
        visible_kpts_num = torch.sum(gt_keypoints_2d[..., -1]>0, dim=1).float()
        e = torch.sum(torch.exp(-e), dim=1) / (visible_kpts_num + 1e-8)
        return e.unsqueeze(-1)

    def confidence_loss(self, pred_conf, gt_conf):
        """
        Compute confidence loss for the predicted and GT confidences.
        The GT confidences are OKS between pred and gt 2D kpts.
        """
        return self.criterion_regr(pred_conf, gt_conf)

    def keypoint_loss(self, pred_keypoints_2d, gt_keypoints_2d):
        """
        Compute 2D reprojection loss on the keypoints.
        The confidence is binary and indicates whether the keypoints exist or not.
        The available keypoints are different for each dataset.
        """
        conf = gt_keypoints_2d[:, :, -1].unsqueeze(-1).clone()
        loss = (conf * self.criterion_keypoints(pred_keypoints_2d[:, -24:], gt_keypoints_2d[:, :, :-1]))
        return loss.mean(), loss.mean(dim=[1, 2]).detach()

    def keypoint_3d_loss(self, pred_keypoints_3d, gt_keypoints_3d):
        """
        Compute 3D keypoint loss for the examples that 3D keypoint annotations are available.
        The loss is weighted by the confidence
        """
        conf = gt_keypoints_3d[:, :, -1].unsqueeze(-1).clone()
        gt_keypoints_3d = gt_keypoints_3d[:, :, :-1].clone()
        pred_keypoints_3d = pred_keypoints_3d[..., -24:, :]
        if len(gt_keypoints_3d) > 0:
            gt_pelvis = (gt_keypoints_3d[:, 2, :] + gt_keypoints_3d[:, 3, :]) / 2
            gt_keypoints_3d = gt_keypoints_3d - gt_pelvis[:, None, :]
            pred_pelvis = (pred_keypoints_3d[:, 2, :] + pred_keypoints_3d[:, 3, :]) / 2
            pred_keypoints_3d = pred_keypoints_3d - pred_pelvis[:, None, :]
            if hasattr(self, 'criterion_3d_keypoints'):
                return (conf * self.criterion_3d_keypoints(pred_keypoints_3d, gt_keypoints_3d)).mean()
            else:
                return (conf * self.criterion_keypoints(pred_keypoints_3d, gt_keypoints_3d)).mean()
        else:
            return torch.tensor(0).float().cuda()
    
    def keypoint_3d_aligned_loss(self, pred_keypoints_3d, gt_keypoints_3d):
        """
        Compute 3D keypoint loss for the examples that 3D keypoint annotations are available.
        The loss is weighted by the confidence
        """
        conf = gt_keypoints_3d[:, :, -1].unsqueeze(-1).clone()
        gt_keypoints_3d = gt_keypoints_3d[:, :, :-1].clone()
        pred_keypoints_3d = pred_keypoints_3d[..., -24:, :]
        if len(gt_keypoints_3d) > 0:
            gt_pelvis = (gt_keypoints_3d[:, 2, :] + gt_keypoints_3d[:, 3, :]) / 2
            gt_keypoints_3d = gt_keypoints_3d - gt_pelvis[:, None, :]
            pred_pelvis = (pred_keypoints_3d[:, 2, :] + pred_keypoints_3d[:, 3, :]) / 2
            pred_keypoints_3d = pred_keypoints_3d - pred_pelvis[:, None, :]
            # Rigid align
            pred_keypoints_3d = compute_similarity_transform(pred_keypoints_3d, gt_keypoints_3d)
            if hasattr(self, 'criterion_3d_keypoints'):
                return (conf * self.criterion_3d_keypoints(pred_keypoints_3d, gt_keypoints_3d)).mean()
            else:
                return (conf * self.criterion_keypoints(pred_keypoints_3d, gt_keypoints_3d)).mean()
        else:
            return torch.tensor(0).float().cuda() 

    def shape_loss(self, pred_vertices, gt_vertices, has_smpl):
        """
        Compute per-vertex loss on the shape for the examples that SMPL annotations are available.
        """
        pred_vertices_with_shape = pred_vertices[has_smpl == 1]
        gt_vertices_with_shape = gt_vertices[has_smpl == 1]
        if len(gt_vertices_with_shape) > 0:
            return self.criterion_shape(pred_vertices_with_shape, gt_vertices_with_shape)
        else:
            return torch.tensor(0).float().cuda()

    def smpl_losses(self, pred_rotmat, pred_betas, gt_pose, gt_betas, has_smpl):
        """
        Compute SMPL parameter loss for the examples that SMPL annotations are available.
        """
        batch_size = pred_rotmat.shape[0]
        pred_rotmat_valid = pred_rotmat[has_smpl == 1].view(-1, 3, 3)
        gt_rotmat_valid = batch_rodrigues(gt_pose[has_smpl == 1].view(-1, 3))
        pred_betas_valid = pred_betas[has_smpl == 1]
        gt_betas_valid = gt_betas[has_smpl == 1]
        if len(pred_rotmat_valid) > 0:
            loss_regr_pose = self.criterion_regr(pred_rotmat_valid, gt_rotmat_valid)
            loss_regr_betas = self.criterion_regr(pred_betas_valid, gt_betas_valid)
        else:
            loss_regr_pose = torch.tensor(0).float().cuda()
            loss_regr_betas = torch.tensor(0).float().cuda()
        return loss_regr_pose, loss_regr_betas