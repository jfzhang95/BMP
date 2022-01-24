"""
Demo code

Example usage:

python3 tools/demo.py --config=configs/bmp/finetune.py --image_folder=/path/to/demo_images/ --output_folder=results_demo/ --ckpt ./work_dirs/finetune/latest.pth
"""
import os
os.environ['PYOPENGL_PLATFORM'] = 'osmesa'

import pyrender 
import OpenGL
import torch
torch.multiprocessing.set_sharing_strategy('file_system')
from torch import nn

import argparse
import os
import os.path as osp
import sys
import cv2
import numpy as np
import mmcv
from tqdm import tqdm
import trimesh

PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, PROJECT_PATH)

from mmcv import Config
from mmcv.runner import Runner

from mmcv.parallel import DataContainer as DC
from mmcv.parallel import MMDataParallel
from mmdet.apis.train import build_optimizer
from mmdet.models.utils.smpl.renderer import Renderer
from mmdet import __version__
from mmdet.models import build_detector
from mmdet.datasets.transforms import ImageTransform
from mmdet.datasets.utils import to_tensor
from mmdet.models.utils.smpl.smpl import SMPL, JointMapper

denormalize = lambda x: x.transpose([1, 2, 0]) * np.array([0.229, 0.224, 0.225])[None, None, :] + \
                        np.array([0.485, 0.456, 0.406])[None, None,]

# dataset settings
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

# Initialize SMPL model
openpose_joints = [24, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5, 8, 1, 4,
                   7, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34]
extra_joints = [8, 5, 45, 46, 4, 7, 21, 19, 17, 16, 18, 20, 47, 48, 49, 50, 51, 52, 53, 24, 26, 25, 28, 27]
joints = torch.tensor(openpose_joints + extra_joints, dtype=torch.int32)
joint_mapper = JointMapper(joints)
smpl_params = dict(model_path='data/smpl',
                   joint_mapper=joint_mapper,
                   body_pose_param='identity',
                   dtype=torch.float32,
                   vposer_ckpt=None,
                   gender='neutral')
smpl = SMPL(**smpl_params)

def renderer_bv(img_t, verts_t, trans_t, bboxes_t, focal_length, render):
    R_bv = torch.zeros(3, 3)
    R_bv[0, 0] = R_bv[2, 1] = 1
    R_bv[1, 2] = -1
    bbox_area = (bboxes_t[:, 2] - bboxes_t[:, 0] + 1) * (bboxes_t[:, 3] - bboxes_t[:, 1] + 1)
    area_mask = torch.tensor(bbox_area > bbox_area.max() * 0.05)
    verts_t, trans_t = verts_t[area_mask], trans_t[area_mask]
    verts_t = verts_t + trans_t.unsqueeze(1)
    verts_tr = torch.einsum('bij,kj->bik', verts_t, R_bv)
    verts_tfar = verts_tr  # verts_tr + trans_t.unsqueeze(1)
    p_min, p_max = verts_tfar.view(-1, 3).min(0)[0], verts_tfar.view(-1, 3).max(0)[0]
    p_center = 0.5 * (p_min + p_max)
    verts_center = (verts_tfar.view(-1, 3) - p_center).view(verts_t.shape[0], -1, 3)

    dis_min, dis_max = (verts_tfar.view(-1, 3) - p_center).min(0)[0], (
            verts_tfar.view(-1, 3) - p_center).max(0)[0]
    h, w = img_t.shape[-2:]
    ratio_max = abs(0.9 - 0.5)
    z_x = dis_max[0] * focal_length / (ratio_max * w) + torch.abs(dis_min[2])
    z_y = dis_max[1] * focal_length / (ratio_max * h) + torch.abs(dis_min[2])
    z_x_0 = (-dis_min[0]) * focal_length / (ratio_max * w) + torch.abs(
        dis_min[2])
    z_y_0 = (-dis_min[1]) * focal_length / (ratio_max * h) + torch.abs(
        dis_min[2])
    z = max(z_x, z_y, z_x_0, z_y_0)
    verts_right = verts_tfar - p_center + torch.tensor([0, 0, z])
    img_right = render([torch.ones_like(img_t)], [verts_right],
                       translation=[torch.zeros_like(trans_t)])
    return img_right[0]


def prepare_dump(pred_results, img, render, FOCAL_LENGTH):
    verts = pred_results['pred_vertices'] + pred_results['pred_translation'][:, None]
    pred_trans = pred_results['pred_translation'].clone().detach().cpu()
    pred_betas = pred_results['pred_betas'].clone().detach().cpu()
    pred_rotmat = pred_results['pred_rotmat'].clone().detach().cpu()
    pred_verts = pred_results['pred_vertices'].clone().detach().cpu()
    pred_kpts2d = pred_results['pred_kpts2d'].clone().detach().cpu().numpy()
    bboxes = pred_results['bboxes']
    img_bbox = img.copy()
    img_th = torch.tensor(img_bbox.transpose([2, 0, 1]))
    _, H, W = img_th.shape
    try:
        fv_rendered = render([img_th.clone()], [pred_verts], translation=[pred_trans])[0]
        bv_rendered = renderer_bv(img_th, pred_verts, pred_trans, bbox_results[0], FOCAL_LENGTH, render)
    except Exception as e:
        print(e)
        return None

    total_img = np.zeros((3 * H, W, 3))
    total_img[:H] += img
    total_img[H:2 * H] += fv_rendered.transpose([1, 2, 0])
    total_img[2 * H:] += bv_rendered.transpose([1, 2, 0])
    total_img = (total_img * 255).astype(np.uint8)
    return total_img

def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--config', help='train config file path')
    parser.add_argument('--image_folder', help='Path to folder with images')
    parser.add_argument('--output_folder', default='model_output', help='Path to save results')
    parser.add_argument('--ckpt', type=str, default='')
    parser.add_argument('--flip_aug', type=bool, default=True)
    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    if args.ckpt:
        cfg.resume_from = args.ckpt

    FOCAL_LENGTH = cfg.get('FOCAL_LENGTH', 1000)

    model = build_detector(
        cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)
    if cfg.checkpoint_config is not None:
        # save mmdet version, config file content and class names in
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            mmdet_version=__version__,
            config=cfg.text,
            CLASSES=('Human',))
    # add an attribute for visualization convenience
    model.CLASSES = ('Human',)

    model = MMDataParallel(model, device_ids=[0]).cuda()

    # build runner
    optimizer = build_optimizer(model, cfg.optimizer)

    runner = Runner(model, lambda x: x, optimizer, cfg.work_dir,
                    cfg.log_level)
    runner.resume(cfg.resume_from)
    model = runner.model
    model.eval()
    render = Renderer(focal_length=FOCAL_LENGTH)
    img_transform = ImageTransform(
            size_divisor=32, **img_norm_cfg)
    img_scale = cfg.common_val_cfg.img_scale
    
    with torch.no_grad():
        folder_name = args.image_folder
        output_folder = args.output_folder
        os.makedirs(output_folder, exist_ok=True)
        images = os.listdir(folder_name)
        for image in tqdm(images):
            file_name = osp.join(folder_name, image)
            img = cv2.imread(file_name)
            ori_shape = img.shape

            if args.flip_aug:
                # Add test flip augmentation
                img_ori, img_shape_ori, pad_shape_ori, scale_factor_ori = \
                        img_transform(img, img_scale)
                img_flip, img_shape_flip, pad_shape_flip, scale_factor_flip = \
                        img_transform(img, img_scale, flip=True)
                img_ori = img_ori.transpose(2, 0, 1)
                img_flip = img_flip.transpose(2, 0, 1)

                # Force padding for the issue of multi-GPU training
                padded_img_ori = np.zeros((img_ori.shape[0], img_scale[1], img_scale[0]), dtype=img.dtype)
                padded_img_flip = np.zeros((img_flip.shape[0], img_scale[1], img_scale[0]), dtype=img.dtype)
                w_left = img_ori.shape[-1] // 2
                w_right = img_ori.shape[-1] - w_left
                w_center = img_scale[0] // 2
                h_top = img_ori.shape[-2] // 2
                h_bottom = img_ori.shape[-2] - h_top
                h_center = img_scale[1] // 2
                padded_img_ori[:, h_center-h_top: h_center+h_bottom, w_center-w_left: w_center+w_right] = img_ori
                padded_img_flip[:, h_center-h_top: h_center+h_bottom, w_center-w_left: w_center+w_right] = img_flip
                img_ori = padded_img_ori.transpose(1, 2, 0)
                img_flip = padded_img_flip.transpose(1, 2, 0)
                img_ori = mmcv.imnormalize(img_ori, **img_norm_cfg).transpose(2, 0, 1).astype(np.float32)
                img_flip = mmcv.imnormalize(img_flip, **img_norm_cfg).transpose(2, 0, 1).astype(np.float32)
                imgs = np.concatenate((img_ori[None, ...], img_flip[None, ...]), axis=0)
                data_batch = dict(
                    img=DC([to_tensor(imgs)], stack=True),
                    img_meta=DC([
                        {'img_shape':img_shape_ori, 'scale_factor':scale_factor_ori, 'flip':False, 'ori_shape':ori_shape},
                        {'img_shape':img_shape_flip, 'scale_factor':scale_factor_flip, 'flip':True, 'ori_shape':ori_shape}], cpu_only=True),
                    )                
            else:
                img, img_shape, pad_shape, scale_factor = img_transform(img, img_scale, flip=False)
                img = img.transpose(2, 0, 1)
                
                # Force padding for the issue of multi-GPU training
                w_left = img.shape[-1] // 2
                w_right = img.shape[-1] - w_left
                w_center = img_scale[0] // 2
                h_top = img.shape[-2] // 2
                h_bottom = img.shape[-2] - h_top
                h_center = img_scale[1] // 2
                padded_img = np.zeros((img.shape[0], img_scale[1], img_scale[0]), dtype=img.dtype)
                padded_img[:, h_center-h_top: h_center+h_bottom, w_center-w_left: w_center+w_right] = img
                img = padded_img.transpose(1, 2, 0)
                img = mmcv.imnormalize(img, **img_norm_cfg).transpose(2, 0, 1).astype(np.float32)

                data_batch = dict(
                    img=DC([to_tensor(img[None, ...])], stack=True),
                    img_meta=DC([{'img_shape':img_shape, 'scale_factor':scale_factor, 'flip':False, 'ori_shape':ori_shape}], cpu_only=True),
                )
            pred_results = model(**data_batch, return_loss=False)
            try:
                pred_results['bbox_results'] = pred_results['bboxes']
                if pred_results is not None:
                    if args.flip_aug:
                        img = denormalize(imgs[0])
                    else:
                        img = denormalize(img)
                    img_viz = prepare_dump(pred_results, img, render, FOCAL_LENGTH)
                    cv2.imwrite(f'{file_name.replace(folder_name, output_folder)}.output.jpg', img_viz[:, :, ::-1])
            except Exception as e:
                tqdm.write(f"Fail on {file_name}")
                tqdm.write(str(e))


if __name__ == '__main__':
    main()
