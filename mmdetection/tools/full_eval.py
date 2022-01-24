import torch
from torch import nn
torch.multiprocessing.set_sharing_strategy('file_system')

import argparse
import os
import os.path as osp
import pickle
from tqdm import tqdm
import sys
import numpy as np
import matplotlib.pyplot as plt
import time
from mmcv import Config
from mmcv.runner import Runner

"""
python3 tools/full_eval.py configs/bmp/finetune.py ultimatum --ckpt ./work_dirs/finetune/latest.pth
"""

PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, PROJECT_PATH)
from mmdetection.mmdet.core.utils import AverageMeter
from mmdetection.mmdet.datasets import build_dataloader_fuse
from mmdetection.mmdet.apis.adv_runner import AdvRunner
from mmdetection.mmdet.apis.train import build_optimizer

from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmdet import __version__
from mmdet.datasets import get_dataset
from mmdet.apis import (train_detector, init_dist, get_root_logger,
                        set_random_seed, train_smpl_detector_fuse)
from mmdet.models import build_detector
from mmdet.datasets.concat_dataset import ConcatDataset
from mmdet.models.smpl_heads.smpl_common import Discriminator
from mmdet.models.utils.smpl_utils import batch_rodrigues, J24_TO_J14, H36M_TO_J14
from mmdet.models.utils.pose_utils import reconstruction_error
from mmdet.core.utils.eval_utils import H36MEvalHandler, EvalHandler, PanopticEvalHandler, \
    MuPoTSEvalHandler, PW3DEvalHandler, PoseTrackEvalHandler
from mmdet.models.utils.smpl.smpl import SMPL, JointMapper

import neural_renderer as nr

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
denormalize = lambda x: x.transpose([1, 2, 0]) * np.array([0.229, 0.224, 0.225])[None, None, :] + \
                        np.array([0.485, 0.456, 0.406])[None, None,]

# dataset settings
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

data_path = '/path/to/datasets/'
TEST_MODE = True
eval_dataset_mapper = dict(
    cropped_h36m=dict(
        type='CommonDataset',
        ann_file=data_path+'rcnn-pretrain/' + 'h36m/' + 'val.pkl',
        img_prefix=data_path+'rcnn-pretrain/' + 'h36m/' + 'images/',
        test_mode=TEST_MODE,
    ),
    full_h36m=dict(
        type='CommonDataset',
        ann_file=data_path+'h36m/' + 'extras/rcnn/h36m_val.pkl',
        img_prefix=data_path+'h36m/',
        test_mode=TEST_MODE,
    ),
    ultimatum=dict(
        type='CommonDataset',
        ann_file=data_path+'panoptic/' + 'annotations/160422_ultimatum1.pkl',
        img_prefix=data_path+'panoptic/',
        test_mode=TEST_MODE,
    ),
    haggling=dict(
        type='CommonDataset',
        ann_file=data_path+'panoptic/' + 'annotations/160422_haggling1.pkl',
        img_prefix=data_path+'panoptic/',
        test_mode=TEST_MODE,
    ),
    pizza=dict(
        type='CommonDataset',
        ann_file=data_path+'panoptic/' + 'annotations/160906_pizza1.pkl',
        img_prefix=data_path+'panoptic/',
        test_mode=TEST_MODE,
    ),
    mafia=dict(
        type='CommonDataset',
        ann_file=data_path+'panoptic/' + 'annotations/160422_mafia2.pkl',
        img_prefix=data_path+'panoptic/',
        test_mode=TEST_MODE,
    ),
    mupots=dict(
        type='CommonDataset',
        ann_file=data_path+'MuPoTS3D/' + 'rcnn/all_sorted.pkl',
        img_prefix=data_path+'MuPoTS3D/',
        test_mode=TEST_MODE,
    ),
    pw3d=dict(
        type='CommonDataset',
        ann_file=data_path+'3DPW/' + 'rcnn/test.pkl',
        img_prefix=data_path+'3DPW/',
        test_mode=TEST_MODE,
    ),
    posetrack=dict(
        type='CommonDataset',
        ann_file=data_path+'posetrack/' + 'rcnn/val.pkl',
        img_prefix=data_path+'posetrack/',
        test_mode=TEST_MODE,
    ),
)

eval_handler_mapper = dict(
    cropped_h36m=H36MEvalHandler,
    full_h36m=H36MEvalHandler,
    ultimatum=PanopticEvalHandler,
    haggling=PanopticEvalHandler,
    pizza=PanopticEvalHandler,
    mafia=PanopticEvalHandler,
    mupots=MuPoTSEvalHandler,
    pw3d=PW3DEvalHandler,
    posetrack=PoseTrackEvalHandler,
)

stable_list = ['mupots']


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('dataset', help='for which dataset will be evaluated')
    parser.add_argument('--work_dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume_from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--validate',
        action='store_true',
        help='whether to evaluate the checkpoint during training')
    parser.add_argument(
        '--gpus',
        type=int,
        default=1,
        help='number of gpus to use '
             '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--create_dummy', action='store_true',
                        help='Create a dummy checkpoint for recursive training on clusters')
    parser.add_argument('--debug', action='store_true',
                        help='debug mode')
    parser.add_argument('--imgs_per_gpu', type=int, default=-1)
    parser.add_argument('--ckpt', type=str, default='')
    parser.add_argument('--use_gt', action='store_true')
    parser.add_argument('--viz_dir', type=str, default='')
    parser.add_argument('--dump_pkl', action='store_true')
    parser.add_argument('--paper_dir', type=str, default='')
    parser.add_argument('--nms_thr', type=float, default=None)
    parser.add_argument('--flip_aug', type=bool, default=False)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    # update configs according to CLI args
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    cfg.gpus = args.gpus

    if args.ckpt:
        cfg.resume_from = args.ckpt

    if args.imgs_per_gpu > 0:
        cfg.data.imgs_per_gpu = args.imgs_per_gpu
    if args.nms_thr:
        cfg.test_cfg.rcnn.nms.iou_thr = args.nms_thr

    FOCAL_LENGTH = cfg.get('FOCAL_LENGTH', 1000)

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # init logger before other steps
    logger = get_root_logger(cfg.log_level)
    logger.info('Distributed training: {}'.format(distributed))

    # set random seeds
    if args.seed is not None:
        logger.info('Set random seed to {}'.format(args.seed))
        set_random_seed(args.seed)

    model = build_detector(
        cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)
    train_dataset = get_dataset(cfg.datasets[0].train)
    if cfg.checkpoint_config is not None:
        # save mmdet version, config file content and class names in
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            mmdet_version=__version__,
            config=cfg.text,
            CLASSES=train_dataset.CLASSES)
    # add an attribute for visualization convenience
    model.CLASSES = train_dataset.CLASSES

    model = MMDataParallel(model, device_ids=range(cfg.gpus)).cuda()

    # build runner
    optimizer = build_optimizer(model, cfg.optimizer)

    runner = Runner(model, lambda x: x, optimizer, cfg.work_dir,
                    cfg.log_level)
    runner.resume(cfg.resume_from)
    model = runner.model
    model.eval()

    dataset_cfg = eval_dataset_mapper[args.dataset]
    if args.dataset == 'pw3d':
        cfg.common_val_cfg.update({'eval_pw3d': True})
    if args.flip_aug:
        cfg.common_val_cfg.update({'flip_ratio': 1})
    dataset_cfg.update(cfg.common_val_cfg)
    dataset_cfg.pop('max_samples')
    dataset = get_dataset(dataset_cfg)
    shuffle = False if args.dataset in stable_list else True
    data_loader = build_dataloader_fuse(
        dataset,
        1,
        0,
        cfg.gpus,
        dist=False,
        shuffle=shuffle,
        drop_last=False,
    )

    dump_dir = os.path.join(cfg.work_dir, f'eval_{args.dataset}')
    os.makedirs(dump_dir, exist_ok=True)
    if args.viz_dir:
        os.makedirs(args.viz_dir, exist_ok=True)
    eval_handler = eval_handler_mapper[args.dataset](writer=tqdm.write, viz_dir=args.viz_dir,
                                                     FOCAL_LENGTH=FOCAL_LENGTH,
                                                     work_dir=cfg.work_dir)

    with torch.no_grad():
        for i, data_batch in enumerate(tqdm(data_loader)):
            data_batch['img'].data[0] = data_batch['img'].data[0][0]
            data_batch['img_meta'].data[0] = data_batch['img_meta'].data[0][0][0]
            file_name = data_batch['img_meta'].data[0]['file_name']
            
            try:
                pred_results = model(**data_batch, return_loss=False)
                if args.paper_dir:
                    os.makedirs(args.paper_dir, exist_ok=True)
                    img = denormalize(data_batch['img'].data[0][0].numpy())
                    verts = pred_results['pred_vertices'] + pred_results['pred_translation']
                    dump_folder = osp.join(args.paper_dir, file_name)
                    os.makedirs(dump_folder, exist_ok=True)
                    plt.imsave(osp.join(dump_folder, 'img.png'), img)
                    for obj_i, vert in enumerate(verts):
                        nr.save_obj(osp.join(dump_folder, f'{obj_i}.obj'), vert,
                                    torch.tensor(smpl.faces.astype(np.int64)))
                save_pack = eval_handler(data_batch, pred_results, use_gt=args.use_gt)
                if args.dump_pkl:
                    with open(osp.join(dump_dir, f"{save_pack['file_name']}.pkl"), 'wb') as f:
                        pickle.dump(save_pack, f)
            except Exception as e:
                tqdm.write(f"Fail on {file_name}")
                tqdm.write(str(e))
    eval_handler.finalize()


if __name__ == '__main__':
    main()
