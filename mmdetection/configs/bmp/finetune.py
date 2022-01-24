# model settings
from mmdetection.mmdet.core.utils.smpl_tensorboard import SMPLBoard
import os.path as osp
from mmdetection.mmdet.core.utils.radam import RAdam
from mmdetection.mmdet.core.utils.lr_hooks import SequenceLrUpdaterHook, PowerLrUpdaterHook
import math

WITH_NR = False
FOCAL_LENGTH = 1000
model = dict(
    type='BMP',
    pretrained='modelzoo://resnet50',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),  # C2, C3, C4, C5
        frozen_stages=1,
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=0,
        num_outs=5),
    smpl_head=dict(
        type='BMPHead',
        num_classes=2,
        in_channels=256,
        stacked_convs=7,
        smpl_feat_channels=256,
        strides=[8, 8, 16, 32, 32],
        scale_ranges=((1, 64), (32, 128), (64, 256), (128, 512), (256, 2048)),
        sigma=0.2,
        num_grids=[40, 36, 24, 16, 12],
        norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
        use_1x1=True,       # whether use 1x1 conv, set False to use 3x3 conv
        pred_conf=True,     # whether predict pose confidence
        use_centernet_loss=False,  # whether use centernet style focal loss
        loss_cate=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_cfg=dict(
            type='BMPLoss', eval_pose=True, normalize_kpts=False,
            adversarial_cfg=True,
            FOCAL_LENGTH=FOCAL_LENGTH,
            img_size=(832, 512)),
    )
)
re_weight = {'loss_cate': 2.0, 'loss_conf': 1.0, 'loss_disc': 1 / 60., 
             'adv_loss_fake': 1 / 60., 'adv_loss_real': 1 / 60.}
# model training and testing settings
train_cfg = dict()
test_cfg = dict(
    nms_pre=500,
    score_thr=0.1,
    pose_thr=0.5,
    nms_thr=0.5,
    oks_thr=0.3,
    max_per_img=100
)
# dataset settings
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
square_bbox = False
data_path = '/path/to/datasets/'
common_train_cfg = dict(
    img_scale=(832, 512),
    img_norm_cfg=img_norm_cfg,
    size_divisor=32,
    flip_ratio=0.5,
    rot_factor=0,
    occ_aug=False,
    syn_occ=True,
    noise_factor=1e-3,  # To avoid color jitter.
    with_mask=False,
    with_crowd=False,
    with_label=True,
    with_kpts2d=True,
    with_kpts3d=True,
    with_pose=True,
    with_shape=True,
    with_trans=True,
    val_every=-1,
    square_bbox=square_bbox,
    mosh_path=data_path+'h36m/extras/mosh_annot.npz',
    voc_path=data_path+'VOCdevkit/VOC2012/',
)
common_val_cfg = dict(
    img_scale=(832, 512),
    img_norm_cfg=img_norm_cfg,
    size_divisor=32,
    with_mask=False,
    with_crowd=False,
    with_label=True,
    with_kpts2d=True,
    with_kpts3d=True,
    with_pose=True,
    with_shape=True,
    with_trans=True,
    max_samples=8,
    square_bbox=square_bbox,
    mosh_path=data_path+'h36m/extras/mosh_annot.npz',
    voc_path=data_path+'VOCdevkit/VOC2012/',
    with_nr=WITH_NR,
    use_poly=True,
)

h36m_dataset_type = 'H36MDataset'
coco_dataset_type = 'COCOKeypoints'
common_dataset = 'CommonDataset'

h36m_data_root = data_path+'h36m/'
coco_data_root = data_path+'coco/'
pose_track_root = data_path+'posetrack/'
mpii_root = data_path+'mpii/'
mpi_inf_3dhp_root = data_path+'mpi_inf_3dhp/'
panoptic_root = data_path+'panoptic/'
muco_root = data_path+'MuCo/'
lspet_root = data_path+'lspet/'

IGNORE_PSEUDO_GT = True

datasets = [
    dict(
        train=dict(
            type=common_dataset,
            ann_file=h36m_data_root + 'extras/rcnn/h36m_train.pkl',
            img_prefix=h36m_data_root,
            sample_weight=0.6,
            **common_train_cfg
        ),
        val=dict(
            type=common_dataset,
            ann_file=h36m_data_root + 'extras/rcnn/h36m_val.pkl',
            img_prefix=h36m_data_root,
            sample_weight=0.6,
            **common_val_cfg
        ),
    ),
    dict(
        train=dict(
            type=common_dataset,
            ann_file=muco_root + 'rcnn/train.pkl',
            img_prefix=muco_root,
            sample_weight=0.3,
            **common_train_cfg
        ),
    ),
    dict(
        train=dict(
            type=common_dataset,
            ann_file=coco_data_root + 'annotations/train_densepose_2014_depth_nocrowd.pkl',
            img_prefix=coco_data_root + 'train2014/',
            sample_weight=0.3,
            ignore_smpl=IGNORE_PSEUDO_GT,
            ignore_3d=IGNORE_PSEUDO_GT,
            **common_train_cfg
        ),
        val=dict(
            type=common_dataset,
            ann_file=coco_data_root + 'annotations/val_densepose_2014_depth_nocrowd.pkl',
            img_prefix=coco_data_root + 'val2014/',
            sample_weight=0.3,
            ignore_smpl=IGNORE_PSEUDO_GT,
            ignore_3d=IGNORE_PSEUDO_GT,
            **common_val_cfg
        ),
    ),
    dict(
        train=dict(
            type=common_dataset,
            ann_file=pose_track_root + 'rcnn/train.pkl',
            img_prefix=pose_track_root,
            sample_weight=0.3,
            **common_train_cfg
        ),
        val=dict(
            type=common_dataset,
            ann_file=pose_track_root + 'rcnn/val.pkl',
            img_prefix=pose_track_root,
            sample_weight=0.3,
            **common_val_cfg
        ),
    ),
    dict(
        train=dict(
            type=common_dataset,
            ann_file=mpii_root + 'rcnn/train.pkl',
            img_prefix=mpii_root + 'images/',
            sample_weight=0.3,
            ignore_smpl=IGNORE_PSEUDO_GT,
            ignore_3d=IGNORE_PSEUDO_GT,
            **common_train_cfg
        ),
        val=dict(
            type=common_dataset,
            ann_file=mpii_root + 'rcnn/val.pkl',
            img_prefix=mpii_root + 'images/',
            sample_weight=0.3,
            ignore_smpl=IGNORE_PSEUDO_GT,
            ignore_3d=IGNORE_PSEUDO_GT,
            **common_val_cfg
        ),
    ),
    dict(
        train=dict(
            type=common_dataset,
            ann_file=lspet_root + 'rcnn/train.pkl',
            img_prefix=lspet_root + 'images/',
            sample_weight=0.3,
            ignore_smpl=IGNORE_PSEUDO_GT,
            ignore_3d=IGNORE_PSEUDO_GT,
            **common_train_cfg
        ),
    ),
    dict(
        train=dict(
            type=common_dataset,
            ann_file=mpi_inf_3dhp_root + 'rcnn/train.pkl',
            img_prefix=mpi_inf_3dhp_root,
            sample_weight=0.1,
            **common_train_cfg
        ),
        val=dict(
            type=common_dataset,
            ann_file=mpi_inf_3dhp_root + 'rcnn/val.pkl',
            img_prefix=mpi_inf_3dhp_root,
            sample_weight=0.1,
            ignore_3d=True,
            **common_val_cfg
        ),
    ),
]
data = dict(
    imgs_per_gpu=32,
    workers_per_gpu=8,
    train=common_train_cfg,
    val=common_val_cfg,
)

# optimizer
optimizer = dict(type=RAdam, lr=1e-4, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
adv_optimizer = dict(type=RAdam, lr=1e-4, weight_decay=0.0001)
adv_optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = SequenceLrUpdaterHook(
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    seq=[1e-4]
)

checkpoint_config = dict(interval=1)
# yapf:disable
# runtime settings
total_epochs = 50
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/finetune'
load_from = None
resume_from = osp.join(work_dir, 'latest.pth')
workflow = [('train', 1), ('val', 1)]

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
    ])
evaluation = dict(interval=10)
# yapf:enable
fuse = True
time_limit = 1 * 3000000  # In sceonds
log_grad = True
