from pycocotools.coco import COCO
import numpy as np 
import os
import os.path as osp
import math
from tqdm import tqdm
import argparse
import json
import pickle
from utils import gather_per_image

parser = argparse.ArgumentParser(description='Preprocess MuCo')
parser.add_argument('dataset_path')
parser.add_argument('out_path')


def muco_extract(dataset_path, out_path):
    # conver joints to global order, 17 joints
    joints_idx = [13, 15, 8, 7, 6, 9, 10, 11, 2, 1, 0, 3, 4, 5, 14, 16, 18]

    # structs we need
    imgnames_, widths_, heights_, \
    bboxes_, parts_, Ss_, roots_  = [], [], [], [], [], [], []

    # json annotation file
    json_path = osp.join(dataset_path,
                             f'MuCo-3DHP.json')
    json_data = json.load(open(json_path, 'r'))

    imgs = {}
    for img in json_data['images']:
        imgs[img['id']] = img
    print()

    for annot in tqdm(json_data['annotations']):
        # keypoints processing
        keypoints = np.array(annot['keypoints_img'])
        keypoints_vis = np.expand_dims(np.array(annot['keypoints_vis'], dtype=np.int), -1)
        keypoints = np.concatenate((keypoints, keypoints_vis), -1)

        # keypoints
        part = np.zeros([24,3])
        part[joints_idx] = keypoints[:-4]

        S24 = np.zeros([24,4])
        keypoints_3d = np.array(annot['keypoints_cam'])
        root_3d = keypoints_3d[14:15].copy()
        keypoints_3d -= root_3d
        keypoints_3d /= 1000.
        keypoints_3d = np.concatenate((keypoints_3d, keypoints_vis), -1)
        S24[joints_idx] = keypoints_3d[:-4]

        # image name
        image_id = annot['image_id']
        img_name = str(imgs[image_id]['file_name'])
        height, width = imgs[image_id]['height'], imgs[image_id]['width']
        # img_name_full = osp.join(args.dataset_path, img_name)

        # bbox
        bbox = annot['bbox']
        x, y, w, h = bbox
        x1 = np.max((0, x))
        y1 = np.max((0, y))
        x2 = np.min((width - 1, x1 + np.max((0, w - 1))))
        y2 = np.min((height - 1, y1 + np.max((0, h - 1))))
        bbox = np.array([x1, y1, x2, y2])

        # store data
        imgnames_.append(img_name)
        bboxes_.append(bbox)
        parts_.append(part)
        Ss_.append(S24)
        widths_.append(width)
        heights_.append(height)
        roots_.append(root_3d)
    
    imgnames = np.array(imgnames_)
    bboxes = np.array(bboxes_)
    kpts2d = np.array(parts_)
    kpts3d = np.array(Ss_)
    widths = np.array(widths_)
    heights = np.array(heights_)
    roots = np.array(roots_)
    data = gather_per_image(dict(filename=imgnames, bboxes=bboxes, kpts2d=kpts2d, kpts3d=kpts3d,
                                 width=widths, height=heights, roots=roots), img_dir=dataset_path)

    # store the data struct
    if not osp.isdir(out_path):
        os.makedirs(out_path)
    out_file = osp.join(out_path, 'train_full.pkl')

    with open(out_file, 'wb') as f:
        pickle.dump(data, f)


if __name__ == "__main__":
    args = parser.parse_args()
    muco_extract(args.dataset_path, args.out_path)