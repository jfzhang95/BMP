import numpy as np
from tqdm import trange
import os
import cv2
from PIL import Image
import argparse

def gather_per_image(x, img_dir):
    filenames = x['filename']
    indices = np.argsort(filenames)
    x = {k: v[indices] for k,v in x.items()}
    filenames = x['filename']

    image_boxes = [[]]
    old_name = str(filenames[0])
    img_count = 0
    for i in range(len(filenames)):
        name = str(filenames[i])
        if name != old_name:
            img_count += 1
            image_boxes.append([])
        old_name = name
        image_boxes[img_count].append(i)

    data = [{} for _ in range(len(image_boxes))]
    img_shapes = []
    for i in trange(len(image_boxes)):
        for key in x.keys():
            data[i][key] = x[key][image_boxes[i]]
        height = data[i]['height'][0]
        width = data[i]['width'][0]            
        data[i]['filename'] = data[i]['filename'][0]
        data[i]['height'] = data[i]['height'][0]
        data[i]['width'] = data[i]['width'][0]
        data[i]['bboxes'][:, :2] = np.maximum(data[i]['bboxes'][:, :2], np.zeros_like(data[i]['bboxes'][:, :2]))
        data[i]['bboxes'][:, 2] = np.minimum(data[i]['bboxes'][:, 2], width * np.ones_like(data[i]['bboxes'][:, 2]))
        data[i]['bboxes'][:, 3] = np.minimum(data[i]['bboxes'][:, 3], height * np.ones_like(data[i]['bboxes'][:, 3]))
    return data


def vectorize_distance(a, b):
    """
    Calculate euclid distance on each row of a and b
    :param a: Nx... np.array
    :param b: Mx... np.array
    :return: MxN np.array representing correspond distance
    """
    N = a.shape[0]
    a = a.reshape ( N, -1 )
    M = b.shape[0]
    b = b.reshape ( M, -1 )
    a2 = np.tile ( np.sum ( a ** 2, axis=1 ).reshape ( -1, 1 ), (1, M) )
    b2 = np.tile ( np.sum ( b ** 2, axis=1 ), (N, 1) )
    dist = a2 + b2 - 2 * (a @ b.T)
    return np.sqrt(dist)