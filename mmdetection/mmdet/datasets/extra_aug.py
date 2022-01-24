import mmcv
import numpy as np
from numpy import random
import cv2
import os
import xml
import PIL

from mmdet.core.evaluation.bbox_overlaps import bbox_overlaps


class OcclusionAugmentation(object):
    """Randomly drop the appearance information of a keypoint.
    1. choose a keypoint from the keypoints set
    2. set the shape of the neighboor area as circle 
    3. radius is randomly chosen within the range of [0.1*w, 0.2*w] (w denotes person size)
    4. randomly permute the circle center
    """

    def __init__(self):
        pass

    def __call__(self, img, bboxes, kpts2d):
        im_h, im_w, _ = img.shape
        num_person = len(bboxes)
        count = np.random.randint(1, num_person*2)
        person_indices = [i for i in range(num_person)]
        for _ in range(count):
            if np.random.uniform() > 0.5:
                n = random.choice(person_indices)
                kpts = kpts2d[n]
                visible_kpts = kpts[kpts[..., -1]>0.2]
                if len(visible_kpts) < 12:
                    continue
                bbox = bboxes[n]
                x1, y1, x2, y2 = bbox
                h, w = y2-y1, x2-x1
                p_size = np.sqrt(np.abs(h*w))

                # choose random kpt
                x, y = visible_kpts[np.random.choice(range(len(visible_kpts)), 1)[0]][:2]

                # shift center
                delta_x = (np.random.randn() * 0.1) * p_size
                delta_y = (np.random.randn() * 0.1) * p_size
                x = int(np.clip(x+delta_x, 0, im_w))
                y = int(np.clip(y+delta_y, 0, im_h))

                # randomly sample r
                r = int((np.random.rand() * 0.1 + 0.1) * p_size)
                
                # drop appearance information
                mask = np.zeros(img.shape[:2], dtype=np.uint8)
                mask = cv2.circle(mask, (x, y), r, (255, 255, 255), -1)
                img = cv2.add(img, np.zeros(np.shape(img), dtype=np.uint8), mask=~mask)
            else:
                continue
        return img


class SyntheticOcclusion(object):
    def __init__(self, pascal_voc_root_path):
        self.occluder = self.load_occluders(pascal_voc_root_path)

    def __call__(self, img, bboxes, kpts2d):
        result = img.copy()
        im_h, im_w, _ = img.shape
        width_height = np.asarray([im_w, im_h])
        im_scale_factor = min(width_height) / 256
        count = np.random.randint(1, 8)

        num_person = len(bboxes)
        person_indices = [i for i in range(num_person)]

        for _ in range(count):
            if np.random.uniform() > 0.5:  
                # occluded in person
                occluder = random.choice(self.occluder)
                n = random.choice(person_indices)
                kpts = kpts2d[n]
                visible_kpts = kpts[kpts[..., -1]>0.2]
                if len(visible_kpts) < 12:
                    continue
                bbox = bboxes[n]
                x1, y1, x2, y2 = bbox
                h, w = y2-y1+1, x2-x1+1
                p_size = np.sqrt(np.abs(h*w))
                
                # choose random kpt
                x, y = visible_kpts[np.random.choice(range(len(visible_kpts)), 1)[0]][:2]

                # shift center
                delta_x = (np.random.randn() * 0.1) * p_size
                delta_y = (np.random.randn() * 0.1) * p_size
                x = int(np.clip(x+delta_x, 0, im_w))
                y = int(np.clip(y+delta_y, 0, im_h))
                center = np.array([x, y])

                obj_scale_factor = p_size / 256
                random_scale_factor = np.random.uniform(0.2, 1.0)
                scale_factor = random_scale_factor * im_scale_factor + 1e-8
                occluder = resize_by_factor(occluder, scale_factor)
                paste_over(im_src=occluder, im_dst=result, center=center)
            else:
                continue    
        return result
    
    @staticmethod
    def load_occluders(pascal_voc_root_path):
        occluders = []
        structuring_element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 8))
        
        annotation_paths = list_filepaths(os.path.join(pascal_voc_root_path, 'Annotations'))
        for annotation_path in annotation_paths:
            xml_root = xml.etree.ElementTree.parse(annotation_path).getroot()
            is_segmented = (xml_root.find('segmented').text != '0')

            if not is_segmented:
                continue

            boxes = []
            for i_obj, obj in enumerate(xml_root.findall('object')):
                is_person = (obj.find('name').text == 'person')
                is_difficult = (obj.find('difficult').text != '0')
                is_truncated = (obj.find('truncated').text != '0')
                if not is_person and not is_difficult and not is_truncated:
                    bndbox = obj.find('bndbox')
                    box = [int(bndbox.find(s).text) for s in ['xmin', 'ymin', 'xmax', 'ymax']]
                    boxes.append((i_obj, box))

            if not boxes:
                continue

            im_filename = xml_root.find('filename').text
            seg_filename = im_filename.replace('jpg', 'png')

            im_path = os.path.join(pascal_voc_root_path, 'JPEGImages', im_filename)
            seg_path = os.path.join(pascal_voc_root_path,'SegmentationObject', seg_filename)

            im = np.asarray(PIL.Image.open(im_path))
            labels = np.asarray(PIL.Image.open(seg_path))

            for i_obj, (xmin, ymin, xmax, ymax) in boxes:
                object_mask = (labels[ymin:ymax, xmin:xmax] == i_obj + 1).astype(np.uint8)*255
                object_image = im[ymin:ymax, xmin:xmax]
                if cv2.countNonZero(object_mask) < 500:
                    # Ignore small objects
                    continue

                # Reduce the opacity of the mask along the border for smoother blending
                eroded = cv2.erode(object_mask, structuring_element)
                object_mask[eroded < object_mask] = 192
                object_with_mask = np.concatenate([object_image, object_mask[..., np.newaxis]], axis=-1)
                
                # Downscale for efficiency
                object_with_mask = resize_by_factor(object_with_mask, 0.5)
                occluders.append(object_with_mask)

        return occluders


class PhotoMetricDistortion(object):

    def __init__(self,
                 brightness_delta=32,
                 contrast_range=(0.5, 1.5),
                 saturation_range=(0.5, 1.5),
                 hue_delta=18):
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta

    def __call__(self, img, boxes, labels):
        # random brightness
        if random.randint(2):
            delta = random.uniform(-self.brightness_delta,
                                   self.brightness_delta)
            img += delta

        # mode == 0 --> do random contrast first
        # mode == 1 --> do random contrast last
        mode = random.randint(2)
        if mode == 1:
            if random.randint(2):
                alpha = random.uniform(self.contrast_lower,
                                       self.contrast_upper)
                img *= alpha

        # convert color from BGR to HSV
        img = mmcv.bgr2hsv(img)

        # random saturation
        if random.randint(2):
            img[..., 1] *= random.uniform(self.saturation_lower,
                                          self.saturation_upper)

        # random hue
        if random.randint(2):
            img[..., 0] += random.uniform(-self.hue_delta, self.hue_delta)
            img[..., 0][img[..., 0] > 360] -= 360
            img[..., 0][img[..., 0] < 0] += 360

        # convert color from HSV to BGR
        img = mmcv.hsv2bgr(img)

        # random contrast
        if mode == 0:
            if random.randint(2):
                alpha = random.uniform(self.contrast_lower,
                                       self.contrast_upper)
                img *= alpha

        # randomly swap channels
        if random.randint(2):
            img = img[..., random.permutation(3)]

        return img, boxes, labels


class Expand(object):

    def __init__(self, mean=(0, 0, 0), to_rgb=True, ratio_range=(1, 4)):
        if to_rgb:
            self.mean = mean[::-1]
        else:
            self.mean = mean
        self.min_ratio, self.max_ratio = ratio_range

    def __call__(self, img, boxes, labels):
        if random.randint(2):
            return img, boxes, labels

        h, w, c = img.shape
        ratio = random.uniform(self.min_ratio, self.max_ratio)
        expand_img = np.full((int(h * ratio), int(w * ratio), c),
                             self.mean).astype(img.dtype)
        left = int(random.uniform(0, w * ratio - w))
        top = int(random.uniform(0, h * ratio - h))
        expand_img[top:top + h, left:left + w] = img
        img = expand_img
        boxes += np.tile((left, top), 2)
        return img, boxes, labels


class RandomCrop(object):

    def __init__(self, min_ious=(0.1, 0.3, 0.5, 0.7, 0.9), min_crop_size=0.3):
        # 1: return ori img
        self.sample_mode = (1, *min_ious, 0)
        self.min_crop_size = min_crop_size

    def __call__(self, img, boxes, labels):
        h, w, c = img.shape
        while True:
            mode = random.choice(self.sample_mode)
            if mode == 1:
                return img, boxes, labels

            min_iou = mode
            for i in range(50):
                new_w = random.uniform(self.min_crop_size * w, w)
                new_h = random.uniform(self.min_crop_size * h, h)

                # h / w in [0.5, 2]
                if new_h / new_w < 0.5 or new_h / new_w > 2:
                    continue

                left = random.uniform(w - new_w)
                top = random.uniform(h - new_h)

                patch = np.array((int(left), int(top), int(left + new_w),
                                  int(top + new_h)))
                overlaps = bbox_overlaps(
                    patch.reshape(-1, 4), boxes.reshape(-1, 4)).reshape(-1)
                if overlaps.min() < min_iou:
                    continue

                # center of boxes should inside the crop img
                center = (boxes[:, :2] + boxes[:, 2:]) / 2
                mask = (center[:, 0] > patch[0]) * (
                    center[:, 1] > patch[1]) * (center[:, 0] < patch[2]) * (
                        center[:, 1] < patch[3])
                if not mask.any():
                    continue
                boxes = boxes[mask]
                labels = labels[mask]

                # adjust boxes
                img = img[patch[1]:patch[3], patch[0]:patch[2]]
                boxes[:, 2:] = boxes[:, 2:].clip(max=patch[2:])
                boxes[:, :2] = boxes[:, :2].clip(min=patch[:2])
                boxes -= np.tile(patch[:2], 2)

                return img, boxes, labels


class ExtraAugmentation(object):

    def __init__(self,
                 photo_metric_distortion=None,
                 expand=None,
                 random_crop=None):
        self.transforms = []
        if photo_metric_distortion is not None:
            self.transforms.append(
                PhotoMetricDistortion(**photo_metric_distortion))
        if expand is not None:
            self.transforms.append(Expand(**expand))
        if random_crop is not None:
            self.transforms.append(RandomCrop(**random_crop))

    def __call__(self, img, boxes, labels):
        img = img.astype(np.float32)
        for transform in self.transforms:
            img, boxes, labels = transform(img, boxes, labels)
        return img, boxes, labels


def paste_over(im_src, im_dst, center):
    """Pastes `im_src` onto `im_dst` at a specified position, with alpha blending, in place.
    Locations outside the bounds of `im_dst` are handled as expected (only a part or none of
    `im_src` becomes visible).
    Args:
        im_src: The RGBA image to be pasted onto `im_dst`. Its size can be arbitrary.
        im_dst: The target image.
        alpha: A float (0.0-1.0) array of the same size as `im_src` controlling the alpha blending
            at each pixel. Large values mean more visibility for `im_src`.
        center: coordinates in `im_dst` where the center of `im_src` should be placed.
    """

    width_height_src = np.asarray([im_src.shape[1], im_src.shape[0]])
    width_height_dst = np.asarray([im_dst.shape[1], im_dst.shape[0]])

    center = np.round(center).astype(np.int32)
    raw_start_dst = center - width_height_src // 2
    raw_end_dst = raw_start_dst + width_height_src

    start_dst = np.clip(raw_start_dst, 0, width_height_dst)
    end_dst = np.clip(raw_end_dst, 0, width_height_dst)
    region_dst = im_dst[start_dst[1]:end_dst[1], start_dst[0]:end_dst[0]]

    start_src = start_dst - raw_start_dst
    end_src = width_height_src + (end_dst - raw_end_dst)
    region_src = im_src[start_src[1]:end_src[1], start_src[0]:end_src[0]]
    color_src = region_src[..., 0:3]
    alpha = region_src[..., 3:].astype(np.float32)/255

    im_dst[start_dst[1]:end_dst[1], start_dst[0]:end_dst[0]] = (
            alpha * color_src + (1 - alpha) * region_dst)


def resize_by_factor(im, factor):
    """Returns a copy of `im` resized by `factor`, using bilinear interp for up and area interp
    for downscaling.
    """
    new_size = tuple(np.round(np.array([im.shape[1], im.shape[0]]) * factor).astype(int))
    interp = cv2.INTER_LINEAR if factor > 1.0 else cv2.INTER_AREA
    return cv2.resize(im, new_size, fx=factor, fy=factor, interpolation=interp)


def list_filepaths(dirpath):
    names = os.listdir(dirpath)
    paths = [os.path.join(dirpath, name) for name in names]
    return sorted(filter(os.path.isfile, paths))