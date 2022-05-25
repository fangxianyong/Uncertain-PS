"""
Based on https://github.com/asanakoy/kaggle_carvana_segmentation
"""
import torch
import torch.utils.data as data
from torch.autograd import Variable as V
from PIL import Image

import cv2
import numpy as np
import os
import scipy.misc as misc
import albumentations as A

#图像增强

def data_augmentation(image, mask, edge):
    r_image, r_mask, r_edge = image, mask, edge
    r_image = _normalize(r_image,[0.485, 0.456, 0.406],
                        [0.229, 0.224, 0.225])
    r_image, r_mask, r_edge = _resize(r_image, r_mask, r_edge)
    r_image, r_mask, r_edge = _randomHorizontalFlip(r_image, r_mask, r_edge)
    r_image, r_mask, r_edge = _randomVerticleFlip(r_image, r_mask, r_edge)
    r_image, r_mask, r_edge = _randomRotate90(r_image, r_mask, r_edge)
    r_image, r_mask, r_edge = _toTensor(r_image, r_mask, r_edge)
    return r_image, r_mask, r_edge

def no_data_augmentation(image, mask, edge):
    r_image, r_mask, r_edge = image, mask, edge
    r_image = _normalize(r_image,[0.485, 0.456, 0.406],
                       [0.229, 0.224, 0.225])
    r_image, r_mask, r_edge = _resize(r_image, r_mask, r_edge)
    r_image, r_mask, r_edge = _toTensor(r_image, r_mask, r_edge)
    return r_image, r_mask, r_edge

def _normalize(img, mean, std, max_pixel_value=255.0):
    mean = np.array(mean, dtype=np.float32)
    mean *= max_pixel_value

    std = np.array(std, dtype=np.float32)
    std *= max_pixel_value

    denominator = np.reciprocal(std, dtype=np.float32)

    img = img.astype(np.float32)
    img -= mean
    img *= denominator
    return img

def _resize(image,mask,edge,size=(224,224)):
    image = cv2.resize(image,size)
    mask = cv2.resize(mask,size)
    edge = cv2.resize(edge,size)
    return image, mask,edge

def _toTensor(image,mask,edge):
    r_image, r_mask, r_edge = image.copy(), mask.copy(), edge.copy()
    r_image = np.transpose(r_image,(2,0,1))
    r_image = torch.from_numpy(r_image).float()
    r_mask = torch.from_numpy(r_mask).float()
    r_edge = torch.from_numpy(r_edge).float()
    return r_image, r_mask, r_edge

def _randomHueSaturationValue(image, hue_shift_limit=(-180, 180),
                             sat_shift_limit=(-255, 255),
                             val_shift_limit=(-255, 255), u=0.7):
    if np.random.random() < u:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(image)
        hue_shift = np.random.randint(hue_shift_limit[0], hue_shift_limit[1]+1)
        hue_shift = np.uint8(hue_shift)
        h += hue_shift
        sat_shift = np.random.uniform(sat_shift_limit[0], sat_shift_limit[1])
        s = cv2.add(s, sat_shift)
        val_shift = np.random.uniform(val_shift_limit[0], val_shift_limit[1])
        v = cv2.add(v, val_shift)
        image = cv2.merge((h, s, v))
        #image = cv2.merge((s, v))
        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

    return image


def _randomShiftScaleRotate(image, mask, edge,
                           shift_limit=(-0.0, 0.0),
                           scale_limit=(-0.0, 0.0),
                           rotate_limit=(-0.0, 0.0), 
                           aspect_limit=(-0.0, 0.0),
                           borderMode=cv2.BORDER_CONSTANT, u=0.9):
    if np.random.random() < u:
        height, width, channel = image.shape

        angle = np.random.uniform(rotate_limit[0], rotate_limit[1])
        scale = np.random.uniform(1 + scale_limit[0], 1 + scale_limit[1])
        aspect = np.random.uniform(1 + aspect_limit[0], 1 + aspect_limit[1])
        sx = scale * aspect / (aspect ** 0.5)
        sy = scale / (aspect ** 0.5)
        dx = round(np.random.uniform(shift_limit[0], shift_limit[1]) * width)
        dy = round(np.random.uniform(shift_limit[0], shift_limit[1]) * height)

        cc = np.math.cos(angle / 180 * np.math.pi) * sx
        ss = np.math.sin(angle / 180 * np.math.pi) * sy
        rotate_matrix = np.array([[cc, -ss], [ss, cc]])

        box0 = np.array([[0, 0], [width, 0], [width, height], [0, height], ])
        box1 = box0 - np.array([width / 2, height / 2])
        box1 = np.dot(box1, rotate_matrix.T) + np.array([width / 2 + dx, height / 2 + dy])

        box0 = box0.astype(np.float32)
        box1 = box1.astype(np.float32)
        mat = cv2.getPerspectiveTransform(box0, box1)
        image = cv2.warpPerspective(image, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                                    borderValue=(
                                        0, 0,
                                        0,))
        mask = cv2.warpPerspective(mask, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                                   borderValue=(
                                       0, 0,
                                       0,))
        edge = cv2.warpPerspective(mask, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                                   borderValue=(
                                       0, 0,
                                       0,))

    return image, mask,edge

def _randomHorizontalFlip(image, mask, edge, u=0.5):
    if np.random.random() < u:
        image = cv2.flip(image, 1)
        mask = cv2.flip(mask, 1)
        edge = cv2.flip(edge,1)

    return image, mask,edge

def _randomVerticleFlip(image, mask,edge, u=0.5):
    if np.random.random() < u:
        image = cv2.flip(image, 0)
        mask = cv2.flip(mask, 0)
        edge = cv2.flip(edge,0)

    return image, mask,edge

def _randomRotate90(image, mask, edge,u=0.5):
    if np.random.random() < u:
        image=np.rot90(image)
        mask=np.rot90(mask)
        edge = np.rot90(edge)

    return image, mask,edge





