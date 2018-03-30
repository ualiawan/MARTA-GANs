# -*- coding: utf-8 -*-

import scipy.misc
import numpy as np

def center_crop(x, crop_h, crop_w=None, resize_w=64):
    if crop_w is None:
        crop_w = crop_h
    h, w = x.shape[:2]
    j = int(round((h - crop_h)/2.))
    i = int(round((w - crop_w)/2.))
    return scipy.misc.imresize(x[j:j+crop_h, i:i+crop_w],
                               [resize_w, resize_w])

def load_image(image_path, image_size, is_crop=True, resize_w=64):
    image =  scipy.misc.imread(image_path).astype(np.float)
    if is_crop:
        cropped_image  = center_crop(image, image_size, resize_w=resize_w)
    else:
        cropped_image =image
    return np.array(cropped_image)/127.5 - 1.


def get_labels(num_labels, lables_file):
    style_labels = list(np.loadtxt(lables_file, str, delimiter='\n'))
    if num_labels > 0:
        style_labels = style_labels[:num_labels]
    return style_labels