from preprocess import *
import numpy as np
import imgaug.augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage


def rot_2D_random(img, lbl):
    segmap = SegmentationMapsOnImage(lbl.astype(np.int8, copy=False), shape=lbl.shape)
    # random_rotate = np.random.randint(90,size=1)
    aug = iaa.Affine(rotate=(-90, 90))

    img, lbl_aug = aug(image=img, segmentation_maps=segmap)

    lbl = lbl_aug.get_arr()

    return img, lbl

def random_flip_2D(img, lbl):

    aug = np.random.randint(1, 4)
    flip_np_lbl = lbl
    flip_img = img
    np_lbl = label_cat2num(lbl)
    if (aug == 1):
        flip_img = np.flipud(img)
        flip_np_lbl = np.flipud(np_lbl)

    elif (aug == 2):
        flip_img = np.fliplr(img)
        flip_np_lbl = np.fliplr(np_lbl)
    flip_lbl = label_num2cat(flip_np_lbl, num_class)

    return flip_img, flip_lbl