import numpy as np
from skimage.io import imread, imsave, imshow, show
from glob import glob
import os


HEIGHT = 296
WIDTH = 296
PATCH_HEIGHT = 32
PATCH_WIDTH = 32
PADDING_SHAPE = (436, 436)

train_data_dir = ''

val_data_dir = ''

seq_length = 15
extract_stride = 4


directions = [(0, 1), (-1, 2), (-1, 1), (-2, 1),  (-1, 0), (-2, -1), (-1, -1), (-1, -2)]


def load_img(file_dir, img_name):
    """
    load origin img and responding ground truth img
    :param img_name: 
    :return: img gt_img
    """
    img = imread(file_dir + img_name + '.jpg')
    gt_img = imread(file_dir + img_name + '_1.bmp')
    return img, gt_img


def get_coordinate(img, gt_img, padding):
    """
    get crop from img and gt_img
    :param img: 
    :param gt_img: 
    :param padding: 
    :return: center point_move of the gt
    """
    x_1 = -1
    x_2 = -1
    # from top to down
    for i in range(gt_img.shape[0]):
        chip = gt_img[i]
        if np.any(chip > 0):
            x_1 = i
            break

    # from down to top
    for i in range(gt_img.shape[0] - 1,