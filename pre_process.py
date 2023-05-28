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
    for i in range(gt_img.shape[0] - 1, -1, -1):
        chip = gt_img[i]
        if np.any(chip > 0):
            x_2 = i
            break

    y_1 = -1
    y_2 = -1
    # from left to right
    for i in range(gt_img.shape[1]):
        chip = gt_img[:, i]
        if np.any(chip > 0):
            y_1 = i
            break

    # from right to left
    for i in range(gt_img.shape[1] - 1, -1, -1):
        chip = gt_img[:, i]
        if np.any(chip > 0):
            y_2 = i
            break

    coordinate = [x_1, x_2, y_1, y_2]

    return (coordinate[0] + coordinate[1]) // 2, (coordinate[2] + coordinate[3]) // 2


def _get_patch(img, gt_img, coord):
    patch = img[coord[0] - 15: coord[0] + 17, coord[1] - 15:coord[1] + 17]
    anno = gt_img[coord[0] - 15: coord[0] + 17, coord[1] - 15:coord[1] + 17]
    return patch, anno


def set_patch(result, counter_map,  patch, coord):
    """
    set patch in result
    :param result: whole image result
    :param patch: patch result
    :param coord: coordinate of center point_move
    :return: 
    """
    result[coord[0] - 15: coord[0] + 17, coord[1] - 15:coord[1] + 17] += patch
    counter_map[coord[0] - 15: coord[0] + 17, coord[1] - 15:coord[1] + 17] += 1


def _extract_patches(img, gt_img, coordinate, length=15, stride=4):
    """
    extract patch from raw images, for each sequence extract 10 patches with stride 4
    :param img: 
    :param gt_img:
    :param coordinate: 
    :param length: 
    :param stride: 
    :return: 
    """
    seq_patches = []
    seq_annos = []
    for direction in directions:
        patches, annos = direction_extract(img, gt_img, coordinate, direction, length, stride)
        seq_patches.extend(patches)
        seq_annos.extend(annos)
    return seq_patches, seq_annos


def direction_extract(img, gt_img, coordinate, direction, length=15, stride=4):
    """
    extract patches along specified angle
    :param img: 
    :param coordinate: 
    :param direction: tuple for axis
    :param length: 
    :param stride: 
    :return: two list patchs and annos. each list contains two np array sequence 
    """
    patches = []
    annos = []
    coord = []
    coord.append(coordinate[0])
    coord.append(coordinate[1])
    # positive direction
    pos_patches = []
    pos_annos = []

    for i in range(length):
        coord[0] += direction[0] * stride
        coord[1] += direction[1] * stride
        patch, anno = _get_patch(img, gt_img, coord)
        pos_patches.append(patch)
        pos_annos.append(anno)
    patches.append(np.array(pos_patches))
    annos.append(np.array(pos_annos))

    # negetive directioin
    coord[0] = coordinate[0]
    coord[1] = coordinate[1]
    neg_patches = []
    neg_annos = []

    for i in range(length):
        coord[0] -= direction[0] * stride
        coord[1] -= direction[1] * stride
        patch, anno = _get_patch(img, gt_img, coord)
        neg_pat