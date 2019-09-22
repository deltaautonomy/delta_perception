#!/usr/bin/env python

import os
import sys
import glob
import argparse

ROS_CV = '/opt/ros/kinetic/lib/python2.7/dist-packages'
if ROS_CV in sys.path: sys.path.remove(ROS_CV)

import cv2
from natsort import natsorted


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', '-i', type=str,
                        help='Path to images folder')
    parser.add_argument('--output_path', '-o', type=str,
                        help='Path to output folder')
    parser.add_argument('--dataset', '-d', type=str,
                        help='Dataset (CULane | CULane2 | Cityscapes)', default='CULane2')
    args = parser.parse_args()
    return args


def split_filename(filename, split_ext=False):
    if split_ext:
        return ''.join(os.path.basename(filename).split('.')[:-1])
    return os.path.basename(filename)


def crop(filename, save_path, dataset='CULane', idx=0, total=0, save=True, display=False):
    if dataset == 'CULane':
        dims = (288, 800)
        offset = (50, 0)
    elif dataset == 'CULane2':
        dims = (208, 976)
        offset = (0, 0)
    elif dataset == 'Cityscapes':
        dims = (512, 1024)
        offset = (32, 0)
    else:
        raise ValueError('Dataset not supported')

    img = cv2.imread(filename)
    img = cv2.resize(img, None, fx=dims[1]/img.shape[1],
                     fy=dims[1]/img.shape[1], interpolation=cv2.INTER_LINEAR)

    diff = int(abs(img.shape[0] - dims[0]) / 2) + offset[0]
    img = img[diff:diff + dims[0]]

    if display:
        cv2.imshow('Cropped', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    if save:
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        output_filename = os.path.join(save_path, '%s-crop-%s.jpg' % (
            dataset.lower(), split_filename(filename, split_ext=True)))
        cv2.imwrite(output_filename, img)

        counter = '%%0%dd' % len(str(total))
        text = '\rSaving (%s/%s): %%s %%s %s' % (counter, counter, ' ' * 10)
        sys.stdout.write(
            text % (idx + 1, total, output_filename, str(img.shape)))
        sys.stdout.flush()


if __name__ == '__main__':
    args = parse_args()

    # Load the image filenames
    images = natsorted([f for f in glob.glob(
        os.path.join(args.image_path, '*.jpg'))])

    print('Processing %d images' % len(images))
    for i, image in enumerate(images):
        crop(image, args.output_path, args.dataset, i, len(images))
    print('\nDone')
