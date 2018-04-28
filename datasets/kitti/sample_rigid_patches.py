#!/usr/bin/env python
# coding=utf-8

import argparse
import os

import cv2
import numpy as np
from PIL import Image
from datasets.cityscapes.labels import *
from datasets.cityscapes.sample_rigid_patches import search_bg_patches


def annotation2instances(anno_fn):
    # Load annotation image
    anno = np.array(Image.open(anno_fn))

    inst_ids = [id for id in np.unique(anno) if id2label[id // 256].hasInstances]
    bbox = []
    for id in inst_ids:
        y, x = np.where(anno == id)
        bbox.append([np.min(x), np.min(y), np.max(x) - np.min(x), np.max(y) - np.min(y)])
    return bbox


def extract_rigid_patches(obj_bbox, w, h):
    if len(obj_bbox):
        return search_bg_patches(obj_bbox, w, h)
    else:
        return [[0, 0, w, h]]


if __name__ == "__main__":
    parser = argparse.ArgumentParser('sample rigid patches')
    parser.add_argument('--fold', dest='fold', help='KITTI semantic segmantation directory', type=str,
                        default='../datasets/KITTI/dataset/segmentation')
    parser.add_argument('--fold_rigid', dest='fold_rigid', help='output image directory', type=str,
                        default='../datasets/rigid')
    parser.add_argument('--num_imgs', dest='num_imgs', help='number of images', type=int, default=1000000)
    args = parser.parse_args()

    for arg in vars(args):
        print('[%s] = ' % arg, getattr(args, arg))

    # splits = os.listdir(os.path.join(args.fold, 'leftImg8bit'))
    splits = ['train']
    for sp in splits:
        img_fold_raw = os.path.join(args.fold, sp + 'ing', 'image_2')
        anno_fold = os.path.join(args.fold, sp + 'ing', 'instance')
        img_fold_rigid = os.path.join(args.fold_rigid, sp, 'kitti')
        if not os.path.isdir(img_fold_rigid):
            os.makedirs(img_fold_rigid)

        img_list = os.listdir(img_fold_raw)
        num_imgs = min(args.num_imgs, len(img_list))
        print('split %s, use %d/%d images' % (sp, num_imgs, len(img_list)))

        for n in range(num_imgs):
            fn = img_list[n]
            name = fn.rsplit('.', 1)[0]
            path_raw = os.path.join(img_fold_raw, fn)
            path_anno = os.path.join(anno_fold, fn)
            if os.path.isfile(path_raw) and os.path.isfile(path_anno):
                im = cv2.imread(path_raw)
                obj_bbox = annotation2instances(path_anno)

                # for bbox in obj_bbox:
                #     cv2.rectangle(im, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (255, 0, 0), 16)
                # cv2.imshow("image", cv2.resize(im, (int(im.shape[1] / 2), int(im.shape[0] / 2))))
                # cv2.waitKey(0)

                rigid_patches = extract_rigid_patches(obj_bbox, im.shape[1], im.shape[0])
                for i, patch_bbox in enumerate(rigid_patches):
                    path_rigid = os.path.join(img_fold_rigid, '{}_{}.png'.format(name, i))
                    x = patch_bbox[0]
                    y = patch_bbox[1]
                    w = patch_bbox[2]
                    h = patch_bbox[3]

                    # cv2.rectangle(im, (x, y), (x + w, y + h),
                    #               (np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256)),
                    #               thickness=8)
                    # cv2.imshow("image", cv2.resize(im, (int(im.shape[1] / 2), int(im.shape[0] / 2))))
                    # cv2.waitKey(0)

                    cv2.imwrite(path_rigid, im[y:y + h, x:x + w])
            else:
                if not os.path.isfile(path_raw):
                    print('Cannot find {}'.format(path_raw))
                if not os.path.isfile(path_anno):
                    print('Cannot find {}'.format(path_anno))
