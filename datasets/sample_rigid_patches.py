#!/usr/bin/env python
# coding=utf-8

import argparse
import os

import cv2
import numpy as np
from multiprocessing import Pool

from datasets.cityscapes.read_cityscapes_json import json2instances


def intersection_ratio(bbox_A, bbox_B):
    # determine the (x, y)-coordinates of the intersection rectangle
    inter_x0 = max(bbox_A[0], bbox_B[0])
    inter_y0 = max(bbox_A[1], bbox_B[1])
    inter_x1 = min(bbox_A[0] + bbox_A[2], bbox_B[0] + bbox_B[2])
    inter_y1 = min(bbox_A[1] + bbox_A[3], bbox_B[1] + bbox_B[3])

    # compute the area of intersection rectangle
    area_inter = (inter_x1 - inter_x0) * (inter_y1 - inter_y0)

    # compute the area of both the prediction and ground-truth
    # rectangles
    area_A = bbox_A[2] * bbox_A[3]
    area_B = bbox_B[2] * bbox_B[3]

    # return the intersection ratio of each patch
    return area_inter / float(area_A), area_inter / float(area_B)


def add_new_box(stored_boxes, new_box):
    new_boxes = []
    for bbox in stored_boxes:
        if max(intersection_ratio(bbox, new_box)) >= 0.8:
            if bbox[2] * bbox[3] > new_box[2] * new_box[3]:
                return stored_boxes
        else:
            new_boxes.append(bbox)
    new_boxes.append(new_box)
    return new_boxes


def search_bg_patches(obj_bbox, w, h):
    obj_bbox = np.array(obj_bbox)
    bg_patches = []

    # Minimum width and height of resulting patches.
    min_w, min_h = max(96, w * 0.2), max(96, h * 0.2)

    lw_bounds = obj_bbox[:, 1].copy()
    lw_bounds.sort()
    lw_bounds = np.hstack((lw_bounds, h))

    up_bounds = obj_bbox[:, 1] + obj_bbox[:, 3]
    up_bounds.sort()
    up_bounds = np.hstack((0, up_bounds))

    # Iterate each vertical section.
    # The lower boundary of the section is the upper boundary of a bounding box,
    # and the upper boundary of the section is the lower boundary of a bounding box.
    for sec_up in reversed(lw_bounds):
        for sec_lw in up_bounds:
            if sec_up - sec_lw < min_h:
                continue

            # Find bounding boxes that are fully or partly included in the section.
            bbox_included = []
            for bbox in obj_bbox:
                if not (bbox[1] >= sec_up or bbox[1] + bbox[3] <= sec_lw):
                    bbox_included.append(bbox)

            # Add the left and right boundaries of the included bounding boxes into a list, with labels.
            h_bounds = [[0, 1], [w, 0]]
            for bbox in bbox_included:
                h_bounds.append((bbox[0], 0))
                h_bounds.append((bbox[0] + bbox[2], 1))
            h_bounds = np.array(h_bounds)

            # Sort the boundaries from low to high.
            h_bounds.view('i8,i8').sort(axis=0, order=['f0'])

            # Find unoccupied horizontal sections.
            occupied_cnt = 1
            left = 0
            for b in h_bounds:
                if b[1]:
                    # Reached a right boundary.
                    occupied_cnt -= 1
                    if not occupied_cnt:
                        left = b[0]
                else:
                    # Reached an left boundary.
                    if not occupied_cnt:
                        right = b[0]
                        if right - left > min_h:
                            bg_patches = add_new_box(bg_patches, [left, sec_lw, right - left, sec_up - sec_lw])
                    occupied_cnt += 1
    return bg_patches


def extract_rigid_patches(instances, w, h):
    obj_bbox = []
    for polygon, _ in instances:
        polygon = np.array(polygon)
        xmin = np.min(polygon[:, 0])
        xmax = np.max(polygon[:, 0])
        ymin = np.min(polygon[:, 1])
        ymax = np.max(polygon[:, 1])
        obj_bbox.append([xmin, ymin, xmax - xmin, ymax - ymin])

    if len(obj_bbox):
        return search_bg_patches(obj_bbox, w, h)
    else:
        return [[0, 0, w, h]]


if __name__ == "__main__":
    parser = argparse.ArgumentParser('sample rigid patches')
    parser.add_argument('--fold', dest='fold', help='CityScapes directory', type=str,
                        default='../datasets/CityScapes')
    parser.add_argument('--fold_rigid', dest='fold_rigid', help='output image directory', type=str,
                        default='../datasets/rigid')
    parser.add_argument('--num_imgs', dest='num_imgs', help='number of images', type=int, default=1000000)
    args = parser.parse_args()

    for arg in vars(args):
        print('[%s] = ' % arg, getattr(args, arg))

    # splits = os.listdir(os.path.join(args.fold, 'leftImg8bit'))
    splits = ['train', 'val', 'test']
    for sp in splits:
        if sp == 'train_extra':
            gt = 'gtCoarse'
        else:
            gt = 'gtFine'

        img_fold_raw = os.path.join(args.fold, 'leftImg8bit', sp)
        anno_fold = os.path.join(args.fold, gt, sp)

        for subfolder in os.listdir(img_fold_raw):
            img_subfolder_path = os.path.join(img_fold_raw, subfolder)
            anno_subfolder_path = os.path.join(anno_fold, subfolder)

            img_subfolder_rigid = os.path.join(args.fold_rigid, sp, subfolder)
            if not os.path.isdir(img_subfolder_rigid):
                os.makedirs(img_subfolder_rigid)

            img_list = os.listdir(img_subfolder_path)

            num_imgs = min(args.num_imgs, len(img_list))
            print('%s in split %s, use %d/%d images' % (subfolder, sp, num_imgs, len(img_list)))

            for n in range(num_imgs):
                fn = img_list[n]
                name = fn.rsplit('.', 1)[0].rsplit('_', 1)[0]
                path_raw = os.path.join(img_subfolder_path, fn)
                path_anno = os.path.join(anno_subfolder_path, '{}_{}_polygons.json'.format(name, gt))
                if os.path.isfile(path_raw) and os.path.isfile(path_anno):
                    im = cv2.imread(path_raw)

                    instances = json2instances(path_anno)

                    # obj_bbox = []
                    # for polygon, _ in instances:
                    #     polygon = np.array(polygon)
                    #     xmin = np.min(polygon[:, 0])
                    #     xmax = np.max(polygon[:, 0])
                    #     ymin = np.min(polygon[:, 1])
                    #     ymax = np.max(polygon[:, 1])
                    #     obj_bbox.append([xmin, ymin, xmax - xmin, ymax - ymin])
                    # for bbox in obj_bbox:
                    #     cv2.rectangle(im, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (255, 0, 0), 16)
                    # cv2.imshow("image", cv2.resize(im, (int(im.shape[1] / 2), int(im.shape[0] / 2))))
                    # cv2.waitKey(0)

                    rigid_patches = extract_rigid_patches(instances, im.shape[1], im.shape[0])
                    for i, patch_bbox in enumerate(rigid_patches):
                        path_rigid = os.path.join(img_subfolder_rigid, '{}_{}.jpg'.format(name, i))
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
