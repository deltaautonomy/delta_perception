#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 18-5-15 下午4:29
# @Author  : Luo Yao
# @Site    : https://github.com/MaybeShewill-CV/lanenet-lane-detection
# @File    : lanenet_cluster.py
# @IDE: PyCharm Community Edition
"""
实现LaneNet中实例分割的聚类部分
"""
import time
import warnings

import cv2
import statistics
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import MeanShift
from sklearn.cluster import DBSCAN

try:
    from cv2 import cv2
except ImportError:
    pass


class LaneNetCluster(object):

    def __init__(self):
        # Actual colors
        self._color_map = [np.array([255, 0, 0]), # can display upto 8 lanes
                           np.array([0, 255, 0]),
                           np.array([0, 0, 255]),
                           np.array([125, 125, 0]),
                           np.array([0, 125, 125]),
                           np.array([125, 0, 125]),
                           np.array([50, 100, 50]),
                           np.array([100, 50, 100])]

        self.slope_history_first = []
        self.slope_history_second = []
        self.slope_history_third = []

        self.intercept_history_first = []
        self.intercept_history_second = []
        self.intercept_history_third = []

    @staticmethod
    def _cluster(prediction, bandwidth):
        """
        实现论文SectionⅡ的cluster部分
        :param prediction:
        :param bandwidth:
        :return:
        """
        # print ('-------------------------------------------------------------------------------------------')
        # print (' this is _cluster')
        # print ('-------------------------------------------------------------------------------------------')

        ms = MeanShift(bandwidth, bin_seeding=True)
        tic = time.time()
        try:
            ms.fit(prediction)
        except ValueError as err:
            print(err)
            return 0, [], []
        labels = ms.labels_
        cluster_centers = ms.cluster_centers_
        num_clusters = cluster_centers.shape[0]
        return num_clusters, labels, cluster_centers

    @staticmethod
    def _cluster_v2(prediction):
        """
        dbscan cluster
        :param prediction:
        :return:
        """
        db = DBSCAN(eps=0.7, min_samples=50).fit(prediction)
        db_labels = db.labels_
        unique_labels = np.unique(db_labels)
        unique_labels = [tmp for tmp in unique_labels if tmp != -1]
        # print('from_cluser_v2: {:d}'.format(len(unique_labels)))
        num_clusters = len(unique_labels)
        cluster_centers = db.components_

        return num_clusters, db_labels, cluster_centers

    @staticmethod
    def _get_lane_area(binary_seg_ret, instance_seg_ret):
        """
        :param binary_seg_ret:
        :param instance_seg_ret:
        :return:
        """
        idx = np.where(binary_seg_ret == 1)
        lane_embedding_feats = []
        lane_coordinate = []
        for i in range(len(idx[0])):
            lane_embedding_feats.append(instance_seg_ret[idx[0][i], idx[1][i]])
            lane_coordinate.append([idx[0][i], idx[1][i]])

        return np.array(lane_embedding_feats, np.float32), np.array(lane_coordinate, np.int64)

    @staticmethod
    def _thresh_coord(coord):
        """
        :param coord: [(x, y)]
        :return:
        """
        pts_x = coord[:, 0]
        mean_x = np.mean(pts_x)
        idx = np.where(np.abs(pts_x - mean_x) < mean_x)

        return coord[idx[0]]

    @staticmethod
    def _lane_fit(lane_pts):
        """
        :param lane_pts:
        :return:
        """
        if not isinstance(lane_pts, np.ndarray):
            lane_pts = np.array(lane_pts, np.float32)

        x = lane_pts[:, 0]
        y = lane_pts[:, 1]
        x_fit = []
        y_fit = []
        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            try:
                f1 = np.polyfit(x, y, 3)
                p1 = np.poly1d(f1)
                x_min = int(np.min(x))
                x_max = int(np.max(x))
                x_fit = []
                for i in range(x_min, x_max + 1):
                    x_fit.append(i)
                y_fit = p1(x_fit)
            except Warning as e:
                x_fit = x
                y_fit = y
            finally:
                return zip(x_fit, y_fit)

    @staticmethod
    def custom_filter(slope, intercept, slope_history, intercept_history):
        alpha = 0.2 # Weight when detected
        beta = 0.1 # Weight when not detected
        window_size = 15
        slope_threshold = 0.05
        intercept_threshold = 30

        slope_history.append(slope)
        intercept_history.append(intercept)
        # print(slope_history)
        # print(intercept_history)

        if len(slope_history) == window_size:
            # Detections within thresholds
            if abs(slope - slope_history[-2]) > slope_threshold or \
                abs(intercept - intercept_history[-2]) > intercept_threshold:
                weight = beta
            else:
                weight = alpha

            # Moving average filter
            slope_filtered = statistics.mean(slope_history[:-1]) * (1 - weight) + (slope * weight)
            intercept_filtered = statistics.mean(intercept_history[:-1]) * (1 - weight) + (intercept * weight)

            # Remove old data
            slope_history.remove(slope_history[0])
            intercept_history.remove(intercept_history[0])

            return slope_filtered, intercept_filtered

        else:
            return slope, intercept

    def get_lane_mask(self, binary_seg_ret, instance_seg_ret):
        """

        :param binary_seg_ret:
        :param instance_seg_ret:
        :return:
        """
        # print ('-------------------------------------------------------------------------------------------')
        # print (' this is get_lane_mask')
        # print ('-------------------------------------------------------------------------------------------')
        lane_embedding_feats, lane_coordinate = self._get_lane_area(binary_seg_ret, instance_seg_ret)
        num_clusters, labels, cluster_centers = self._cluster_v2(lane_embedding_feats)

        if num_clusters > 8:
            cluster_sample_nums = []
            for i in range(num_clusters):
                cluster_sample_nums.append(len(np.where(labels == i)[0]))
            sort_idx = np.argsort(-np.array(cluster_sample_nums, np.int64))
            cluster_index = np.array(range(num_clusters))[sort_idx[0:8]]
        else:
            cluster_index = range(num_clusters)

        # Output image
        mask_image = np.zeros(shape=[binary_seg_ret.shape[0], binary_seg_ret.shape[1], 3], dtype=np.uint8)
        height = mask_image.shape[0]

        # Output data for validation
        slope_list = []
        intercept_list = []
        detected_points = []
        validation_points = []

        # Line fitting on 3 lanes
        lanes_count = 0
        for index, i in enumerate(cluster_index):  # this is not used - to restrict only 3 lane detections
            if index > 2: break
            lanes_count += 1

            idx = np.where(labels == i)
            coord = lane_coordinate[idx]
            coord = np.flip(coord, axis=1)
            coord = np.array([coord])
            num_points = coord.shape[1]

            # for i in range(num_points):
            #     cv2.circle(mask_image,(coord[0][i][0], coord[0][i][1]), 1, (255,255,255), -1)

            starting_point = np.squeeze(coord[:, 0]).tolist()
            ending_point = np.squeeze(coord[:, num_points -1]).tolist()

            xpoints = [starting_point[0], ending_point[0]]
            ypoints = [starting_point[1], ending_point[1]]

            parameters = np.polyfit(xpoints, ypoints, 1)
            slope, intercept = parameters

            slope_list.append(slope)
            intercept_list.append(intercept)

        # Slope intercept filtering
        if lanes_count == 3:
            # Get indices
            max_slope_index = np.argmax(slope_list)
            min_slope_index = np.argmin(slope_list)
            mid_slope_index = slope_list.index(np.median(slope_list))

            # Lane 1
            slope_first = slope_list[max_slope_index]
            intercept_first = intercept_list[max_slope_index]
            slope_first, intercept_first = LaneNetCluster.custom_filter(slope_first,
                intercept_first, self.slope_history_first, self.intercept_history_first)

            # Lane 2
            slope_second = slope_list[min_slope_index]
            intercept_second = intercept_list[min_slope_index]
            slope_second, intercept_second = LaneNetCluster.custom_filter(slope_second,
                intercept_second, self.slope_history_second, self.intercept_history_second)

            # Lane 3
            slope_third = slope_list[mid_slope_index]
            intercept_third = intercept_list[mid_slope_index]
            slope_third, intercept_third = LaneNetCluster.custom_filter(slope_third,
                intercept_third, self.slope_history_third, self.intercept_history_third)

        elif len(self.slope_history_first):
            slope_first, intercept_first = self.slope_history_first[-1], self.intercept_history_first[-1]
            slope_second, intercept_second = self.slope_history_second[-1], self.intercept_history_second[-1]
            slope_third, intercept_third = self.slope_history_third[-1], self.intercept_history_third[-1]

        # Draw the lanes
        if len(self.slope_history_first):
            # validation heights
            val_y1 = 135
            val_y2 = 80

            # Lane 1
            y1 = int(height) # starting point of the lanes
            y2 = int(0.9*height/5) # end point of the lanes
            x1 = int((y1 - intercept_first)/slope_first)
            x2 = int((y2 - intercept_first)/slope_first)
            val_x1 = int((val_y1 - intercept_first)/slope_first)
            val_x2 = int((val_y2 - intercept_first)/slope_first)
            detected_points.append([x1, y1, x2, y2])
            validation_points.append([val_x1, val_y1, val_x2, val_y2])
            # cv2.circle(mask_image,(val_x1, val_y1), 6, (0, 0, 255), -1)
            # cv2.circle(mask_image,(val_x2, val_y2), 6, (255, 255, 0), -1)
            cv2.line(mask_image, (x1, y1), (x2, y2), (255, 0, 0), 2)

            # Lane 2
            y1 = int(height) # starting point of the lanes
            y2 = int(0.9 * height/5) # end point of the lanes
            x1 = int((y1 - intercept_second)/slope_second)
            x2 = int((y2 - intercept_second)/slope_second)
            val_x1 = int((val_y1 - intercept_second)/slope_second)
            val_x2 = int((val_y2 - intercept_second)/slope_second)
            detected_points.append([x1, y1, x2, y2])
            validation_points.append([val_x1, val_y1, val_x2, val_y2])
            # cv2.circle(mask_image,(val_x1, val_y1), 4, (0, 255, 0), -1)
            # cv2.circle(mask_image,(val_x2, val_y2), 4, (255, 0, 255), -1)
            cv2.line(mask_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Lane 3
            y1 = int(height) # starting point of the lanes
            y2 = int(0.9 * height/5) # end point of the lanes
            x1 = int((y1 - intercept_third)/slope_third)
            x2 = int((y2 - intercept_third)/slope_third)
            val_x1 = int((val_y1 - intercept_third)/slope_third)
            val_x2 = int((val_y2 - intercept_third)/slope_third)
            detected_points.append([x1, y1, x2, y2])
            validation_points.append([val_x1, val_y1, val_x2, val_y2])
            # cv2.circle(mask_image,(val_x1, val_y1), 2, (255, 0, 0), -1)
            # cv2.circle(mask_image,(val_x2, val_y2), 2, (0, 255, 255), -1)
            cv2.line(mask_image, (x1, y1), (x2, y2), (0, 0, 255), 2)

        return mask_image, np.asarray(detected_points[::-1]), np.asarray(validation_points[::-1])


if __name__ == '__main__':
    # print ('-------------------------------------------------------------------------------------------')
    # print (' this is __main__')
    # print ('-------------------------------------------------------------------------------------------')
    binary_seg_image = cv2.imread('binary_ret.png', cv2.IMREAD_GRAYSCALE)
    binary_seg_image[np.where(binary_seg_image == 255)] = 1
    instance_seg_image = cv2.imread('instance_ret.png', cv2.IMREAD_UNCHANGED)
    ele_mex = np.max(instance_seg_image, axis=(0, 1))
    for i in range(3):
        if ele_mex[i] == 0:
            scale = 1
        else:
            scale = 255 / ele_mex[i]
        instance_seg_image[:, :, i] *= int(scale)
    embedding_image = np.array(instance_seg_image, np.uint8)
    cluster = LaneNetCluster()
    mask_image = cluster.get_lane_mask(instance_seg_ret=instance_seg_image, binary_seg_ret=binary_seg_image) # same line as in the test_lane function 
    plt.figure('embedding')
    plt.imshow(embedding_image[:, :, (2, 1, 0)])
    plt.figure('mask_image')
    plt.imshow(mask_image[:, :, (2, 1, 0)])
    plt.show()