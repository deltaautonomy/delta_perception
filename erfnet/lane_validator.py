#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Author  : Heethesh Vhavle
Email   : heethesh@cmu.edu
Version : 1.0.0
Date    : Nov 25, 2019
'''

# Python 2/3 compatibility
from __future__ import print_function, absolute_import, division

# Standalone
if __name__ == '__main__':
    import sys
    sys.path.append('..')

    # Handle paths and OpenCV import
    from scripts.init_paths import *

# Run from scripts module
else:
    # Handle paths and OpenCV import
    from init_paths import *

# External modules
import matplotlib.pyplot as plt
from natsort import natsorted
from scipy.spatial import distance
from scipy.optimize import linear_sum_assignment
from pyclustering.cluster.kmedians import kmedians

# Local python modules
from ipm.ipm import InversePerspectiveMapping
from erfnet.lane_filter import LaneKalmanFilter

LANE_GT_COLOR = np.array([50, 230, 155])


class LaneValidator:
    def __init__(self):
        self.ipm = InversePerspectiveMapping()
        self.lane_filter = LaneKalmanFilter()
        self.first_time = True
        self.kmedians_centroids = np.asarray([[0, 30], [0, 90], [0, 130]], dtype=np.float64)

    def hough_line_detector(self, ipm_lane):
        canny_lane = cv2.Canny(ipm_lane, 50, 200, None, 3)

        # Copy edges to the images that will display the results in BGR.
        dst = cv2.cvtColor(canny_lane, cv2.COLOR_GRAY2BGR)

        # Probabilistic hough lines detector.
        lines_lanes = cv2.HoughLinesP(canny_lane, 1, np.pi / 180, 50, None, 50, 10)
        if lines_lanes is not None:
            lines_lanes = lines_lanes.squeeze(1)
            for i in range(0, len(lines_lanes)):
                l = lines_lanes[i]
                cv2.line(dst, (l[0], l[1]), (l[2], l[3]), (0, 255, 255), 1, cv2.LINE_AA)

        # Handle no line detections.
        return dst, lines_lanes

    def associate_lanes(self, medians, pred_medians):
        association = [None] * 3
        cost = distance.cdist(np.expand_dims(pred_medians[:, 1], 1), np.expand_dims(medians[:, 1], 1))
        idx_a, idx_b = linear_sum_assignment(cost)
        for i, j in zip(idx_a, idx_b): association[i] = medians[j]
        return association

    def detect_lines(self, img, timestamp):
        # Rescale to camera image size (720 * 1280)
        img_gt = cv2.resize(img, (0, 0), fx=1.6, fy=1.6)
        img_gt = img_gt[120:840, :, :]
        ipm_img = self.ipm.transform_image(img_gt)
        output = ipm_img

        # Segment out lanes
        image1 = np.where(img_gt >= LANE_GT_COLOR - 10, 255, 0)
        image2 = np.where(img_gt <= LANE_GT_COLOR + 10, 255, 0)
        img_gt = (image1 & image2).astype('uint8')
        img_gt = self.ipm.transform_image(img_gt)

        # Make the lines thick
        kernel = np.ones((2, 2), np.uint8)
        img_gt_lane = cv2.dilate(img_gt, kernel, iterations=2)
        ret, thresh = cv2.threshold(cv2.cvtColor(img_gt_lane, cv2.COLOR_BGR2GRAY), 10, 255, 0)
        dst, lines = self.hough_line_detector(img_gt_lane)

        # Compute slope and intercept of all lines.
        slopes = (lines[:, 2] - lines[:, 0]) / (lines[:, 3] - lines[:, 1])
        intercepts = np.mean(np.c_[lines[:, 0], lines[:, 2]], axis=1)
        data = np.c_[slopes, intercepts]

        # Create instance of K-Medians algorithm.
        initial_medians = self.kmedians_centroids
        initial_medians[:, 0] = [np.mean(slopes)] * 3
        kmedians_instance = kmedians(data, initial_medians)

        # Run cluster analysis and obtain results.
        kmedians_instance.process()
        clusters = kmedians_instance.get_clusters()
        medians = np.asarray(kmedians_instance.get_medians())
        median_slope = np.median(medians[:, 0])

        # Kalman filter initialization.
        if self.first_time:
            if len(medians) == 3:
                self.first_time = False
                self.lane_filter.initialize_filter(timestamp.to_sec(), medians)
            else:
                sys.stdout.write("\r\033[31mInitialization requires medians for all lanes\t" )
                sys.stdout.flush()
                return None, None

        # Kalman filter predict/update.
        pred_medians = self.lane_filter.predict_step(timestamp.to_sec())
        pred_medians = pred_medians.reshape(6, 2)[:, 0].reshape(3, 2)
        lanes = self.associate_lanes(medians, pred_medians)
        if not self.first_time:
            medians = self.lane_filter.update_step(lanes[0], lanes[1], lanes[2])
            medians = medians.reshape(6, 2)[:, 0].reshape(3, 2)

        # Visualize lines on image.
        colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
        points_ipm = []
        for (slope, intercept), color in zip(medians, colors):
            x = np.array([0, ipm_img.shape[0] - 1]).astype('int')
            y = ((median_slope * x) + intercept).astype('int')
            points_ipm.append(np.c_[y, x])
            cv2.line(output, (y[0], x[0]), (y[1], x[1]), color, 3, cv2.LINE_AA)
        points_ipm = np.vstack(points_ipm)

        # Convert data to meters.
        points_m = self.ipm.transform_points_to_m(points_ipm, transform=False)
        points_m = points_m.reshape(3, 4)
        slopes_m = (points_m[:, 2] - points_m[:, 0]) / (points_m[:, 3] - points_m[:, 1])
        lane_data_m = np.c_[slopes_m, points_m[:, 0]]

        # Convert points from IPM to image coordinates.
        points_img = self.ipm.transform_points_to_px(points_ipm, inverse=True)
        points_img = points_img.reshape(3, 4)
        points_ipm = points_ipm.reshape(3, 4)

        return output, lane_data_m

    @staticmethod
    def slope_intercept_to_points(lanes):
        x = np.linspace(10, 100, 3)
        y1 = lanes[0, 0] * x + lanes[0, 1]
        y2 = lanes[1, 0] * x + lanes[1, 1]
        y3 = lanes[2, 0] * x + lanes[2, 1]
        y = np.r_[y1, y2, y3]
        points = np.r_[np.c_[x, y1], np.c_[x, y2], np.c_[x, y3]]
        return points


def load_files(folder):
    files = natsorted([file for file in os.listdir(folder) if file.endswith('.jpg')])
    return files


if __name__ == '__main__':
    files = load_files('gt_seg')
    validator = LaneValidator()
    
    for file in files:
        img = cv2.imread('gt_seg/' + file)
        output, lanes = validator.detect_lines(img, time.time())
        points = validator.slope_intercept_to_points(lanes)
        plt.plot(points[0:3, 0], points[0:3, 1], '-k', linewidth=3.0, label='1')
        plt.plot(points[3:6, 0], points[3:6, 1], '-k', linewidth=3.0, label='2')
        plt.plot(points[6:9, 0], points[6:9, 1], '-k', linewidth=3.0, label='3')
        plt.show()
        # cv2.imshow('Output', output)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        break

