#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Author  : Heethesh Vhavle
Email   : heethesh@cmu.edu
Version : 1.0.0
Date    : Sep 18, 2019
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
import torch
import torch.backends.cudnn as cudnn
from scipy.spatial import distance
from scipy.optimize import linear_sum_assignment
from pyclustering.cluster.kmedians import kmedians

# Local python modules
from erfnet.models import ERFNet
from erfnet.lane_filter import LaneKalmanFilter
from ipm.ipm import InversePerspectiveMapping


class ERFNetLaneDetector:
    def __init__(self, weights_path=None):
        if weights_path is None: self.weights_path = osp.join(PKG_PATH, 'erfnet/trained/ERFNet_trained.tar')
        else: self.weights_path = weights_path

        # Network properties.
        self.num_classes = 5
        self.img_height = 720
        self.img_width = 1280
        self.net_height = 208
        self.net_width = 976

        # Pre-processing.
        self.input_mean = np.asarray([103.939, 116.779, 123.68])
        self.input_std = np.asarray([1, 1, 1])
        self.top_start = 100
        self.bot_start = 150
        self.top_slice = slice(self.top_start, self.top_start + self.net_height)
        self.bot_slice = slice(self.bot_start, self.bot_start + self.net_height)

        # Post-processing.
        self.lane_exist_thershold = 0.3
        self.ipm = InversePerspectiveMapping()
        self.kmedians_centroids = np.asarray([[0, 30], [0, 90], [0, 130]], dtype=np.float64)
        self.first_time = True
        self.lane_filter = LaneKalmanFilter()

    def setup(self):
        self.model = ERFNet(self.num_classes)

        # Load pretrained weights.
        pretrained_dict = torch.load(self.weights_path)['state_dict']
        model_dict = self.model.state_dict()

        for layer in model_dict.keys():
            if layer.endswith('num_batches_tracked'): continue
            model_dict[layer] = pretrained_dict['module.' + layer]
        self.model.load_state_dict(model_dict)

        self.model = self.model.cuda()
        self.model.eval()

        cudnn.benchmark = True
        cudnn.fastest = True

    def run(self, img, timestamp):
        # Preprocess.
        inputs = self.preprocess(img.copy())

        # Detect lane markings.
        with torch.no_grad():
            output, output_exist = self.model(inputs)
            output = torch.softmax(output, dim=1)
            lane_maps = output.data.cpu().numpy()
            lane_exist = output_exist.data.cpu().numpy()

        # Postprocess.
        postprocessed_map = self.postprocess(lane_maps, lane_exist)
        output, lanes = self.occupancy_map(postprocessed_map, img.copy(), timestamp)
        return output, lanes

    def preprocess(self, img):
        # Resizing to network width.
        img = cv2.resize(img, None, fx=self.net_width / self.img_width,
            fy=self.net_width / self.img_width, interpolation=cv2.INTER_LINEAR)

        # Normalization.
        img = img - self.input_mean[np.newaxis, np.newaxis, ...]
        img = img / self.input_std[np.newaxis, np.newaxis, ...]

        # Cropping to network size.
        img_top = img[self.top_slice].copy()
        img_bot = img[self.bot_slice].copy()

        # Convert to tensor.
        inputs = np.asarray([img_top, img_bot])
        inputs = torch.from_numpy(inputs).permute(0, 3, 1, 2).contiguous().float().cuda()
        # inputs = torch.from_numpy(img_bot).permute(2, 0, 1).contiguous().float().unsqueeze(0).cuda()
        return inputs

    def postprocess(self, lane_maps, lane_exist, best_prob=True):
        # Create blank image.
        output_map = np.zeros((int(self.img_height * self.net_width / self.img_width), self.net_width))

        # Use lane maps with exist probability > threshold.
        if best_prob:
            for i in range(1, 5):
                if lane_exist[0][i - 1] >= self.lane_exist_thershold:
                    output_map[self.top_slice] += lane_maps[0][i]
                if lane_exist[1][i - 1] >= self.lane_exist_thershold:
                    output_map[self.bot_slice] += lane_maps[1][i]
        # Use background map.
        else:
            output_map[self.top_slice] += 1 - lane_maps[0][0]
            output_map[self.bot_slice] += 1 - lane_maps[1][0]

        # Avoid overflow.
        output_map = np.clip(output_map, 0, 1)

        # Resize back to original image size.
        output_map = cv2.resize(output_map, None, fx=self.img_width / self.net_width,
            fy=self.img_width / self.net_width, interpolation=cv2.INTER_LINEAR)

        return np.array(output_map * 255, dtype=np.uint8)

    def associate_lanes(self, medians, pred_medians):
        association = [None] * 3
        cost = distance.cdist(np.expand_dims(pred_medians[:, 1], 1), np.expand_dims(medians[:, 1], 1))
        idx_a, idx_b = linear_sum_assignment(cost)
        for i, j in zip(idx_a, idx_b): association[i] = medians[j]
        return association

    def occupancy_map(self, lane_map, img, timestamp):
        # return lane_map, None
        # Transform images to BEV.
        ipm_img = self.ipm.transform_image(img)
        ipm_lane = self.ipm.transform_image(lane_map)
        output = ipm_img # np.zeros_like(ipm_img, dtype=np.uint8)
        # output = img # np.zeros_like(ipm_img, dtype=np.uint8)

        # Find all lines (n * [x1, y1, x2, y2]).
        dst, lines, n_points = self.hough_line_detector(ipm_lane, ipm_img)
        if lines is None: return output, None

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

        # Draw polygons on lanes.
        poly_img = np.zeros_like(output, dtype=np.uint8)
        for i in range(len(points_ipm) - 1):
            poly_points = np.vstack([points_ipm[i].reshape(2, 2), points_ipm[i + 1].reshape(2, 2)])
            poly_points = cv2.convexHull(poly_points.astype('int'))
            cv2.fillConvexPoly(poly_img, poly_points, colors[i])

        output = cv2.addWeighted(poly_img, 0.5, output, 1.0, 0)
        return output, lane_data_m

    def hough_line_detector(self, ipm_lane, ipm_img):
        canny_lane = cv2.Canny(ipm_lane, 50, 200, None, 3)
        canny_ipm = cv2.Canny(ipm_img, 50, 200, None, 3)

        # Copy edges to the images that will display the results in BGR.
        dst = cv2.cvtColor(canny_lane, cv2.COLOR_GRAY2BGR)
        # dst = cv2.cvtColor(canny_ipm, cv2.COLOR_GRAY2BGR)

        # Probabilistic hough lines detector.
        lines_lanes = cv2.HoughLinesP(canny_lane, 1, np.pi / 180, 50, None, 50, 10)
        if lines_lanes is not None:
            lines_lanes = lines_lanes.squeeze(1)
            for i in range(0, len(lines_lanes)):
                l = lines_lanes[i]
                cv2.line(dst, (l[0], l[1]), (l[2], l[3]), (0, 255, 255), 1, cv2.LINE_AA)

        lines_ipm = cv2.HoughLinesP(canny_ipm, 1, np.pi / 180, 50, None, 150, 10)
        if lines_ipm is not None:
            lines_ipm = lines_ipm.squeeze(1)
            for i in range(0, len(lines_ipm)):
                l = lines_ipm[i]
                cv2.line(dst, (l[0], l[1]), (l[2], l[3]), (255, 255, 0), 1, cv2.LINE_AA)

        # Handle no line detections.
        if lines_lanes is None and lines_ipm is None: return dst, None, (0, 0)
        elif lines_lanes is None: return dst, lines_ipm, (0, len(lines_ipm))
        elif lines_ipm is None: return dst, lines_lanes, (len(lines_lanes), 0)
        else: return dst, np.r_[lines_lanes, lines_ipm], (len(lines_lanes), len(lines_ipm))

    def close(self):
        pass


if __name__ == '__main__':
    erfnet = ERFNetLaneDetector()
    # erfnet.setup()

    # cv2.namedWindow('Test Output', cv2.WINDOW_NORMAL)
    # cv2.resizeWindow('Test Output', (640, 360))
    # cv2.moveWindow('Test Output', 100, 300)

    # img = cv2.imread(sys.argv[1])
    # output = erfnet.run(img)

    # cv2.imshow('Test Output', output)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # Test lane association
    # medians = np.array([[0, 30], [0, 110], [0, 60]])
    medians = np.array([[0, 30], [0, 140]])
    pred_medians = np.array([[0, 30], [0, 60], [0, 130]])
    print(erfnet.associate_lanes(medians, pred_medians))
