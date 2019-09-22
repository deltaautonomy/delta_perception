import os
import sys
import glob
import time
import argparse

import cv2
import numpy as np
from natsort import natsorted

from tools.curve_fitting import CurveFit


class LanePolygon:
    def __init__(self, poly_degree=1, prob_thresh=30, horizon=110):
        self.curve_fitter = CurveFit(poly_degree, prob_thresh, horizon,
                                     color=[255, 255, 255], overlay=False)
        self.lane_colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]

    def fit_polygons(self, all_line_points, shape):
        output = np.dstack([np.zeros(shape)] * 3)

        # Draw polygons
        for i in range(len(all_line_points) - 1):
            # Add all points with corner points for left and right lanes
            points = np.vstack([all_line_points[i], all_line_points[i + 1]])
            if i == 0:
                points = np.vstack([points, (0, shape[0])])
            if i == 2:
                points = np.vstack([points, (shape[1], shape[0])])

            # Convert points list to convex hull to avoid criss-crossing
            points = cv2.convexHull(points)

            # Draw the polygons
            cv2.fillConvexPoly(output, points, self.lane_colors[i])

        return cv2.normalize(output, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--images-path', '-i', required=True)
    parser.add_argument('--lanes-path', '-l', required=True)
    parser.add_argument('--display', '-d', type=int, default=0)
    args = parser.parse_args()

    # Load the test dataset
    bases = natsorted([f.split('.exist')[0]
                       for f in glob.glob(os.path.join(args.lanes_path, '*.txt'))])
    images = natsorted([f for f in glob.glob(
        os.path.join(args.images_path, '*.jpg'))])

    lane_poly = LanePolygon()

    print('Processing %d images' % len(images))

    for i, basename in enumerate(bases):
        print('Processing: %s...%s' % (basename[:20], basename[-20:]))
        lanes = natsorted([f for f in glob.glob(
            os.path.join(args.lanes_path, '%s*.png' % basename))])

        all_line_maps = np.dstack([np.zeros_like(cv2.imread(lanes[0], 0))] * 3)
        all_line_points = []

        # Fit lines
        for j, filename in enumerate(lanes):
            img = cv2.imread(filename, 0)
            line_map, line_points = lane_poly.curve_fitter.fit(img)
            if len(line_points):
                all_line_points.append(line_points)
            all_line_maps = cv2.addWeighted(
                all_line_maps, 1.0, line_map, 1.0, 0)

        # Find and draw polygons
        raw_image = cv2.imread(images[i])
        poly_image = lane_poly.fit_polygons(all_line_points, img.shape)
        output = cv2.addWeighted(poly_image, 1.0, raw_image, 1.0, 0)

        # Display
        if bool(args.display):
            cv2.imshow('Output', output)
            cv2.imwrite('lane_test.jpg', output)
            key = cv2.waitKey(0)
            cv2.destroyAllWindows()
            if key == ord('q'):
                break

    print('\nDone')
