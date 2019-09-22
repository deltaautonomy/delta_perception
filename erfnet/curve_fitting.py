import os
import sys
import glob
import time
import argparse

import cv2
import numpy as np
from natsort import natsorted


class CurveFit:
    def __init__(self, poly_degree=1, thresh=30, horizon=100, color=[0, 0, 255], overlay=False):
        self.poly_degree = poly_degree
        self.thresh = thresh
        self.horizon = horizon
        self.color = color
        self.overlay = overlay

    def fit(self, img):
        # Preprocess
        ret, mask = cv2.threshold(img, self.thresh, 255, cv2.THRESH_BINARY)
        points = cv2.findNonZero(mask)
        if points is None:
            return np.dstack([img] * 3), []

        # Curve fitting
        points = points.squeeze()
        curve = np.polyfit(points[:, 0], points[:, 1], self.poly_degree)
        plotx = np.linspace(0, img.shape[1] - 1, img.shape[1])

        # Compute y-axis points
        if self.poly_degree == 1:
            ploty = curve[0] * plotx + curve[1]
        if self.poly_degree == 2:
            ploty = curve[0] * plotx ** 2 + curve[1] * plotx + curve[2]
        elif self.poly_degree == 3:
            ploty = curve[0] * plotx ** 3 + curve[1] * \
                plotx ** 2 + curve[2] * plotx + curve[3]

        # Clean points
        points = np.vstack(
            [plotx.astype('int'), ploty.round().astype('int')]).T
        # Keep points only below the horizon line
        points = points[points[:, 1] > self.horizon]
        # Keep points within image boundary
        points = points[points[:, 1] < img.shape[0]]

        # Draw
        if self.overlay:
            output = np.dstack([img] * 3)
        else:
            output = np.dstack([np.zeros_like(img)] * 3)
        output[points[:, 1], points[:, 0]] = self.color

        return output, points


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--images-path', '-i', required=True)
    parser.add_argument('--display', '-d', type=int, default=0)
    args = parser.parse_args()

    # Load the test dataset
    images = natsorted([f for f in glob.glob(
        os.path.join(args.images_path, '*.png'))])

    curve_fitter = CurveFit(overlay=True)

    total_frames = 0
    total_time = 0

    print('Processing %d images' % len(images))

    for i, filename in enumerate(images):
        img = cv2.imread(filename, 0)
        start = time.time()
        output = curve_fitter.fit(img)
        end = time.time() - start

        # Benchmark
        total_frames += 1
        total_time += end
        sys.stdout.write('\rAverage FPS: %.3f | Current Time: %.6f %s' %
                         (total_frames/total_time, end, ' ' * 10))
        sys.stdout.flush()

        # Display
        if bool(args.display):
            cv2.imshow('Output', output)
            key = cv2.waitKey(0)
            cv2.destroyAllWindows()
            if key == ord('q'):
                break

    print('\nDone')
