# -*- coding: utf-8 -*-
# @Author: Heethesh Vhavle
# @Date:   Sep 23, 2019
# @Last Modified by:   Heethesh Vhavle
# @Last Modified time: Sep 23, 2019

import numpy as np
import matplotlib.pyplot as plt
from pyclustering.cluster.kmedians import kmedians
from pyclustering.cluster import cluster_visualizer

lane_lines = [[[ 78, 498,  80, 570]],
 [[136, 682, 136, 479]],
 [[137, 827, 137, 683]],
 [[ 30, 622,  32, 687]],
 [[131, 677, 136, 827]],
 [[ 89, 664,  89, 720]],
 [[ 80, 549,  82, 654]],
 [[ 89, 721,  90, 780]],
 [[ 41, 718,  41, 599]],
 [[127, 501, 127, 434]],
 [[ 40, 804,  40, 707]],
 [[130, 626, 131, 676]],
 [[135, 478, 135, 413]]]

ipm_lines = [[[139, 629, 141, 800]],
 [[ 27, 553,  32, 804]],
 [[106,   0, 108, 156]],
 [[ 24, 271,  26, 431]],
 [[136, 430, 139, 628]]]

lane_lines = np.asarray(lane_lines).squeeze(1)
ipm_lines = np.asarray(ipm_lines).squeeze(1)
all_lines = np.r_[lane_lines, ipm_lines]

diff_x = np.abs(all_lines[:, 3] - all_lines[:, 1])
diff_y = np.abs(all_lines[:, 2] - all_lines[:, 0])

angles = np.arctan2(diff_y, diff_x)
y_avg = np.mean(np.c_[all_lines[:, 0], all_lines[:, 2]], axis=1)
data = np.c_[y_avg, angles]

# Create instance of K-Medians algorithm.
initial_medians = np.asarray([[30, 0], [90, 0], [130, 0]])
kmedians_instance = kmedians(data, initial_medians)

# Run cluster analysis and obtain results.
kmedians_instance.process()
clusters = kmedians_instance.get_clusters()
medians = np.asarray(kmedians_instance.get_medians())
median_slope = np.median(medians[:, 1])

# Visualize lines on image.
img = np.zeros((200, 800))
plt.imshow(img)
for intercept, slope in medians:
    x = np.array([0, 799])
    y = (median_slope * x) + intercept
    plt.plot(x, y)
plt.show()
