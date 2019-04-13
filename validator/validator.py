#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Author  : Heethesh Vhavle
Email   : heethesh@cmu.edu
Version : 1.0.0
Date    : Apr 13, 2019
'''

# Python 2/3 compatibility
from __future__ import print_function, absolute_import, division

# Standalone
if __name__ == '__main__':
    import sys
    sys.path.append('..')

    # Handle paths and OpenCV import
    from scripts.init_paths import *
    from scripts.utils import *

# Run from scripts module
else:
    # Handle paths and OpenCV import
    from init_paths import *
    from utils import *


class ObjectDetectionValidator:
    def __init__(self, path):
        if not osp.exists(path):
            os.makedirs(osp.join(path, 'ground-truth'))
            os.makedirs(osp.join(path, 'detection-results'))

    def display_results(self):
        import validator.calculate_map
