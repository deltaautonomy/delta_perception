#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Author  : Heethesh Vhavle
Email   : heethesh@cmu.edu
Version : 1.0.0
Date    : Apr 07, 2019

References:
https://github.com/AlexeyAB/darknet
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

# Built-in modules
from ctypes import *

# Local python modules
import darknet.darknet as darknet

# Global variables
DARKNET_PATH = osp.join(PKG_PATH, 'darknet')

# Handle package path and COCO names path
print('Darknet Package:', DARKNET_PATH)
add_path(DARKNET_PATH)
darknet.fix_coco_names_path(DARKNET_PATH)


class YOLO:
    def __init__(self, configPath=None, weightPath=None, metaPath=None, thresh=0.25):   
        # YOLO parameters
        self.thresh = thresh

        # Set paths
        if configPath is None: self.configPath = osp.join(DARKNET_PATH, 'cfg/yolov3.cfg')
        if not osp.exists(self.configPath):
            raise ValueError('Invalid config path `' + osp.abspath(self.configPath) + '`')

        if weightPath is None: self.weightPath = osp.join(DARKNET_PATH, 'weights/yolov3.weights')
        if not osp.exists(self.weightPath):
            raise ValueError('Invalid weight path `' + osp.abspath(self.weightPath) + '`')

        if metaPath is None: self.metaPath = osp.join(DARKNET_PATH, 'cfg/coco.data')
        if not osp.exists(self.metaPath):
            raise ValueError('Invalid data file path `' + osp.abspath(self.metaPath) + '`')
        
    def setup(self, netMain=None, metaMain=None, altNames=None):
        # Load the model
        if netMain is None:
            self.netMain = darknet.load_net_custom(self.configPath.encode(
                'ascii'), self.weightPath.encode('ascii'), 0, 1) # batch size = 1
        
        if metaMain is None:
            self.metaMain = darknet.load_meta(self.metaPath.encode('ascii'))
        
        if altNames is None:
            try:
                with open(self.metaPath) as metaFH:
                    metaContents = metaFH.read()
                    import re
                    match = re.search('names *= *(.*)$', metaContents, re.IGNORECASE | re.MULTILINE)
                    if match: result = match.group(1)
                    else: result = None
                    try:
                        if osp.exists(result):
                            with open(result) as namesFH:
                                namesList = namesFH.read().strip().split('\n')
                                self.altNames = [x.strip() for x in namesList]
                    except TypeError:
                        pass
            except Exception:
                pass

        # Create an image we reuse for each detect
        self.darknet_image = darknet.make_image(darknet.network_width(self.netMain),
            darknet.network_height(self.netMain), 3)

        # Performance logging
        self.count = 0
        self.total = 0

    def run(self, frame):
        # Preprocess image
        frame_size = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (darknet.network_width(self.netMain),
            darknet.network_height(self.netMain)), interpolation=cv2.INTER_LINEAR)

        # Forward pass
        darknet.copy_image_from_bytes(self.darknet_image, frame_resized.tobytes())
        detections = darknet.detect_image(self.netMain, self.metaMain, self.darknet_image, frame_size, thresh=self.thresh)

        return detections, frame_resized

    @staticmethod
    def cvDrawBoxes(detections, img):
        for detection in detections:
            xmin, ymin, xmax, ymax = detection[2][0],\
                                     detection[2][1],\
                                     detection[2][2],\
                                     detection[2][3]
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 1)
            cv2.putText(img,
                        detection[0].decode() +
                        ' [' + str(round(detection[1] * 100, 2)) + '%]',
                        (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        [0, 255, 0], 2)
        return img


if __name__ == '__main__':
    yolo = YOLO()
    yolo.setup()

    cap = cv2.VideoCapture(osp.join(DARKNET_PATH, 'mission_impossible.mp4'))
    cap.set(3, 1280)
    cap.set(4, 720)

    while True:
        ret, frame = cap.read()
        if not ret:
            cap.release()
            break
        yolo.run(frame)
