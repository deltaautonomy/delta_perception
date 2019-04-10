#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Author  : Heethesh Vhavle
Email   : heethesh@cmu.edu
Version : 1.0.0
Date    : Apr 07, 2019
'''

# Handle paths and OpenCV import
from init_paths import *

########################### Darknet ###########################

# Test packages
from darknet.darknet_video import YOLO

def test_yolo():
    yolo = YOLO()
    yolo.setup()

    cap = cv2.VideoCapture('../darknet/mission_impossible.mp4')
    cap.set(3, 1280)
    cap.set(4, 720)

    while True:
        ret, frame = cap.read()
        if not ret:
            cap.release()
            break
        yolo.run(frame)

########################### Driver ###########################

if __name__ == '__main__':
    test_yolo()
