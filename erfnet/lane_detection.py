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
import torchvision
import torch.nn.parallel
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torchvision import transforms

# Local python modules
from erfnet.models import ERFNet


class ERFNetInference:
    def __init__(self, weights_path=None):
        if weights_path is None: self.weights_path = osp.join(PKG_PATH, 'erfnet/trained/ERFNet_trained.tar')
        else: self.weights_path = weights_path

        # Network properties
        self.num_classes = 5
        self.img_height = 208
        self.img_width = 976
        self.dims = (self.img_height, self.img_width)

        # Preprocessing
        self.input_mean = np.asarray([103.939, 116.779, 123.68])
        self.input_std = np.asarray([1, 1, 1])
        self.crop_offset = (0, 0)
        

    def setup(self):
        self.model = ERFNet(self.num_classes)

        # Load pretrained weights
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

    def run(self, img):
        inputs = self.preprocess(img)
        output, output_exist = self.model(inputs)
        output = torch.softmax(output, dim=1)
        lane_maps = output.data.cpu().numpy().squeeze(0)
        lane_exist = output_exist.data.cpu().numpy()
        return self.postprocess(lane_maps, lane_exist)

    def preprocess(self, img):
        # Resizing to network height
        img = cv2.resize(img, None, fx=self.dims[1] / img.shape[1],
            fy=self.dims[1] / img.shape[1], interpolation=cv2.INTER_LINEAR)

        # Normalization
        img = img - self.input_mean[np.newaxis, np.newaxis, ...]
        img = img / self.input_std[np.newaxis, np.newaxis, ...]

        # Cropping to network size
        diff = int(abs(img.shape[0] - self.dims[0]) / 2) + self.crop_offset[0]
        img = img[diff:diff + self.dims[0]]

        # Convert to tensor
        inputs = torch.from_numpy(img).permute(2, 0, 1).contiguous().float().cuda()
        inputs = torch.unsqueeze(inputs, 0)
        # todo(heethesh): Perform top/bottom stacking
        # inputs = torch.cat((inputs, inputs), 0)
        return inputs

    def postprocess(self, lane_maps, lane_exist):
        return lane_maps[0]

    def close(self):
        pass


if __name__ == '__main__':
    erfnet = ERFNetInference()
    erfnet.setup()

    cv2.namedWindow('Test Output', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Test Output', (640, 360))
    cv2.moveWindow('Test Output', 100, 300)

    img = cv2.imread(sys.argv[1])
    output = erfnet.run(img)
    
    cv2.imshow('Test Output', output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
