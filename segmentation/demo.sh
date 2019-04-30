#!/bin/bash

MODEL="0818_icnet_0.5_1025_resnet_v1"
# MODEL="0818_icnet_1.0_1025_resnet_v1"
# MODEL="0818_pspnet_1.0_713_resnet_v1"

python3 inference.py \
    --input_shape 1024,2048,3 \
    --pad_to_shape 1025,2049 \
    --input_path ../dataset/testset_01/images \
    --config_path weights/${MODEL}/pipeline.config \
    --trained_checkpoint weights/${MODEL}/model.ckpt \
    --output_dir results \
