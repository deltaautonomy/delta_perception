# Semantic Segmentation

Perform pixel-wise semantic segmentation on high-resolution images in real-time with Image Cascade Network (ICNet), the highly optimized version of the state-of-the-art Pyramid Scene Parsing Network (PSPNet). **This project implements ICNet and PSPNet50 in Tensorflow with training support for Cityscapes.**

## Setup

1. Make sure you have TensorFlow for GPU installed.
2. Run the following script to install the dependencies.

```
bash setup.sh
```

## Download Weights

Download pre-trained ICNet and PSPNet50 models <a href="docs/model_zoo.md">here</a> and put them in the weights folder. Extract the weights in the same folder as well.

## Inference

Just run the following command. Update the input and results folder in `demo.sh`. The default input image is `testset_01` folder.

```
bash demo.sh
```

## References

1. This implementation is based off of the original ICNet paper proposed by Hengshuang Zhao titled [ICNet for Real-Time Semantic Segmentation on High-Resolution Images](https://arxiv.org/abs/1704.08545). 
2. Some ideas were also taken from their previous PSPNet paper, [Pyramid Scene Parsing Network](https://arxiv.org/abs/1612.01105).
3. The network compression implemented is based on the paper [Pruning Filters for Efficient ConvNets](https://arxiv.org/abs/1608.08710).
