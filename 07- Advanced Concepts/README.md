# Advanced Concepts

The objective of this assignment is to design a CNN model for image classification on CIFAR10 dataset. 

The CNN model uses the following convolutional concepts-
1. Dilated convolution with strides has been used instead of MaxPooling to downsample spatial dimension (height and width).
2. Depthwise Seperable convolution to use more channels at lower parameter cost
3. Dilated convolution in one of the layers to increase the receptive field which enables to capture the global view of the images.

Image transformations from Albumentation library is also used to provide means of regularization. 

## Model Hyperparameters

Optimizer: SGD
Loss Function: Cross Entropy Loss
Learning Rate: 0.04
Batch Size: 128
Epochs: 36

## Model Summary

## Results

### Training loss/accuracy trend across normalization vs epochs

### Missclassified Images

Below are the list of missclassifed images with their correct and predicted labels
