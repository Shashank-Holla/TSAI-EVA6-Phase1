# Advanced Concepts

The objective of this assignment is to design a CNN model for image classification on CIFAR10 dataset. 

The CNN model uses the following convolutional concepts-
1. Dilated convolution with strides has been used instead of MaxPooling to downsample spatial dimension (height and width).
2. Depthwise Seperable convolution to use more channels at lower parameter cost
3. Dilated convolution in one of the layers to increase the receptive field which enables to capture the global view of the images.

Image transformations from Albumentation library is also used to provide means of regularization. 

## Model Hyperparameters

* Optimizer: SGD
* Loss Function: Cross Entropy Loss
* Learning Rate: 0.04
* Batch Size: 128
* Epochs: 36

## Model Summary

```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 16, 32, 32]             432
              ReLU-2           [-1, 16, 32, 32]               0
       BatchNorm2d-3           [-1, 16, 32, 32]              32
            Conv2d-4           [-1, 16, 32, 32]           2,304
              ReLU-5           [-1, 16, 32, 32]               0
       BatchNorm2d-6           [-1, 16, 32, 32]              32
            Conv2d-7           [-1, 16, 16, 16]           2,304
              ReLU-8           [-1, 16, 16, 16]               0
       BatchNorm2d-9           [-1, 16, 16, 16]              32
           Conv2d-10           [-1, 32, 16, 16]           4,608
             ReLU-11           [-1, 32, 16, 16]               0
      BatchNorm2d-12           [-1, 32, 16, 16]              64
           Conv2d-13           [-1, 32, 16, 16]           9,216
             ReLU-14           [-1, 32, 16, 16]               0
      BatchNorm2d-15           [-1, 32, 16, 16]              64
           Conv2d-16           [-1, 32, 16, 16]             288
           Conv2d-17           [-1, 64, 16, 16]           2,048
             ReLU-18           [-1, 64, 16, 16]               0
      BatchNorm2d-19           [-1, 64, 16, 16]             128
           Conv2d-20           [-1, 32, 16, 16]           2,048
           Conv2d-21             [-1, 32, 8, 8]           9,216
             ReLU-22             [-1, 32, 8, 8]               0
      BatchNorm2d-23             [-1, 32, 8, 8]              64
           Conv2d-24             [-1, 64, 8, 8]          18,432
             ReLU-25             [-1, 64, 8, 8]               0
      BatchNorm2d-26             [-1, 64, 8, 8]             128
           Conv2d-27             [-1, 64, 8, 8]             576
           Conv2d-28            [-1, 128, 8, 8]           8,192
             ReLU-29            [-1, 128, 8, 8]               0
      BatchNorm2d-30            [-1, 128, 8, 8]             256
           Conv2d-31             [-1, 32, 8, 8]           4,096
           Conv2d-32             [-1, 32, 4, 4]           9,216
             ReLU-33             [-1, 32, 4, 4]               0
      BatchNorm2d-34             [-1, 32, 4, 4]              64
           Conv2d-35             [-1, 64, 4, 4]          18,432
             ReLU-36             [-1, 64, 4, 4]               0
      BatchNorm2d-37             [-1, 64, 4, 4]             128
           Conv2d-38             [-1, 10, 4, 4]           5,760
             ReLU-39             [-1, 10, 4, 4]               0
      BatchNorm2d-40             [-1, 10, 4, 4]              20
AdaptiveAvgPool2d-41             [-1, 10, 1, 1]               0
================================================================
Total params: 98,180
Trainable params: 98,180
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.01
Forward/backward pass size (MB): 2.13
Params size (MB): 0.37
Estimated Total Size (MB): 2.52
----------------------------------------------------------------
```

## Results

### Training loss/accuracy trend across normalization vs epochs

![](images/accuracy_loss.png)

### Misclassified Images

Below are the list of missclassifed images with their correct and predicted labels.

![](images/misclassified.png)
