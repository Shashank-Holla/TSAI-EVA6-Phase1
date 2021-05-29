# Back propogation and Architectural Basics

## Objective - Part A

Objective of this activity is to design a 2 layer fully connected network with forward pass, Loss calculation, gradient calculation and backward pass.

![](images/NN.JPG)

Shared above is a fully connected layer with 2 inputs (i1 and i2) and has 2 hidden layer (h1-2 and o1-2). Weights (w1, w2, w3 and w4) are weights for the first hidden layer and (w5, w6, w7 and w8) are for the second hidden layer. Sigmoid function is used as the activation function for the hidden layers. Error is calculated by taking the Square of the difference of expected (t1, t2) and actual values (a_o1, a_o2). ETotal is the total error calculated for the forward pass and is the sum of the errors of each branch (E1 and E2).

The objective of the forward and backward pass calculation is to adjust all the weights of the model in such a way that the total error of the model is at its lowest.

### Process

1. Forward Pass

Hidden layer output is calculated by multiplying the inputs with their weights. Activation function-Sigmoid is then applied on the hidden layer's output to get the activated output. The same step is followed for the second hidden layer.

Error for the branch is calculated by taking the square of the difference of expected output and the actual output. Total error for the network is the sum of the errors of the two branches.

![](images/forwardPass.JPG)

2. Backward pass

During backward pass, the contribution towards the total error by each weight is calculated (dE/dw8, dE/dw7..). Backward pass calculation starts from the right in the neural network (layer near total error)and the contribution from all the subsequent weights is calculated using chain rule.

The weights of the neural network are then updated by taking the difference of the weight with product of the delta calculate above and learning rate. Learning rate decides how much the weights can change on each update.

![](images/backwardpass.JPG)

### Results

Shared below is the trend of the Total Error over iterations for different learning rates. In this case, higher learning rate has given the steepest curve.

![](images/ErrorvsIterations.JPG)


## Objective - Part B

Objective is to design a fully convolutional neural network for image classification on MNIST dataset to achieve test accuracy of 99.4% with model's total parameter less than 20K.

### Model's hyperparameters

* Optimizer: SGD
* Loss Function: Cross Entropy Loss
* Learning Rate: 0.04
* Dropout: 0.069
* Batch Size: 128
* Epochs: 20

### Model Summary
```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1            [-1, 8, 28, 28]              72
              ReLU-2            [-1, 8, 28, 28]               0
       BatchNorm2d-3            [-1, 8, 28, 28]              16
           Dropout-4            [-1, 8, 28, 28]               0
            Conv2d-5           [-1, 16, 28, 28]           1,152
              ReLU-6           [-1, 16, 28, 28]               0
       BatchNorm2d-7           [-1, 16, 28, 28]              32
           Dropout-8           [-1, 16, 28, 28]               0
            Conv2d-9           [-1, 16, 28, 28]           2,304
             ReLU-10           [-1, 16, 28, 28]               0
      BatchNorm2d-11           [-1, 16, 28, 28]              32
          Dropout-12           [-1, 16, 28, 28]               0
        MaxPool2d-13           [-1, 16, 14, 14]               0
           Conv2d-14            [-1, 8, 14, 14]             128
           Conv2d-15           [-1, 16, 12, 12]           1,152
             ReLU-16           [-1, 16, 12, 12]               0
      BatchNorm2d-17           [-1, 16, 12, 12]              32
          Dropout-18           [-1, 16, 12, 12]               0
           Conv2d-19           [-1, 16, 10, 10]           2,304
             ReLU-20           [-1, 16, 10, 10]               0
      BatchNorm2d-21           [-1, 16, 10, 10]              32
          Dropout-22           [-1, 16, 10, 10]               0
           Conv2d-23             [-1, 16, 8, 8]           2,304
             ReLU-24             [-1, 16, 8, 8]               0
      BatchNorm2d-25             [-1, 16, 8, 8]              32
          Dropout-26             [-1, 16, 8, 8]               0
           Conv2d-27             [-1, 16, 6, 6]           2,304
             ReLU-28             [-1, 16, 6, 6]               0
      BatchNorm2d-29             [-1, 16, 6, 6]              32
          Dropout-30             [-1, 16, 6, 6]               0
        AvgPool2d-31             [-1, 16, 1, 1]               0
           Conv2d-32             [-1, 10, 1, 1]             160
================================================================
Total params: 12,088
Trainable params: 12,088
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 1.16
Params size (MB): 0.05
Estimated Total Size (MB): 1.21
----------------------------------------------------------------
```
### Test Results

Best test accuracy: 99.49%

Total parameters: 12,088

**Train/Test Loss and accuracy trend versus epochs.**

![](images/Loss&accuracy.png)
