# Vision Transformers

The objective of this assignment is to train Image classification model on Cats and Dogs dataset using Vision Transformers.

Breakdown of Vision Transformers: [Break down of Vision Transformers](https://github.com/Shashank-Holla/TSAI-EVA6-Phase1/blob/main/13-%20Vision%20Transformers/Vision%20Transformers-%20breakdown/README.md)

## Model Hyperparameters

* Optimizer: Adam
* Loss function: Cross Entropy Loss
* Learning rate: 3e-5
* Batch size: 64
* Epochs: 20

Linear Time Transformer (Linformer)

* Dimensions: 128
* Number of heads: 8
* Depth: 12

## Method of training

The key mechanism and potential bottleneck in standard Transformer models is the model's Self attention. In the self-attention mechanism, the representation of each token is updated by visiting all the other tokens that are present in the previous layer. This operation is essential for retaining long-term information. 

Linformer is a Transformer architecture for tackling the self-attention bottleneck in Transformers. With standard Transformers, the amount of required processing power increases at a geometric rate as the input length increases. With Linformer, however, the number of computations increases only at a linear rate.

## Training logs

Epoch 5 to 10: 

Epoch : 5 - loss : 0.6762 - acc: 0.5736 - val_loss : 0.6739 - val_acc: 0.5787

100%
313/313 [02:05<00:00, 2.49it/s]

Epoch : 6 - loss : 0.6710 - acc: 0.5840 - val_loss : 0.6611 - val_acc: 0.6054

100%
313/313 [02:30<00:00, 2.08it/s]

Epoch : 7 - loss : 0.6648 - acc: 0.5944 - val_loss : 0.6528 - val_acc: 0.6044

100%
313/313 [02:05<00:00, 2.49it/s]

Epoch : 8 - loss : 0.6532 - acc: 0.6042 - val_loss : 0.6582 - val_acc: 0.6080

100%
313/313 [03:13<00:00, 1.61it/s]

Epoch : 9 - loss : 0.6500 - acc: 0.6096 - val_loss : 0.6423 - val_acc: 0.6114

100%
313/313 [02:05<00:00, 2.49it/s]

Epoch : 10 - loss : 0.6389 - acc: 0.6211 - val_loss : 0.6301 - val_acc: 0.6392

