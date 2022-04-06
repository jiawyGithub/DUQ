# Deterministic Uncertainty Quantification (DUQ)

@article{van2020uncertainty,
  title={Uncertainty Estimation Using a Single Deep Deterministic Neural Network},
  author={van Amersfoort, Joost and Smith, Lewis and Teh, Yee Whye and Gal, Yarin},
  booktitle={International Conference on Machine Learning},
  year={2020}
}
官方代码：https://github.com/y0ast/deterministic-uncertainty-quantification

## Dependencies

The code is based on PyTorch and requires a few further dependencies, listed in [environment.yml](environment.yml).

## Datasets and Model

### MNIST & FashionMNIST

model: three convolutional layers
64, 128 and 128 3x3 filters, with a fully connected layer of 256 hidden units on top. 
After every convolutional layer, we perform batch normalization and a 2x2 max pooling operation.

embedding size: 256. 

SGD optimizer, learning rate = 0.05 (decayed by a factor of 5 every 10 epochs), momentum = 0.9, weight decay = 1e-4

train: 30 epochs. 
The centroid updates are done with γ = 0.999. 
The output dimension of the model, d is 256, and we use the same value for the size of the centroids, n.

We normalise our data using per channel mean and standard deviation, as computed on the training set. 
The validation set contains 5000 elements, removed at random from the full 60,000 elements in the training set. 
For the final results, we rerun on the full training set with the final set of hyper parameters.

### CIFAR-10

We use a ResNet-18, as implement in torchvision version 0.4.24. 
We make the following modifications: the first convolutional layer is changed to have 64 3x3 filters with stride 1, the first pooling layer is skipped and the last linear layer is changed to be 512 by 512.

SGD optimizer, learning rate = 0.05 (decayed by a factor 10 every 25 epochs), momentum = 0.9, weight decay = 1e-4

train: 75 epochs. 
The centroid updates are done with γ = 0.999. The output dimension of the model, d is 512, and we use the same value for the size of the centroids n.

We normalise our data using per channel mean and standard deviation, as computed on the training set. 
We augment the data at training time using random horizontal flips (with probability 0.5) and random crops after padding 4 zero pixels on all sides. 
The validation set contains 10,000 elements, removed at random from the full 50,000 elements in the training set. 
For the final results, we rerun on the full training set with the final set of hyper parameters.

## Running
 