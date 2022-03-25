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

### FashionMNIST

We use a model consisting of three convolutional layers of
64, 128 and 128 3x3 filters, with a fully connected layer of
256 hidden units on top. The embedding size is 256. After
every convolutional layer, we perform batch normalization
and a 2x2 max pooling operation.
We use the SGD optimizer with learning rate 0.05 (decayed
by a factor of 5 every 10 epochs), momentum 0.9, weight
decay 100 4
and train for a set 30 epochs. The centroid
updates are done with γ = 0.999. The output dimension of
the model, d is 256, and we use the same value for the size
of the centroids, n.
We normalise our data using per channel mean and standard
deviation, as computed on the training set. The validation
set contains 5000 elements, removed at random from the
full 60,000 elements in the training set. For the final results,
we rerun on the full training set with the final set of hyper
parameters.

### CIFAR-10

## Running
 