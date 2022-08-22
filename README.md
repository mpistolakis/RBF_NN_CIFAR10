# RBF_NN_CIFAR10

This is an aproach to classify images from CIFAR-10 dataset with Radial Bases Activation Functions. The neural networks that tested has only one hidden layer and the output layer with ten softmax units such as the classes. The Neural networs like these take advantage of the non linear functions and they rely at the distance of the input from the predefined centers.The two base hyperparameters of this neuron except the mathematical expression it is the Sigma and the Center. The current project covers methods that define the centers with different ways: randomly(random data), with K-means and K-nearest Centroid algorithms, it also contains methods to find the possible optimal sigmas for the predifined centers based on variance of the data. There is the choice of training or not the Centers and Sigma during backpropagation. Since the radial base function compute distance is computationally heavy, for this reason the dimensions are reduced with the principal component analysis at dimation of 500, with 98% of initial information.

## Instalation

This repository is tested on Python 3.6.
With pip: install the "requirements.txt"
Download the Dataset from University of Toronto site https://www.cs.toronto.edu/~kriz/cifar.html
