# Vanilla-Neural-Network-From-Scratch
Code for Vanilla NN from scratch in Python 3.8.

Simple Neural Network model (one hidden layer) doing multiclass classification with the softmax function and cross-entropy loss function. The popular MNIST dataset (handwritten digit database) is used for the classification task. The method 'Stochastic gradient descent' (SGD) is used for training, L2-Regularization is taken into account for the Backpropagation step and the activation function used is ReLU.

At the end, the code shows the evaluation obtained in the test dataset (accuracy, precision, recall, f-measure), it evaluate both the average scores for all classes and the scores for each class. Additionally, the code allows you to plot the cost function in each epoch (every time all the samples of the dataset are used in the training step) and, finally, it allows you to visualize the test dataset in a 3D space, this feature use PCA (sklearn) for dimensional reduction.

Note. Ensure that all the files of the MNIST.rar are unpack in the same location that the python file.
