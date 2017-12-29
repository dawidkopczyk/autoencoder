# autoencoder

This repository is a quick tutorial to Denoising Autoencoders (DAE) using Keras.

The examples are based on MNIST dataset:
https://www.tensorflow.org/get_started/mnist/beginners#the_mnist_data

The repository contains two scripts:
- dae_mnist - denoising autoencoder using convolutional neural networks (CNN) to encode and decode mnist images.
- dea_predict_mnist - denoising autoencoder using deep neural networks (DNN) to represent the mnist images. Then, the activation values are used as features in another deep neural network to predict handwritten digits. 

