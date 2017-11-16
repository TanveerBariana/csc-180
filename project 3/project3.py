# -*- coding: utf-8 -*-

""" Convolutional Neural Network for MNIST dataset classification task.

References:
    Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner. "Gradient-based
    learning applied to document recognition." Proceedings of the IEEE,
    86(11):2278-2324, November 1998.

Links:
    [MNIST Dataset] http://yann.lecun.com/exdb/mnist/

"""

from __future__ import division, print_function, absolute_import

import tflearn
import scipy
import numpy as nm
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression
from tflearn.data_utils import image_preloader
from tflearn.metrics import Accuracy
acc = Accuracy()
# Data loading and preprocessing
# tflearn.datasets.mnist as mnist
#X, Y, testX, testY = mnist.load_data(one_hot=True)
X , Y = image_preloader(target_path='C:\\Users\\tanve\\Desktop\\csc180\\project 3\\train', image_shape=(100, 100), mode='folder', grayscale=False, categorical_labels=True, normalize=True)
X = nm.reshape(X, (-1, 100, 100, 3))
W, Z = image_preloader(target_path='C:\\Users\\tanve\\Desktop\\csc180\\project 3\\test', image_shape=(100, 100), mode='folder', grayscale=False, categorical_labels=True, normalize=True)
W = nm.reshape(W,(-1, 100, 100, 3))

# Building convolutional network
network = input_data(shape=[None, 100, 100, 3], name='input')
network = conv_2d(network, 64, 3, activation='relu', regularizer="L2")
network = max_pool_2d(network, 2, strides = 2)
network = local_response_normalization(network)
network = conv_2d(network, 64, 3, activation='relu', regularizer="L2")
network = max_pool_2d(network, 2, strides = 2)
network = local_response_normalization(network)
network = fully_connected(network, 1024, activation='tanh')
network = dropout(network, 0.8)
network = fully_connected(network, 1024, activation='tanh')
network = dropout(network, 0.8)
network = fully_connected(network, 2, activation='softmax')
network = regression(network, optimizer='momentum', learning_rate=0.01,
                     loss='categorical_crossentropy', name='target', )

# Training
model = tflearn.DNN(network, tensorboard_verbose=0)
model.fit({'input': X}, {'target': Y}, n_epoch=20,
           validation_set=({'input': W}, {'target': Z}),
           snapshot_step=100, show_metric=True, run_id='convnet_mnist')

