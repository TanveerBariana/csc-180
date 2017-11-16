import neuralNet as net
import numpy as np
from tflearn.data_utils import image_preloader
model = net.model
X , Y = image_preloader(target_path='C:\\Users\\tanve\\Desktop\\csc180\\project 3\\train', image_shape=(100, 100), mode='folder', grayscale=False, categorical_labels=True, normalize=True)
X = np.reshape(X, (-1, 100, 100, 3))

W, Z = image_preloader(target_path='C:\\Users\\tanve\\Desktop\\csc180\\project 3\\test', image_shape=(100, 100), mode='folder', grayscale=False, categorical_labels=True, normalize=True)
W = np.reshape(W, (-1, 100, 100, 3))
model.fit(X, Y, n_epoch=250, validation_set=(W,Z), show_metric=True)
model.save('./ZtrainedNet/final-model.tfl')