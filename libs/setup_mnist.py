""" setup_mnist.py -- mnist data and model loading code

 Copyright (C) 2016, Nicholas Carlini <nicholas@carlini.com>.

 This program is licenced under the BSD 2-Clause licence,
 contained in the LICENCE file in this directory.
"""

import gzip
import numpy as np

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD, Adam


def extract_data(filename, num_images):
    """ extract image data from .tar.gz archive """

    with gzip.open(filename) as bytestream:
        bytestream.read(16)
        buf = bytestream.read(num_images * 28 * 28)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        data = (data / 255) - 0.5
        data = data.reshape(num_images, 28, 28, 1)
        return data


def extract_labels(filename, num_images):
    """ extract labels from .tar.gz archive """
    with gzip.open(filename) as bytestream:
        bytestream.read(8)
        buf = bytestream.read(1 * num_images)
        labels = np.frombuffer(buf, dtype=np.uint8)
    return (np.arange(10) == labels[:, None]).astype(np.float32)


class MNISTModelAllLayers:
    """ Single MNIST channel """
    def __init__(self, layer_sizes=None, init=None):
        self.train_temp = 1

        self.num_channels = 1
        self.image_size = 28
        self.num_labels = 10

        model = Sequential()

        if layer_sizes is None:
            layer_sizes = [32, 32, 64, 64, 200, 200]

        model.add(Conv2D(layer_sizes[0], (3, 3), input_shape=(28, 28, 1)))
        model.add(Activation('relu'))
        model.add(Conv2D(layer_sizes[1], (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(layer_sizes[2], (3, 3)))
        model.add(Activation('relu'))
        model.add(Conv2D(layer_sizes[3], (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Flatten())
        model.add(Dense(layer_sizes[4]))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(layer_sizes[5]))
        model.add(Activation('relu'))
        model.add(Dense(10))

        if init is not None:
            model.load_weights(init)

        self.model = model

    def loss_function(self, correct, predicted):
        """ loss function using softmax cross-entropy"""
        return tf.nn.softmax_cross_entropy_with_logits(logits=predicted /
                                                       self.train_temp,
                                                       labels=correct)

    def train(self, data, file, train_params):
        """ train using supplied parameters
        Arguments:
        train_params (dict) : dictionary containing
            ['learning_rate', 'optimizer', 'batch_size', 'num_epochs'],
            which all correspond to keras traning parameters
 """

        if train_params['optimizer'] == "sgd":
            opt = SGD(lr=train_params['learning_rate'],
                      decay=1e-6,
                      momentum=0.9,
                      nesterov=True)
        elif train_params['optimizer'] == "adam":
            opt = Adam(lr=train_params['learning_rate'])

        self.model.compile(loss=self.loss_function,
                           optimizer=opt,
                           metrics=['accuracy'])

        self.model.fit(data.train_data,
                       data.train_labels,
                       batch_size=train_params['batch_size'],
                       validation_data=(data.validation_data,
                                        data.validation_labels),
                       epochs=train_params['num_epochs'],
                       verbose=2,
                       shuffle=True)

        if file is not None:
            self.model.save(file)

        return self.model

    def predict(self, data):
        """ return prediction results of underlying model """
        return self.model(data)
