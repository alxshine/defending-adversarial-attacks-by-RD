""" setup_mnist.py -- mnist data and model loading code

 Copyright (C) 2016, Nicholas Carlini <nicholas@carlini.com>.

 This program is licenced under the BSD 2-Clause licence,
 contained in the LICENCE file in this directory.
"""

import gzip
import os
import urllib
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


def load_mnist_data():
    """ load MNIST data """
    if not os.path.exists("data"):
        os.mkdir("data")
        files = [
            "train-images-idx3-ubyte.gz", "t10k-images-idx3-ubyte.gz",
            "train-labels-idx1-ubyte.gz", "t10k-labels-idx1-ubyte.gz"
        ]
        for name in files:
            urllib.request.urlretrieve(
                'http://yann.lecun.com/exdb/mnist/' + name, "data/" + name)

    train_data = extract_data("data/train-images-idx3-ubyte.gz", 60000)
    train_labels = extract_labels("data/train-labels-idx1-ubyte.gz", 60000)
    test_data = extract_data("data/t10k-images-idx3-ubyte.gz", 10000)
    test_labels = extract_labels("data/t10k-labels-idx1-ubyte.gz", 10000)

    validation_size = 5000

    validation_data = train_data[:validation_size, :, :, :]
    validation_labels = train_labels[:validation_size]
    train_data = train_data[validation_size:, :, :, :]
    train_labels = train_labels[validation_size:]

    return {
        'validation_data': validation_data,
        'validation_labels': validation_labels,
        'train_data': train_data,
        'train_labels': train_labels,
        'test_data': test_data,
        'test_labels': test_labels
    }


def load_fashion_mnist_data():
    """ load Fashion-MNIST data """
    if not os.path.exists("data/fashion-mnist"):
        os.mkdir("data/fashion-mnist")
        files = [
            "train-images-idx3-ubyte.gz", "t10k-images-idx3-ubyte.gz",
            "train-labels-idx1-ubyte.gz", "t10k-labels-idx1-ubyte.gz"
        ]
        for name in files:

            urllib.request.urlretrieve(
                'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/' +
                name, "data/fashion-mnist/" + name)

    train_data = extract_data("data/fashion-mnist/train-images-idx3-ubyte.gz",
                              60000)
    train_labels = extract_labels(
        "data/fashion-mnist/train-labels-idx1-ubyte.gz", 60000)
    test_data = extract_data("data/fashion-mnist/t10k-images-idx3-ubyte.gz",
                             10000)
    test_labels = extract_labels(
        "data/fashion-mnist/t10k-labels-idx1-ubyte.gz", 10000)

    validation_size = 5000

    validation_data = train_data[:validation_size, :, :, :]
    validation_labels = train_labels[:validation_size]
    train_data = train_data[validation_size:, :, :, :]
    train_labels = train_labels[validation_size:]

    return {
        'validation_data': validation_data,
        'validation_labels': validation_labels,
        'train_data': train_data,
        'train_labels': train_labels,
        'test_data': test_data,
        'test_labels': test_labels
    }


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

    def train(self, data, model_save_file, train_params):
        """ train using supplied parameters
        Arguments:
        data (np.ndarray) : data to train on
        model_save_file (str) : filename where to save trained model
        train_params (dict) : dictionary containing
            ['learning_rate', 'optimizer', 'batch_size', 'num_epochs'],
            which all correspond to keras traning parameters
        Returns:
        the trained keras model
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

        self.model.fit(data['train_data'],
                       data['train_labels'],
                       batch_size=train_params['batch_size'],
                       validation_data=(data['validation_data'],
                                        data['validation_labels']),
                       epochs=train_params['num_epochs'],
                       verbose=2,
                       shuffle=True)

        if model_save_file is not None:
            print("Saving model to {}".format(model_save_file))
            self.model.save(model_save_file)

        return self.model

    def predict(self, data):
        """ return prediction results of underlying model """
        return self.model(data)
