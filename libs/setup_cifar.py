""" setup_cifar.py -- cifar data and model loading code

 Copyright (C) 2016, Nicholas Carlini <nicholas@carlini.com>.

 This program is licenced under the BSD 2-Clause licence,
 contained in the LICENCE file in this directory. """

import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD, Adam


# def load_batch(fpath, label_key='labels'):
#     """  """
#     f = open(fpath, 'rb')
#     d = pickle.load(f, encoding="bytes")
#     for k, v in d.items():
#         del (d[k])
#         d[k.decode("utf8")] = v
#     f.close()
#     data = d["data"]
#     labels = d[label_key]

#     data = data.reshape(data.shape[0], 3, 32, 32)
#     final = np.zeros((data.shape[0], 32, 32, 3), dtype=np.float32)
#     final[:, :, :, 0] = data[:, 0, :, :]
#     final[:, :, :, 1] = data[:, 1, :, :]
#     final[:, :, :, 2] = data[:, 2, :, :]

#     final /= 255
#     final -= .5

#     return final, labels


def load_batch(file_path):
    """ load batch of images and labels from file_path """
    data_file = open(file_path, "rb").read()
    size = 32 * 32 * 3 + 1
    labels = []
    images = []
    for i in range(10000):
        arr = np.fromstring(data_file[i * size:(i + 1) * size], dtype=np.uint8)
        lab = np.identity(10)[arr[0]]
        img = arr[1:].reshape((3, 32, 32)).transpose((1, 2, 0))

        labels.append(lab)
        images.append((img / 255) - .5)
    return np.array(images), np.array(labels)


# class CIFAR:
#     def __init__(self):
#         train_data = []
#         train_labels = []

#         if not os.path.exists("cifar-10-batches-bin"):
#             urllib.request.urlretrieve(
#                 "https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz",
#                 "cifar-data.tar.gz")
#             os.popen("tar -xzf cifar-data.tar.gz").read()

#         for i in range(5):
#             r, s = load_batch("cifar-10-batches-bin/data_batch_" + str(i + 1) +
#                               ".bin")
#             train_data.extend(r)
#             train_labels.extend(s)

#         train_data = np.array(train_data, dtype=np.float32)
#         train_labels = np.array(train_labels)

#         self.test_data, self.test_labels = load_batch(
#             "cifar-10-batches-bin/test_batch.bin")

#         VALIDATION_SIZE = 5000

#         self.validation_data = train_data[:VALIDATION_SIZE, :, :, :]
#         self.validation_labels = train_labels[:VALIDATION_SIZE]
#         self.train_data = train_data[VALIDATION_SIZE:, :, :, :]
#         self.train_labels = train_labels[VALIDATION_SIZE:]


# class CIFARModel:
#     def __init__(self, restore, session=None):
#         self.num_channels = 3
#         self.image_size = 32
#         self.num_labels = 10

#         model = Sequential()

#         model.add(Conv2D(64, (3, 3), input_shape=(32, 32, 3)))
#         model.add(Activation('relu'))
#         model.add(Conv2D(64, (3, 3)))
#         model.add(Activation('relu'))
#         model.add(MaxPooling2D(pool_size=(2, 2)))

#         model.add(Conv2D(128, (3, 3)))
#         model.add(Activation('relu'))
#         model.add(Conv2D(128, (3, 3)))
#         model.add(Activation('relu'))
#         model.add(MaxPooling2D(pool_size=(2, 2)))

#         model.add(Flatten())
#         model.add(Dense(256))
#         model.add(Activation('relu'))
#         model.add(Dense(256))
#         model.add(Activation('relu'))
#         model.add(Dense(10))

#         model.load_weights(restore)

#         self.model = model

#     def predict(self, data):
#         return self.model(data)


class CIFARModelAllLayers:
    """ creates model for a single CIFAR channel """
    def __init__(self,
                 layer_sizes=None,
                 init=None):

        self.train_temp = 1

        self.num_channels = 3
        self.image_size = 32
        self.num_labels = 10

        if layer_sizes is None:
            layer_sizes = [64, 64, 128, 128, 256, 256]

        model = Sequential()
        model.add(Conv2D(layer_sizes[0], (3, 3), input_shape=(32, 32, 3)))
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
        model.add(Dense(layer_sizes[5]))
        model.add(Activation('relu'))
        model.add(Dense(10))

        if init is not None:
            model.load_weights(init)

        self.model = model

    def loss_function(self, correct, predicted):
        """ calculates loss and backpropagation gradients """
        return tf.nn.softmax_cross_entropy_with_logits(labels=correct,
                                                       logits=predicted /
                                                       self.train_temp)

    def train(self,
              data,
              mode_save_file,
              train_params):
        """ train with given parameters
        Arguments:
        data (np.ndarray) : data to train on
        mode_save_file (str) : file to save finished model in
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

        self.model.fit(data.train_data,
                       data.train_labels,
                       batch_size=train_params['batch_size'],
                       validation_data=(data.validation_data,
                                        data.validation_labels),
                       epochs=train_params['num_epochs'],
                       verbose=2,
                       shuffle=True)

        if mode_save_file is not None:
            self.model.save(mode_save_file)

        return self.model

    def predict(self, data):
        """ predict using the underlying model """
        return self.model(data)
