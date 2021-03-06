'''Defending against adversarial attacks by randomized diversification'''

import numpy as np
from datetime import datetime
from scipy.fftpack import dct, idct
import copy
from enum import Enum

PermutationMode = Enum('Permutation', 'full zero identity')


class MultiChannel():
    def __init__(self,
                 model,
                 type="",
                 epochs=100,
                 optimizer="adam",
                 learning_rate=1e-3,
                 batch_size=64,
                 permt=[1, 2, 3],
                 subbands=["d", "h", "v"],
                 model_dir="",
                 img_size=28,
                 img_channels=1,
                 permutation_mode='full',
                 use_cuda=True):

        super(MultiChannel, self).__init__()

        self.image_size = img_size
        self.n_channel = img_channels
        self.use_cuda = use_cuda

        self.type = type
        if permutation_mode == 'full':
            self.permutation_mode = PermutationMode.full
        elif permutation_mode == 'zero':
            self.permutation_mode = PermutationMode.zero
        elif permutation_mode == 'identity':
            self.permutation_mode = PermutationMode.identity
        else:
            raise ValueError(
                """Unknown permutation mode %s, use one of:
-full
-zero
-identity""" % permutation_mode)

        self.model = model
        self.epochs = epochs
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        self.P = permt
        self.SUBBANDS = subbands
        self.NET = []
        self.EPOCHS = []
        self.permutation = []

        self.model_dir = model_dir
        self.name = self.type + "_" + str(
            self.permutation_mode.name) + "_permutation_subband_%s_p%d"

    def train(self, data):
        """ Train the multi-channel model on data """
        for subband in self.SUBBANDS:  # dct subbands
            for p in self.P:  # number of channels per subband

                print("\n\n" + datetime.now().strftime("%Y-%m-%d %H:%M:%S") +
                      ": TYPE = %s, p = %d, subband=%s\n\n" %
                      (self.type, p, subband))

                name = self.name % (subband, p)

                # dct subband permutation per channel
                permutation = self.generate_sign_permutation(subband)
                permutation_file_name = self.model_dir + '/permutation_' + name
                print('saving permutation to %s' % permutation_file_name)
                np.save(permutation_file_name, permutation)
                data['train_data'] = self.apply_dct_permutation(
                    data['train_data'], permutation)
                data['validation_data'] = self.apply_dct_permutation(
                    data['validation_data'], permutation)

                # classifier training per channel
                train_params = {
                    'learning_rate': self.learning_rate,
                    'optimizer': self.optimizer,
                    'batch_size': self.batch_size,
                    'num_epochs': self.epochs
                }
                # for epoch in range(self.epochs):
                self.model.train(
                    data, self.model_dir + "/" + name + "_epochs_%d" % self.epochs,
                    train_params)

    def test_init(self, epochs):
        """ initialize model from saved checkpoint """
        i = -1
        for p in self.P:
            for subband in self.SUBBANDS:
                i += 1

                # --- load model ----
                pref = self.model_dir + "/" + self.name % (subband, p)
                model = copy.deepcopy(self.model)
                model.model.load_weights(pref + "_epochs_%d" % epochs[i])
                self.NET.append(model)
                # --- end load model ----

                # --- load permutation ----
                self.permutation.append(
                    np.load(self.model_dir + "/permutation_" + self.name %
                            (subband, p) + ".npy"))
                # --- end load permutation ----

    def predict(self, x):
        """ predict labels for x """
        pred_labels = np.zeros((x.shape[0], 10))

        N = len(self.NET)
        for i in range(N):

            inputs = self.apply_dct_permutation(x.copy(), self.permutation[i])
            pred_labels += self.NET[i].model.predict(inputs)

        return pred_labels

    # -----------------------------------------------------------------------
    def apply_dct_permutation(self, data, permutation):
        """ apply DCT transformation and permutation to data """
        n = data.shape[0]

        for i in range(n):
            for c in range(self.n_channel):
                xdct = dct(dct(data[i, :, :, c]).T)
                xdct = self.apply_sign_permutation(xdct, permutation)
                data[i, :, :, c] = idct(idct(xdct).T)
                nrm = np.sqrt(np.sum(data[i, :, :, c]**2))
                data[i, :, :, c] /= nrm

        return data

    def apply_sign_permutation(self, data, permutation):
        dim = data.shape

        data = np.reshape(data, (-1, self.image_size**2))
        data = np.multiply(data, np.tile(permutation, (data.shape[0], 1)))

        return np.reshape(data, dim)

    def generate_sign_permutation(self, subband=""):
        if self.permutation_mode == PermutationMode.zero:
            permutation = np.zeros((1, self.image_size**2))
        elif self.permutation_mode == PermutationMode.identity:
            permutation = np.ones((1, self.image_size**2))
        else:
            permutation = np.random.normal(size=self.image_size**2)
            permutation[permutation >= 0] = 1
            permutation[permutation != 1] = -1

        if subband == "d":  # D - diagonal
            permutation = np.reshape(permutation,
                                     (self.image_size, self.image_size))
            permutation[0:self.image_size // 2, :] = 1
            permutation[:, 0:self.image_size // 2] = 1

        elif subband == "v":  # V - vertical
            permutation = np.reshape(permutation,
                                     (self.image_size, self.image_size))
            permutation[:, 0:self.image_size // 2] = 1
            permutation[self.image_size // 2:self.image_size, :] = 1

        elif subband == "h":  # H - horizontal
            permutation = np.reshape(permutation,
                                     (self.image_size, self.image_size))
            permutation[0:self.image_size // 2, :] = 1
            permutation[:, self.image_size // 2:self.image_size] = 1

        elif subband == "dhv":
            permutation = np.reshape(permutation,
                                     (self.image_size, self.image_size))
            permutation[0:self.image_size // 2, 0:self.image_size // 2] = 1

        return np.reshape(permutation, (self.image_size**2))
