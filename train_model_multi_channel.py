"""Defending against adversarial attacks by randomized diversification"""

import argparse
import os
import keras
import tensorflow as tf

import libs.setup_mnist as setup_mnist
import libs.setup_cifar as setup_cifar
import libs.model_multi_channel as mcm

#######################################################


def make_dir(dir_name):
    """Create directory dir_name if it doesn't exist"""
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    return dir_name


#######################################################


#######################################################


def train(model_type, epochs, optimizer, learning_rate, batch_size):
    """ Train a multi-channel model

    Args:
        model_type (str) : the type of model to train,
            one of: ['mnist', 'fashion_mnist', 'cifar']
        epochs : the epochs to train for
        optimizer : the keras optimizer to use,
            see https://keras.io/optimizers/
        learning_rate : the keras learning rate
        batch_size : the keras batch size
    """

    channels_per_subband = [1, 2, 3]  # number of channels per subband
    dct_subbands = ["d", "h", "v"]  # DCT subbands
    model_save_dir = make_dir(os.path.join("models", model_type))

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        keras.backend.set_session(sess)

        # training data and parameters
        if model_type == "mnist":
            data = setup_mnist.load_mnist_data()
            nn_layer_sizes = [32, 32, 64, 64, 200, 200]
            model = setup_mnist.MNISTModelAllLayers(nn_layer_sizes)
            image_size = 28
            num_channels = 1
        elif model_type == "fashion_mnist":
            data = setup_mnist.load_fashion_mnist_data()
            nn_layer_sizes = [32, 32, 64, 64, 200, 200]
            model = setup_mnist.MNISTModelAllLayers(nn_layer_sizes)
            image_size = 28
            num_channels = 1
        elif model_type == "cifar":
            data = setup_cifar.load_cifar_data()
            nn_layer_sizes = [64, 64, 128, 128, 256, 256]
            model = setup_cifar.CIFARModelAllLayers(nn_layer_sizes)
            image_size = 32
            num_channels = 3

        # multi-channel model initialization with
        # classifier defined in model variable
        multi_channel_model = mcm.MultiChannel(model,
                                               type=model_type,
                                               epochs=epochs,
                                               optimazer=optimizer,
                                               learning_rate=learning_rate,
                                               batch_size=batch_size,
                                               permt=channels_per_subband,
                                               subbands=dct_subbands,
                                               model_dir=model_save_dir,
                                               img_size=image_size,
                                               img_channels=num_channels)
        # multi-channel model training
        multi_channel_model.train(data)


def main():
    """ main function for training """
    parser = argparse.ArgumentParser(
        description="Train multi-channel system with randomized diversification.")
    parser.add_argument("--type", default="mnist", help="The dataset.")
    parser.add_argument("--save_to", default="models",
                        help="Path where to save models.")
    parser.add_argument("--is_zero", default=False, type=int,
                        help="Is to use hard thresholding.")
    parser.add_argument("--epochs", default=50, type=int,
                        help="The number of epochs.")
    parser.add_argument("--lr", default=1e-3, type=float,
                        help="Learning rate.")
    parser.add_argument("--optimizer", default="adam",
                        help="The oprimazation technique.")
    parser.add_argument("--batch_size", default=64,
                        type=int, help="Batch size.")

    args = parser.parse_args()

    train(args.type, args.epochs, args.optimizer, args.lr, args.batch_size)


if __name__ == "__main__":
    main()
