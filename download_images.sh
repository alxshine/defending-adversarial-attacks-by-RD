#!/bin/sh

# download adversarial images
wget https://cuicloud.unige.ch/index.php/s/QcSPGSLSRCzc2gm/download -O images.zip
unzip -q images.zip

# download training images
wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz -P data/
wget http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz -P data/
wget http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz -P data/
wget http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz -P data/
