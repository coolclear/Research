## train.py -- train the MNIST and CIFAR models
##
## Copyright (C) 2017, Nicholas Carlini <nicholas@carlini.com>.
##
## This program is licenced under the BSD 2-Clause licence,
## contained in the LICENCE file in this directory.

## Modified by Yang Zhang <yz78@rice.edu>

import os
import sys
import tensorflow as tf
import numpy as np

from setup_cifar import CIFAR
sys.path.append('/home/yang/Research/CNN/Prepare_Model')
from Prepare_Model import prepare_GBPdenoising_end2end

# np.random.seed(13)
# tf.set_random_seed(13)

def fn(correct, predicted):
    return tf.nn.softmax_cross_entropy_with_logits(labels=correct,
                                                   logits=predicted)
def get_lr(epoch):
    return 0.1*(.5**(epoch/300*10))

def identical(x):
    return x

def main():

if __name__ == "__main__":
    # setup the GPUs to use
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    main()
