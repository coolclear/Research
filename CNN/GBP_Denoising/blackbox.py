import os
import sys
sys.path.append('/home/yang/Research/CNN/')
from Prepare_Data import pickle_load
from Prepare_Model import prepare_GBPdenoising_end2end, prepare_resnet

import numpy as np
import tensorflow as tf

import foolbox
from foolbox.models import TensorFlowModel


trainable = False

Attacks = [
    'DeepFool',
    'FGSM',
    'IterGS',
    'IterG'
]

def softmax_np(x, axis=None):
    return np.exp(x) / np.sum(np.exp(x), axis=axis)

def main():

    with tf.Session() as sess:

        # End2End
        tf_model = prepare_GBPdenoising_end2end(sess=sess,
                                                saved='./Models/CIFAR10-32_Resnet.ckpt')

        input_pl = tf_model.inputs
        logits = tf_model.logits

        # foolbox - construct a tensorflow model
        model = TensorFlowModel(input_pl, logits, bounds=(0, 255))

        # load in the data
        num_advs = 0.
        num_mis = 0.

        for attack in Attacks:

            (x_test, y_test) = pickle_load("./", "ADVs_CIFAR10_Resnet_off_Linf_{}.pkl".format(attack))

            for index, adv in enumerate(x_test):

                num_advs = num_advs + 1

                preds = model.predictions(adv)

                if y_test[index] != np.argmax(preds):

                    print("Fooled!")
                    num_mis = num_mis + 1

        print("Accuracy = {}".format(num_mis / num_advs))

