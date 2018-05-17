import numpy as np
import tensorflow as tf

import os
import sys
sys.path.append('/home/yang/Research/CNN/')
from Prepare_Data import pickle_load, prepare_CIFAR10, prepare_SVHN
from Prepare_Model import prepare_GBPdenoising_end2end, prepare_resnet



import foolbox
from foolbox.models import TensorFlowModel


trainable = False

Attacks = [
    'DeepFool',
    # 'FGSM',
    # 'IterGS',
    # 'IterG'
]

def softmax_np(x, axis=None):
    return np.exp(x) / np.sum(np.exp(x), axis=axis)

def main():

    with tf.Session() as sess:

        # End2End
        tf_model = prepare_GBPdenoising_end2end(sess=sess,
                                                saved='./Models/CIFAR10-32_End2End.ckpt')

        # # pure Resnet
        # tf_model = prepare_resnet(sess=sess,
        #                           load_weights='./Models/CIFAR10-32_Resnet.ckpt',
        #                           num_classes=10)

        input_pl = tf_model.inputs
        logits = tf_model.logits

        # foolbox - construct a tensorflow model
        model = TensorFlowModel(input_pl, logits, bounds=(0, 255))

        # load in the data
        num_advs = 0.
        num_mis = 0.

        for attack in Attacks:

            print(attack)

            # (x_test, y_test) = pickle_load("./", "ADVs_CIFAR10_Resnet_off_Linf_{}.pkl".format(attack))

            (x_train, y_train), (x_test, y_test) = prepare_SVHN("./")

            print(len(x_test))

            for index, adv in enumerate(x_test):

                num_advs = num_advs + 1

                preds = model.predictions(adv)

                if y_test[index] != np.argmax(preds):

                    # print("misclassified")
                    num_mis = num_mis + 1

        print("Total number of ADVs = {}".format(num_advs))
        print("Accuracy = {}".format((num_advs - num_mis) / num_advs))

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    main()