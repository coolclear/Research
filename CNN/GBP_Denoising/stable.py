import os
import sys
sys.path.append('/home/yang/Research/CNN/')

from Prepare_Data import pickle_load, prepare_CIFAR10, prepare_CIFAR100, prepare_SVHN
from Prepare_Model import prepare_GBPdenoising_end2end, prepare_resnet
sys.path.append('/home/yang/Research/CNN/Tools')
from Plot import simple_plot
from logging import warning
import numpy as np
import tensorflow as tf
import keras
from keras.datasets import cifar10
import pickle as pkl

trainable = False

ADVs_Resnet = [
    "ADVs_Resnet_DeepFool",
    "ADVs_Resnet_FGSM",
    "ADVs_Resnet_IterG",
    "ADVs_Resnet_IterGS",
    "ADVs_Resnet_LBFG",
    "ADVs_Resnet_SalMap",
    "ORI"
]

ADVs_End2End = [
    # "ADVs_End2End_DeepFool",
    # "ADVs_End2End_FGSM",
    # "ADVs_End2End_IterG",
    # "ADVs_End2End_IterGS",
    # "ADVs_End2End_LBFG",
    # "ADVs_End2End_SalMap",
    "ORI"
]

def softmax_np(x, axis=None):
    return np.exp(x) / np.sum(np.exp(x), axis=axis)

def main():

    num_classes = 100

    with tf.Session() as sess:

        # pure Resnet
        tf_model = prepare_resnet(sess=sess,
                                  load_weights='./Models/CIFAR100_Resnet.ckpt',
                                  num_classes=num_classes)

        # tf_model = prepare_GBPdenoising_end2end(sess=sess,
        #                                         trainable=trainable,
        #                                         saved='./Models/LearningCurve_End2End_Trainable_{}.ckpt'.format(trainable))

        input_pl = tf_model.inputs
        logits = tf_model.logits

        # for tag in ADVs_End2End:

            # print(tag)

            # if tag == "ORI":
            #     (x_train, y_train), (x_test, y_test) = cifar10.load_data()
            #     # x_test = x_test[:1000]
            #     # y_test = y_test[:1000]
            # else:
            #     with open('{}.pkl'.format(tag), 'rb') as file:
            #         (x_test, y_test) = pkl.load(file)
            #
            # print("Number of examples = {}".format(len(x_test)))
            #
            # if len(x_test) == 0:
            #     continue

        (x_train, y_train), (x_test, y_test) = prepare_CIFAR100()

        # predict one by one
        # for each, we predict for N times to test the model stability
        # set N = 1 for normal prediction
        # check the testing accuracy
        test_accu_single = 0.
        test_accu_vote = 0.
        var = 0.
        times = 50
        stable = 0.
        for index, image in enumerate(x_test):

            if index % 500 == 0:
                print(index)

            label = y_test[index]

            batch_image = np.expand_dims(image, axis=0)
            # prediction
            logit_vals = []
            for step in range(times):
                logit_vals.append(np.argmax(sess.run(logits, feed_dict={input_pl: batch_image})[0]))

            diffs = len(np.unique(logit_vals))
            var += diffs

            if diffs == 1:
                stable += 1

            if logit_vals[0] == label:
                test_accu_single += 1

            if np.bincount(logit_vals).argmax() == label:
                test_accu_vote += 1

        print("Single Test Accuracy = {:.4f}".format(test_accu_single / len(x_test)))
        print("Vote Test Accuracy = {:.4f}".format(test_accu_vote / len(x_test)))
        print("Stability = {:.4f}".format(var / len(x_test)))
        print("Percentage = {:.4f}".format(stable / len(x_test)))

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    main()