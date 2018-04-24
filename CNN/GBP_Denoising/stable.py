import os
import sys
sys.path.append('/home/yang/Research/CNN/')
from Prepare_Model import prepare_GBPdenoising_end2end
sys.path.append('/home/yang/Research/CNN/Tools')
from Plot import simple_plot
from logging import warning
import numpy as np
import tensorflow as tf
import keras
from keras.datasets import cifar10

trainable = False

def softmax_np(x, axis=None):
    return np.exp(x) / np.sum(np.exp(x), axis=axis)

def main():

    # load in the data
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    with tf.Session() as sess:

        # load in the trained model
        tf_model = prepare_GBPdenoising_end2end(sess=sess,
                                                trainable=trainable,
                                                saved='./Models/LearningCurve_End2End_Trainable_{}.ckpt'.format(trainable))

        input_pl = tf_model.inputs
        logits = tf_model.output

        # predict one by one
        # for each, we predict for N times to test the model stability
        # check the testing accuracy
        test_accu_single = 0.
        test_accu_vote = 0.
        var = 0.
        times = 50
        for index, image in enumerate(x_test):

            if index % 500 == 0:
                print(index)

            label = y_test[index]

            batch_image = np.expand_dims(image, axis=0)
            # prediction
            logit_vals = []
            for step in range(times):
                logit_vals.append(np.argmax(sess.run(logits, feed_dict={input_pl: batch_image}), axis=1))

            diffs = len(np.unique(logit_vals))
            var += diffs

            if logit_vals[0] == label:
                test_accu_single += 1

            if np.bincount(logit_vals).argmax() == label:
                test_accu_vote += 1

        print("Single Test Accuracy = {:.4f}".format(test_accu_single / len(x_test)))
        print("Vote Test Accuracy = {:.4f}".format(test_accu_vote / len(x_test)))
        print("Stability = {:.4f}".format(var / len(x_test)))

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    main()