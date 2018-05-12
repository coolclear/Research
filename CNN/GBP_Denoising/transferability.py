import os
import sys
sys.path.append('/home/yang/Research/CNN/')

from Prepare_Model import prepare_resnet
from Prepare_Data import pickle_load, prepare_CIFAR10, prepare_CIFAR100, prepare_SVHN
import numpy as np
import tensorflow as tf

Tags = [
    "ORI",
    "GBP_0",
    "GBP_1",
    "GBP_2",
    "GBP_3",
    "GBP_4"
]

def softmax_np(x, axis=None):
    return np.exp(x) / np.sum(np.exp(x), axis=axis)

def main():

    num_classes = 10

    with tf.Session() as sess:

        # Resnet
        tf_model = prepare_resnet(sess=sess,
                                  load_weights='./Models/CIFAR10-GBP0_Resnet.ckpt',
                                  num_classes=num_classes)

        input_pl = tf_model.inputs
        keepprob_pl = tf_model.kp
        logits = tf_model.logits

        for tag in Tags:

            print(tag)

            if tag == "ORI":
                (x_train, y_train), (x_test, y_test) = prepare_CIFAR10()
            else:
                (x_train, y_train), (x_test, y_test) = pickle_load("./", "CIFAR10_{}.pkl".format(tag))

            test_accu_single = 0.
            for index, image in enumerate(x_test):

                label = y_test[index]

                batch_image = np.expand_dims(image, axis=0)
                # prediction
                if np.argmax(sess.run(logits, feed_dict={input_pl: batch_image, keepprob_pl: 1.0})[0]) == label:
                    test_accu_single += 1

            print("Single Test Accuracy = {:.4f}".format(test_accu_single / len(x_test)))

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    main()