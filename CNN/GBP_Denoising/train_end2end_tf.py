## Modified by Yang Zhang <yz78@rice.edu>

import os
import sys
sys.path.append('/home/yang/Research/CNN/')
from Prepare_Model import prepare_GBPdenoising_end2end

import tensorflow as tf
import keras
from keras.datasets import cifar10


def main():

    num_epochs = 200
    batch_size = 128
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    tf_model = prepare_GBPdenoising_end2end()
    input_pl = tf_model.inputs
    label_pl = tf_model.labels
    cross_entropy = tf_model.cost
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    accuracy = tf_model.accuracy
    init = tf.global_variables_initializer()

    saver = tf.train.Saver()

    with tf.Session() as sess:

        sess.run(init)

        for epoch in range(num_epochs):

            for b in range(int(len(x_train) / batch_size)):

                # prepare batch
                train_X_batch = x_train[batch_size * b: batch_size * b + batch_size]
                train_y_batch = y_train[batch_size * b: batch_size * b + batch_size]

                # train
                _, train_accu = \
                    sess.run([train_step, accuracy],feed_dict={input_pl: train_X_batch, label_pl: train_y_batch})

                msg = "epoch = {}, batch = {}, accu = {:.4f}".format(epoch, b, train_accu)

                print(msg)

                test_accu = 0

                # evaluate the test set at the end of each epoch
                if b == int(len(x_train) / batch_size) - 1:

                    for i in range(int(len(x_test) / batch_size)):

                        # prepare batch
                        test_X_batch = x_test[batch_size * i: batch_size * i + batch_size]
                        test_y_batch = y_test[batch_size * i: batch_size * i + batch_size]

                        # accumulate
                        test_accu += \
                            sess.run(accuracy, feed_dict={input_pl: test_X_batch, label_pl: test_y_batch}) * batch_size

                    msg = "Epoch = {}, Test Accuracy = {:.4f}".format(epoch, test_accu / len(x_test))

                    print(msg)

        saver.save(sess, 'Models/Pure_TF.ckpt')

if __name__ == "__main__":
    # setup the GPUs to use
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    main()
