## Modified by Yang Zhang <yz78@rice.edu>

import os
import sys
import tensorflow as tf
from keras.datasets import cifar10
sys.path.append('/home/yang/Research/CNN/Prepare_Model')
from Prepare_Model import prepare_GBPdenoising_end2end

def main():

    num_epochs = 100
    batch_size = 128
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    tf_model = prepare_GBPdenoising_end2end()
    input_pl = tf_model.inputs
    label_pl = tf_model.labels
    cross_entropy = tf_model.cost
    train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)
    accuracy = tf_model.accuracy

    saver = tf.train.Saver()

    with tf.Session() as sess:

        for epoch in range(num_epochs):

            for b in range(int(len(x_train) / batch_size)):

                # prepare batch
                train_X_batch = x_train[batch_size * b: batch_size * b + batch_size]
                train_y_batch = y_train[batch_size * b: batch_size * b + batch_size]

                # train
                sess.run([train_step],feed_dict={input_pl: train_X_batch, label_pl: train_y_batch})

                # print the test accuracy at the end of each epoch
                if b == int(len(x_train) / batch_size) - 1:

                    # testing
                    _, test_accu = \
                        sess.run(accuracy,
                                 feed_dict={input_pl: x_test, label_pl: y_test})

                    msg = "epoch = {}, test accu = {:.4f}".format(epoch, test_accu)

                    print(msg)

        saver.save(sess, 'Models/Pure_TF.ckpt')

if __name__ == "__main__":
    # setup the GPUs to use
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    main()
