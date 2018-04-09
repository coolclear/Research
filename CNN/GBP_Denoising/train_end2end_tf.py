import os
import sys
sys.path.append('/home/yang/Research/CNN/')
sys.path.append('/home/yang/Research/CNN/Tools')
from Prepare_Model import prepare_GBPdenoising_end2end
from Plot import grid_plot

import tensorflow as tf
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator


def main():

    trainable = False
    num_classes = 10
    num_epochs = 1
    batch_size = 64

    ########################################## Prepare the Data ########################################################

    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    samples = x_test[:100]

    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True)

    # compute quantities required for featurewise normalization
    # (std, mean, and principal components if ZCA whitening is applied)
    datagen.fit(x_train)

    ########################################## Prepare the Data ########################################################


    ################################## Tensor Operations for the Training ##############################################

    tf_model = prepare_GBPdenoising_end2end(trainable=trainable) # build up the computational graph

    input_pl = tf_model.inputs # get the input placeholder
    label_pl = tf_model.labels # get the label placeholder
    phase_pl = tf_model.phase # get the phase placeholder

    gbp_reconstruction = tf_model.gbp_reconstruction # the gbp reconstruction output port

    cross_entropy = tf_model.cost # model cost

    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(1e-3,
                                               global_step=global_step,
                                               decay_steps=20000,
                                               decay_rate=0.9)
    train_step = tf.train.AdamOptimizer(learning_rate)\
        .minimize(cross_entropy, global_step=global_step) # training operation

    accuracy = tf_model.accuracy # model prediction accuracy

    init = tf.global_variables_initializer() # initializer

    saver = tf.train.Saver(tf.all_variables()) # model saver

    ################################## Tensor Operations for the Training ##############################################

    with tf.Session() as sess:

        sess.run(init)

        for e in range(num_epochs):

            b = 0
            for x_batch, y_batch in datagen.flow(x_train, y_train, batch_size=batch_size):

                _, train_accu = \
                    sess.run([train_step, accuracy],
                             feed_dict={input_pl: x_batch,
                                        label_pl: y_batch,
                                        phase_pl: 1})

                if b % 50 == 0: # print less message

                    msg = "Epoch = {}, Batch = {}, Accu = {:.4f}".format(e, b, train_accu)

                    print(msg)

                b += 1

                if b >= len(x_train) / batch_size:

                    # calculate the testing accuracy

                    test_accu = 0.
                    for i in range(int(len(x_test) / batch_size)):

                        # prepare batch
                        test_X_batch = x_test[batch_size * i: batch_size * i + batch_size]
                        test_y_batch = y_test[batch_size * i: batch_size * i + batch_size]

                        # accumulate
                        test_accu += \
                            sess.run(accuracy,
                                     feed_dict={input_pl: test_X_batch,
                                                label_pl: test_y_batch,
                                                phase_pl: 0}) * batch_size

                    msg = "Epoch = {}, Test Accuracy = {:.4f}".format(e, test_accu / len(x_test))

                    print(msg)

                    # calculate the gbp reconstruction on the samples

                    reconstructions = sess.run(gbp_reconstruction,
                                               feed_dict={input_pl: samples})

                    grid_plot([10, 10], reconstructions,
                              'GBP_Reconstruction_Epoch_{}'.format(e),
                              './Visualization/Trainable_{}/'.format(trainable),
                              'Epoch_{}'.format(e))

                    break

        saver.save(sess, 'Models/End2End_Trainable_{}.ckpt'.format(trainable))

if __name__ == "__main__":
    # setup the GPUs to use
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    main()

