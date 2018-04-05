import os
import sys
sys.path.append('/home/yang/Research/CNN/')
from Prepare_Model import prepare_GBPdenoising_end2end

import tensorflow as tf
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator


def main():

    num_classes = 10
    num_epochs = 300
    batch_size = 128

    ########################################## Prepare the Data ########################################################

    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    datagen = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True)

    # compute quantities required for featurewise normalization
    # (std, mean, and principal components if ZCA whitening is applied)
    datagen.fit(x_train)

    ########################################## Prepare the Data ########################################################


    ################################## Tensor Operations for the Training ##############################################

    tf_model = prepare_GBPdenoising_end2end() # build up the computational graph

    input_pl = tf_model.inputs # get the input placeholder
    label_pl = tf_model.labels # get the label placeholder

    cross_entropy = tf_model.cost # model cost

    train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy) # training operation

    accuracy = tf_model.accuracy # model prediction accuracy

    init = tf.global_variables_initializer() # initializer

    saver = tf.train.Saver() # model saver

    ################################## Tensor Operations for the Training ##############################################

    with tf.Session() as sess:

        sess.run(init)

        for e in range(num_epochs):

            b = 0
            for x_batch, y_batch in datagen.flow(x_train, y_train, batch_size=batch_size):

                _, train_accu = \
                    sess.run([train_step, accuracy], feed_dict={input_pl: x_batch, label_pl: y_batch})

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
                            sess.run(accuracy, feed_dict={input_pl: test_X_batch, label_pl: test_y_batch}) * batch_size

                    msg = "Epoch = {}, Test Accuracy = {:.4f}".format(e, test_accu / len(x_test))

                    print(msg)

                    break

        saver.save(sess, 'Models/Pure_TF.ckpt')

if __name__ == "__main__":
    # setup the GPUs to use
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    main()

