import os
import sys
sys.path.append('/home/yang/Research/CNN/')
sys.path.append('/home/yang/Research/CNN/Tools')

from Prepare_Model import prepare_GBP_End2End, prepare_Resnet
from Prepare_Data import prepare_CIFAR10, prepare_CIFAR100, prepare_SVHN, pickle_load
from Plot import grid_plot

import tensorflow as tf
import keras
from keras.preprocessing.image import ImageDataGenerator

model_type = [
    # 'End2End',
    'Resnet'
]

dataset = [
    'CIFAR10',
    # 'CIFAR100',
    # 'SVHN'
]

def main():

    for type in model_type:
        for set in dataset:
            train(set, type)

def train(dataset, model_type, lr=1e-3, num_epochs=2, batch_size=64):

    ########################################## Prepare the Data ########################################################

    if dataset == 'CIAFR10':
        (x_train, y_train), (x_test, y_test) = prepare_CIFAR10()
        num_classes = 10
        input_dim = 32
    elif dataset == 'CIFAR100':
        (x_train, y_train), (x_test, y_test) = prepare_CIFAR100()
        num_classes = 100
        input_dim = 32
    else:
        (x_train, y_train), (x_test, y_test) = prepare_SVHN("./")
        num_classes = 10
        input_dim = 32

    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    steps_per_epoch = int((len(x_train) - 1) / batch_size) + 1
    print('Steps per epoch = ', steps_per_epoch)

    datagen = ImageDataGenerator(
        rotation_range=5,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True)

    # compute quantities required for featurewise normalization
    # (std, mean, and principal components if ZCA whitening is applied)
    datagen.fit(x_train)

    ########################################## Prepare the Data ########################################################

    tf.reset_default_graph() # erase whatever the previous graph

    ################################## Tensor Operations for the Training ##############################################

    # Input and Output
    input_ph = tf.placeholder(tf.float32, [None, input_dim, input_dim, 3])
    label_ph = tf.placeholder(tf.float32, [None, num_classes])

    # Model/Graph
    if model_type == 'End2End':
        _, tf_model = prepare_GBP_End2End(num_classes, inputT=input_ph)
        gbp_reconstruction = tf_model.gbp_reconstructions  # the gbp reconstruction output port
    else:
        _, tf_model = prepare_Resnet(num_classes, inputT=input_ph)

    phase_ph = tf_model.phase # get the phase placeholder
    kp_ph = tf_model.kp # get the keep probability placeholder

    # Loss
    cross_entropy_loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=label_ph, logits=tf_model.logits))

    # Learning Rate
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(lr, global_step=global_step, decay_steps=20000, decay_rate=0.9)
    tf.summary.scalar('lr', learning_rate)

    # Train Step
    # notice that we have the batch_normalization, the training op will be different
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_step = tf.train.AdamOptimizer(learning_rate)\
            .minimize(cross_entropy_loss, global_step=global_step) # training operation

    # Accuracy
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(tf_model.logits, 1), tf.argmax(label_ph, 1)), tf.float32))
    tf.summary.scalar('Accuracy', accuracy) # TensorBoard

    # TensorBoard
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter('TensorBoard/{}/{}/Train'.format(dataset, model_type))
    test_writer = tf.summary.FileWriter('TensorBoard/{}/{}/Test'.format(dataset, model_type))

    init = tf.global_variables_initializer() # initializer
    saver = tf.train.Saver(tf.all_variables()) # model saver

    ################################## Tensor Operations for the Training ##############################################

    ####################################### Actual Training Happens Here ###############################################

    with tf.Session() as sess:

        sess.run(init)

        for e in range(num_epochs):

            b = 0

            for x_batch, y_batch in datagen.flow(x_train, y_train, batch_size=batch_size):

                _, train_accu, summary = \
                    sess.run([train_step, accuracy, merged],
                             feed_dict={input_ph: x_batch,
                                        label_ph: y_batch,
                                        phase_ph: True})

                train_writer.add_summary(summary, b + e * steps_per_epoch)

                if b % 50 == 0:

                    msg = "Epoch = {}, Batch = {}, Accu = {:.4f}".format(e, b, train_accu)

                    print(msg)

                b += 1

                if b == steps_per_epoch:

                    summary = sess.run(merged,
                                       feed_dict={input_ph: x_test[:512],
                                                  label_ph: y_test[:512],
                                                  kp_ph: 1.0})
                    test_writer.add_summary(summary, b + e * steps_per_epoch)


                    if e % 1 == 0: # save every 5 epoches
                        saver.save(sess, '{}_{}/Model'.format(dataset, model_type), global_step=b + e * steps_per_epoch)

                    break



        # Testing Accuracy
        test_accu = 0.
        for i in range(int(len(x_test) / batch_size)):

            # prepare batch
            test_X_batch = x_test[batch_size * i: batch_size * i + batch_size]
            test_y_batch = y_test[batch_size * i: batch_size * i + batch_size]

            # accumulate
            test_accu += \
                sess.run(accuracy,
                         feed_dict={input_ph: test_X_batch,
                                    label_ph: test_y_batch,
                                    kp_ph: 1.0}) * batch_size

        msg = "Test Accuracy = {:.4f}".format(test_accu / len(x_test))
        print(msg)

    ####################################### Actual Training Happens Here ###############################################

if __name__ == "__main__":
    # setup the GPUs to use
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    main()

