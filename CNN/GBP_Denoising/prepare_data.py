import sys
import os
sys.path.append('/home/yang/Research/CNN/')
sys.path.append('/home/yang/Research/CNN/Prepare_Model/')
sys.path.append('/home/yang/Research/CNN/Tools')

from keras.datasets import cifar10
import numpy as np
import tensorflow as tf
from Prepare_Model import prepare_GBP_shallow_CNN
from Prepare_Data import prepare_SVHN, prepare_CIFAR10, prepare_CIFAR100, list_load
import pickle as pkl
from Plot import grid_plot, simple_plot

def GBP_Reconstruction(model, output_dim):

    # generate a random index within the range [0, output_dim)
    random_index = np.random.choice(output_dim, 1)

    # the tf operation to calculate the GBP reconstruction
    # notice that this operation is compatible with image_batch : [None, input_dim, input_dim, 3]
    tfOp_gbp_raw = tf.gradients(model.logits[:, random_index], model.layers_dic['images'])[0]
    tfOp_gbp_submin = tf.map_fn(lambda img : img - tf.reduce_min(img), tfOp_gbp_raw)
    tfOp_gbp_divmax = tf.map_fn(lambda img : img / tf.reduce_max(img), tfOp_gbp_submin)
    tfOp_gbp_255 = tf.map_fn(lambda img : tf.cast(img * 255, tf.int32), tfOp_gbp_divmax, dtype=tf.int32)

    return tfOp_gbp_255

def Map(tfOp, ph, images, sess):

    batch_size = 256
    num_examples = images.shape[0]
    num_batches = int((num_examples - 1) / batch_size) + 1
    counter = 0
    result = None

    while counter <  num_batches:

        if counter != num_batches - 1: # not the last batch
            image_batch = images[counter * batch_size : (counter + 1) * batch_size]
        else: # the last batch
            image_batch = images[counter * batch_size : ]

        print('Processing {}/{}'.format(counter, num_batches))

        if counter == 0:
            result = sess.run(tfOp, feed_dict={ph: image_batch})
        else:
            result = np.concatenate((result, sess.run(tfOp, feed_dict={ph: image_batch})), axis=0)

        counter += 1

    print(result.shape)

    return result

def main():

    # the length of the logits vector
    # the only requirement is a positive integer
    # can be randomized
    output_dim = 100

    # reset graph & start session
    tf.reset_default_graph()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    # prepare a randomly initialized shallow CNN with the gradient has been overwritten to "GBP"
    # in terms of the GBP reconstruction, this model can be any as long as it's a ConvNet.
    model = prepare_GBP_shallow_CNN(sess=sess, output_dim=output_dim)

    # tf operation for GBP reconstruction
    tfOp_gbp_reconstruction = GBP_Reconstruction(model, output_dim)

    # [num_examples, 32, 32, 3]
    data_dir = './Plots/SVHN'
    names = ["TEST_1_DeepFool.png", "TEST_1_FGSM.png", "TEST_1_IterGS.png", "TEST_1_IterG.png",
             "TEST_0_DeepFool.png", "TEST_0_FGSM.png", "TEST_0_IterGS.png", "TEST_0_IterG.png",
             "TEST_2_DeepFool.png", "TEST_2_FGSM.png", "TEST_2_IterGS.png", "TEST_2_IterG.png",
    ]
    X_test_ori, _ = list_load(data_dir, names, size=(32, 32))


    # map each training example to its corresponding GBP reconstruction
    # X_train_gbp = Map(tfOp_gbp_reconstruction, model.layers_dic['images'], X_train_ori, sess)
    X_test_gbp = Map(tfOp_gbp_reconstruction, model.layers_dic['images'], X_test_ori, sess)

    # save to pickle
    f = open('./{}.pkl'.format('SVHN-Plot'), 'wb')
    # pkl.dump((X_train_gbp, y_train), f, -1)
    pkl.dump(X_test_gbp, f, -1)
    f.close()

    # # visualization
    # grid_plot([3, 6], X_train_ori[:18], 'Original_CIFAR10', './Visualization', 'Examples_Ori_CIFAR10')
    # grid_plot([3, 6], X_train_gbp[:18], 'GBP_CIFAR10', './Visualization', 'Examples_GBP_CIFAR10')

    for index, image in enumerate(X_test_gbp):
        simple_plot(image, names[index] + 'GBP', './Plots/SVHN/')

if __name__ == '__main__':
    # setup the GPUs to use
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    main()











