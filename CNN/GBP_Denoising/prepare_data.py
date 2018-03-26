import sys
sys.path.append('/home/yang/Research/CNN/')
sys.path.append('/home/yang/Research/CNN/Prepare_Model')
sys.path.append('/home/yang/Research/CNN/Tools')

from keras.datasets import cifar10
import numpy as np
import tensorflow as tf
from Prepare_Model import prepare_GBP_small_CNN
import pickle as pkl


# length of the logits vector
output_dim = 100

# reset graph & start session
tf.reset_default_graph()
sess = tf.Session()

# prepare a randomly initialized shallow CNN with the gradient has been overwritten to GBP
model = prepare_GBP_small_CNN(sess, input_dim=32, output_dim=output_dim)

def GBP_Reconstruction(image):

    # generate a random index
    random_index = np.random.choice(output_dim, 1)

    # the tf tensor to calculate the GBP visualization
    gbp_tensor = tf.gradients(model.logits[:, random_index], model.layers_dic['images'])[0]

    temp_result = sess.run(gbp_tensor, feed_dict={model.layers_dic['images'] : tf.expand_dims(image, 0)})
    temp_result -= np.min(temp_result)
    temp_result /= np.max(temp_result)

    return temp_result

# [num_examples, 32, 32, 3]
(X_train_ori, y_train), (X_test_ori, y_test) = cifar10.load_data()
print(X_train_ori.shape)
print(X_test_ori.shape)

# each example to its corresponding GBP reconstruction
map_op_train = tf.map_fn(lambda img : GBP_Reconstruction(img), tf.constant(X_train_ori, dtype=tf.float32))
map_op_test = tf.map_fn(lambda img : GBP_Reconstruction(img), tf.constant(X_test_ori, dtype=tf.float32))
X_train_gbp, X_test_gbp = sess.run(map_op_train, map_op_test)

# save to pickle
f = open('./../../../Data/{}.pkl'.format('CIFAR10_GBP'), 'wb')
pkl.dump((X_train_gbp, y_train), f, -1)
pkl.dump((X_test_gbp, y_test), f, -1)
f.close()











