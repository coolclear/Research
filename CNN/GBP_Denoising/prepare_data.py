import sys
sys.path.append('/home/yang/Research/CNN/')
sys.path.append('/home/yang/Research/CNN/Prepare_Model')

from keras.datasets import cifar10
import numpy as np
import tensorflow as tf
from Prepare_Model import prepare_GBP_small_CNN


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

    temp_result = sess.run(gbp_tensor, feed_dict={model.layers_dic['images'] : np.expand_dims(image, 0)})
    temp_result -= np.min(temp_result)
    temp_result /= np.max(temp_result)

    return temp_result

# [num_examples, 3, 32, 32]
(X_train_ori, y_train), (X_test_ori, y_test) = cifar10.load_data()
print(X_train_ori.shape)

# reshape to [num_examples, 32, 32, 3]
X_train_ori_trans = np.transpose(X_train_ori, (0, 2, 3, 1))
print(X_train_ori_trans.shape)
X_test_ori_trans = np.transpose(X_test_ori, (0, 2, 3, 1))

# each example to its corresponding GBP reconstruction
X_train_gbp = [GBP_Reconstruction(img) for img in X_train_ori_trans]
X_test_gbp = [GBP_Reconstruction(img) for img in X_test_ori_trans]











