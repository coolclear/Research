import os, sys
sys.path.append('/home/yang/open-convnet-black-box/VGGImagenet/')
sys.path.append('/home/yang/open-convnet-black-box/VGGImagenet/Tools/')
import numpy as np
import tensorflow as tf
from Prepare_Model import prepare_cnn3
from Prepare_Data import list_load
from Plot import simple_plot


images = [
    'Dog_1.JPEG',
    'Dog_2.JPEG',
    'Dog_3.JPEG',
    'Dog_4.JPEG',
    'Dog_5.JPEG',
    'tabby.png'
]

sal_type = [
    'PlainSaliency',
    'Deconv',
    'GuidedBackprop'
]

def super_saliency(tensor, inputs, num_to_viz):
    result = []
    shape = int(np.prod(tensor.get_shape()[1:]))
    tensor_flat = tf.reshape(tensor, [-1, shape])
    pick_indices = np.random.choice(shape, num_to_viz)
    for idx in pick_indices:
        result.append(tf.gradients(tensor_flat[:, idx], inputs)[0])
    return tf.stack(result)

def job(cnn3, sal_type, sess, if_pool, batch_img, fns):

    # first: pick one layer
    # second: pick num_to_viz neurons from this layer
    # third: calculate the saliency map w.r.t self.imgs for each picked neuron

    num_to_viz = 10
    layer_name = 'FC'

    # shape = (num_to_viz, num_input_images, 224, 224, 3)
    # TF Graph
    saliencies = super_saliency(cnn3.layers_dic[layer_name], cnn3.layers_dic['Input'], num_to_viz)

    # shape = (num_input_images, num_to_viz, 224, 224, 3)
    saliencies_val = sess.run(saliencies, feed_dict={cnn3.images: batch_img})
    saliencies_val_trans = np.transpose(saliencies_val, (1, 0, 2, 3, 4))

    for idx, name in enumerate(fns):
        save_dir = "01272018/{}/{}/Pooling_{}/".format(name, sal_type, if_pool)
        for index, image in enumerate(saliencies_val_trans[idx]):
            simple_plot(image, name + '_' + sal_type + '_' + str(if_pool) + '_' + str(index), save_dir)

def main():

    for sal in sal_type:

        for pool in [True, False]: # with and without pooling

            tf.reset_default_graph()
            sess = tf.Session()
            cnn3 = prepare_cnn3(sal, sess, pool=pool)

            batch_img, fns = list_load("./../data_imagenet", images)

            for idx, image in enumerate(batch_img):

                job(cnn3, sal, sess, pool, np.expand_dims(image, axis=0), [fns[idx]])

            sess.close()


if __name__ == '__main__':
    # setup the GPUs to use
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    main()
