import os, sys
sys.path.append('/home/yang/open-convnet-black-box/VGGImagenet/')
sys.path.append('/home/yang/open-convnet-black-box/VGGImagenet/Tools/')
import numpy as np
import tensorflow as tf
from Prepare_Model import prepare_vgg, prepare_resnet
from Prepare_Data import list_load
np.set_printoptions(threshold=np.nan)
from  Plot import  simple_plot

sal_type = [
   'PlainSaliency',
   'Deconv',
    'GuidedBackprop'
]

layers = [
    'fc'
]

model_type = [
    # 'trained',
    'random'
]

N = [
    1,
    3,
    5,
    8,
    10
]

images = [
    # 'Dog_1.JPEG',
    # 'Dog_2.JPEG',
    # 'Dog_3.JPEG',
    'Dog_4.JPEG',
    # 'Dog_5.JPEG',
    'tabby.png',
    # 'laska.png'
]

def super_saliency(tensor, inputs, num_to_viz):
    result = []
    shape = int(np.prod(tensor.get_shape()[1:]))
    tensor_flat = tf.reshape(tensor, [-1, shape])
    pick_indices = np.random.choice(shape, num_to_viz)
    for idx in pick_indices:
        result.append(tf.gradients(tensor_flat[:, idx], inputs)[0])
    return tf.stack(result)

def job(vgg, sal_type, sess, init, batch_img, fns, n):

    # first: pick one layer
    # second: pick num_to_viz neurons from this layer
    # third: calculate the saliency map w.r.t self.imgs for each picked neuron

    num_to_viz = 50
    for layer_name in layers:

        print(layer_name)

        # shape = (num_to_viz, num_input_images, 224, 224, 3)
        # TF Graph
        saliencies = super_saliency(vgg.layers_dic[layer_name], vgg.imgs, num_to_viz)

        # shape = (num_input_images, num_to_viz, 224, 224, 3)
        saliencies_val = sess.run(saliencies, feed_dict={vgg.images: batch_img})
        saliencies_val_trans = np.transpose(saliencies_val, (1, 0, 2, 3, 4))

        for idx, name in enumerate(fns):
            save_dir = "resnet_{}/{}/{}/{}/{}".format(n, name, init, sal_type, layer_name)
            for index, sal in enumerate(saliencies_val_trans[idx]):
                if np.count_nonzero(sal):
                    simple_plot(sal, "_" + str(index), save_dir)

def main():

    for sal in sal_type:
        for n in N:
            for init in model_type:
                tf.reset_default_graph()
                sess = tf.Session()
                resnet = prepare_resnet(sal, init, sess, n)

                batch_img, fns = list_load("./../data_imagenet", images)
                job(resnet, sal, sess, init, batch_img, fns, n)

                sess.close()


if __name__ == '__main__':
    # setup the GPUs to use
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    main()
