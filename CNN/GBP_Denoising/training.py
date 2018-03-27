from __future__ import print_function
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import os
import pickle as pkl
import numpy as np
from keras.datasets import cifar10

def main():

    # we could either train the model on original cifar10
    # or the one preprocessed by the GBP reconstruction
    if_gbp_preprocess = False

    if if_gbp_preprocess:
        with open('CIFAR10_GBP.pkl', 'rb') as file:
            (x_train, y_train) = pkl.load(file)
            (x_test, y_test) = pkl.load(file)
    else:
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    ######################### data pre-processing starts here ####################

    # for CIFAR10_ORI, we normalize it roughly in the same way as we do to CIFAR10_GBP
    if not if_gbp_preprocess:
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train /= 255.
        x_test /= 255.

    # subtract the mean
    mean = np.mean(x_train, axis=0)
    x_train -= mean
    x_test -= mean

    ######################### data pre-processing ends here   ####################

    # hyper-parameters
    batch_size = 32
    num_classes = 10
    epochs = 50
    save_dir = os.path.join(os.getcwd(), 'Saved_Models')
    model_name = 'CIFAR10_Trained_Model_GBP_{}.h5'.format(if_gbp_preprocess)

    # one hot encoding
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    ########################## model starts here ################################

    model = Sequential()

    model.add(Conv2D(32, (3, 3), padding='same', input_shape=x_train.shape[1:]))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    ########################## model ends here ###################################

    opt = keras.optimizers.Adagrad(lr=0.01)

    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test),
              shuffle=True)

    # save the model
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    model_path = os.path.join(save_dir, model_name)
    model.save(model_path)

    # evaluate at the end
    scores = model.evaluate(x_test, y_test, verbose=1)
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])

if __name__ == '__main__':
    # setup the GPUs to use
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    main()