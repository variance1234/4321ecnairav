from keras.layers import Input
from keras.datasets import mnist

from keras.preprocessing.image import ImageDataGenerator

from keras.utils import to_categorical

from imageio import imsave, imread

import numpy as np

from keras.optimizers import SGD
from keras.callbacks import LearningRateScheduler

from .Model1 import Model1
from .Model2 import Model2
from .Model3 import Model3

from ..models_list import TRAIN, VALIDATE, TEST
from ..models_utils import LRLog, set_weights_to_randoms


def load_lenet_model(name, load_weight=True):
    # input image dimensions
    input_shape = (28, 28, 1)
    input_tensor = Input(shape=input_shape)
    if name == 'LeNet1':
        model = Model1(input_tensor=input_tensor)
    elif name == 'LeNet4':
        model = Model2(input_tensor=input_tensor)
    elif name == 'LeNet5':
        model = Model3(input_tensor=input_tensor)

    if not load_weight:
        set_weights_to_randoms(model)

    return model


def load_data_list(dataset, type=TEST, datagen=False, batch_size=128):
    # load data
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    if type == TRAIN:
        x_data = x_train
        y_data = y_train
    elif type == VALIDATE:
        # x_data = x_test
        # y_data = y_test
        x_data = x_test[:int(len(y_test) * 0.75), ...]
        y_data = y_test[:int(len(y_test) * 0.75)]
    else:
        # x_data = x_test
        # y_data = y_test
        x_data = x_test[int(len(y_test) * 0.75):, ...]
        y_data = y_test[int(len(y_test) * 0.75):]

    x_data = x_data.reshape(x_data.shape[0], 28, 28, 1)
    x_data = x_data.astype('float32')

    if datagen:
        x_data = x_data / 255.0

        dg = ImageDataGenerator()
        dg.fit(x_data)

        # return datagen.flow().
        return dg.flow(x_data, to_categorical(y_data, 10), batch_size=batch_size)
    else:
        return np.array(range(len(y_data))), x_data, y_data


def load_data(dataset, data_index, pre_preprocess=True, noise=None):
    if isinstance(dataset, str):
        # load from the path dataset
        try:
            img_data = imread(dataset + '/' + str(data_index))
            img_data = np.expand_dims(img_data, axis=3)
        except Exception:
            return None
    else:
        # slide the data
        img_data = dataset[int(data_index),]

    img_data = img_data.astype('float32')

    img_data = np.expand_dims(img_data, axis=0)

    if noise is not None:
        img_data = img_data + noise

    if pre_preprocess:
        img_data /= 255

    return img_data


### Training optimizer and setting

def lr_schedule(epoch):
    """Learning Rate Schedule
    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.
    # Arguments
        epoch (int): The number of epochs
    # Returns
        lr (float32): learning rate
    """
    lr = 5e-4
    if epoch > 12:
        lr = 1e-5
    elif epoch > 8:
        lr = 5e-5
    elif epoch > 5:
        lr = 1e-4
    elif epoch > 2:
        lr = 2e-4
    print('Learning rate: ', lr)
    return lr


def get_optimizer():
    # lr_scheduler = LearningRateScheduler(lr_schedule)

    lr_log = LRLog()

    # callbacks = [lr_scheduler, lr_log]
    callbacks = [lr_log]

    # optimizer = SGD(lr=lr_schedule(0))
    optimizer = SGD(lr=0.01)

    epochs = 50

    return optimizer, epochs, callbacks


def visualize_data(imagedata):
    return imagedata


def visualize_output(model_output):
    # TODO: Visualize classification label
    return 0
