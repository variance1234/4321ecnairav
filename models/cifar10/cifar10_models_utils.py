from keras.datasets import cifar10

from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical

from ..models_utils import *

from imageio import imsave, imread
import os.path

from keras.optimizers import Adam, SGD
from keras.callbacks import LearningRateScheduler, ReduceLROnPlateau

from ..models_list import TRAIN, VALIDATE, TEST

from .resnet_models import ResNet

from .wideresnet_original import WideResNet34_10

global x_train_mean


def load_resnet_model(name, load_weight=True):
    if name == 'ResNet56v1':
        if load_weight:
            model = load_model(get_script_folder(__file__) + '/ResNet56v1.h5')
        else:
            model = ResNet('ResNet56v1')

    elif name == 'ResNet38v1':
        if load_weight:
            model = load_model(get_script_folder(__file__) + '/ResNet38v1.h5')
        else:
            model = ResNet('ResNet38v1')
    elif name == 'ResNet32v1':
        if load_weight:
            model = load_model(get_script_folder(__file__) + '/ResNet32v1.h5')
        else:
            model = ResNet('ResNet32v1')

    return model


def load_wideresnet_model(load_weight=True):
    if load_weight:
        model = load_model(get_script_folder(__file__) + '/WideResNet34-10.h5')
    else:
        model = WideResNet34_10()

    return model


def load_x_train_mean():
    (x_train, _), (_, _) = cifar10.load_data()
    x_train_mean = np.mean(x_train, axis=0)
    return x_train_mean


def load_data_list(datadir, type=TEST, datagen=False, batch_size=128):
    global x_train_mean

    # load default CIFAR10 data
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    if type == TRAIN:
        x_data = x_train
        y_data = y_train
    elif type == VALIDATE:
        x_data = x_test[:int(len(y_test) * 0.75), ...]
        y_data = y_test[:int(len(y_test) * 0.75)]
    else:
        x_data = x_test[int(len(y_test) * 0.75):, ...]
        y_data = y_test[int(len(y_test) * 0.75):]

    indexes = [str(i) for i in range(len(y_data))]
    x_data = x_data.astype('float32')
    y_data = [y[0] for y in y_data]

    x_train = x_train.astype('float32')
    x_train_mean = np.mean(x_train, axis=0)

    if os.path.exists(datadir):
        return indexes, datadir, y_data
    else:
        if datagen:

            x_data -= x_train_mean
            x_data = x_data / 255.0

            if type == TRAIN:
                dg = ImageDataGenerator(
                    # randomly shift images horizontally
                    width_shift_range=0.1,
                    # randomly shift images vertically
                    height_shift_range=0.1,
                    # randomly flip images
                    horizontal_flip=True,
                )
            else:
                dg = ImageDataGenerator()

            dg.fit(x_data)

            # Return datagen.flow().
            return dg.flow(x_data, to_categorical(y_data, 10), batch_size=batch_size)
        else:
            return indexes, [x_data, x_train_mean], y_data


def load_data(dataset, data_index, pre_preprocess=True, noise=None):
    if isinstance(dataset, str):
        # load from the path dataset
        try:
            img_data = imread(dataset + '/' + str(data_index))
        except Exception:
            return None
    else:
        # slide the data
        img_data = dataset[0][int(data_index),]

    img_data = img_data.astype('float32')

    img_data = np.expand_dims(img_data, axis=0)

    if noise is not None:
        img_data = img_data + noise

    if pre_preprocess:
        if isinstance(dataset, str):
            global x_train_mean
            if x_train_mean is None:
                x_train_mean = load_x_train_mean()

            img_data -= x_train_mean
            img_data /= 255
        else:
            img_data -= dataset[1]
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
    lr = 1e-3
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
    return lr


def get_optimizer():
    lr_scheduler = LearningRateScheduler(lr_schedule)

    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                                   cooldown=0,
                                   patience=5,
                                   min_lr=0.5e-6)
    callbacks = [lr_reducer, lr_scheduler]

    optimizer = Adam(lr=lr_schedule(0))

    epochs = 200

    return optimizer, epochs, callbacks


def adv_lr_schedule(epoch):
    lr = 0.1
    if epoch > 102:
        lr = 0.01
    if epoch > 153:
        lr = 0.001
    return lr


def get_adv_optimizer():
    lr_scheduler = LearningRateScheduler(adv_lr_schedule)

    # lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
    #                               cooldown=0,
    #                               patience=5,
    #                               min_lr=0.5e-6)
    # callbacks = [lr_reducer, lr_scheduler]
    callbacks = [lr_scheduler]

    # optimizer = Adam(lr=adv_lr_schedule(0))
    optimizer = SGD(lr=adv_lr_schedule(0), momentum=0.9)

    epochs = 200

    return optimizer, epochs, callbacks


def visualize_data(imagedata):
    return imagedata


def visualize_output(model_output):
    # TODO: Visualize classification label
    return 0
