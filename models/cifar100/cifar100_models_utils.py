from keras.datasets import cifar100

from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical

from keras.models import load_model

from ..models_utils import *

from imageio import imsave, imread
import os.path

from keras.optimizers import Adam, SGD
from keras.callbacks import LearningRateScheduler

from ..models_list import TRAIN, VALIDATE, TEST
from .wideresnet_original import WideResNet28_10


def load_wideresnet_model(load_weight=True):
    if load_weight:
        model = load_model(get_script_folder(__file__) + '/WideResNet28-10.h5')
    else:
        model = WideResNet28_10()

    return model


def load_data_list(datadir, type=TEST, datagen=False, batch_size=128):
    # load default CIFAR100 data
    (x_train, y_train), (x_test, y_test) = cifar100.load_data()

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

    if os.path.exists(datadir):
        return indexes, datadir, y_data
    else:
        if datagen:
            x_data /= 255.0

            x_mean = np.array([125.3, 123.0, 113.9]) / 255.0
            x_std = np.array([63.0, 62.1, 66.7]) / 255.0

            if K.image_data_format() == 'channels_last':
                for i in range(3):
                    x_data[:, :, :, i] -= x_mean[i]
                    x_data[:, :, :, i] /= x_std[i]
            else:
                for i in range(3):
                    x_data[:, i, :, :] -= x_mean[i]
                    x_data[:, i, :, :] /= x_std[i]

            if type == TRAIN:
                dg = ImageDataGenerator(
                    # randomly shift images horizontally
                    width_shift_range=0.125,
                    # randomly shift images vertically
                    height_shift_range=0.125,
                    # fill with reflection
                    fill_mode="reflect",
                    # randomly flip images
                    horizontal_flip=True
                )
            else:
                dg = ImageDataGenerator()

            dg.fit(x_data)

            # Return datagen.flow()
            return dg.flow(x_data, to_categorical(y_data, 100), batch_size=batch_size)
        else:
            return indexes, x_data, y_data


def load_data(dataset, data_index, pre_preprocess=True, noise=None):
    if isinstance(dataset, str):
        # load from the path dataset
        try:
            img_data = imread(dataset + '/' + str(data_index))
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
        img_data /= 255.0

        x_mean = np.array([125.3, 123.0, 113.9]) / 255.0
        x_std = np.array([63.0, 62.1, 66.7]) / 255.0

        if K.image_data_format() == 'channels_last':
            for i in range(3):
                img_data[:, :, :, i] -= x_mean[i]
                img_data[:, :, :, i] /= x_std[i]
        else:
            for i in range(3):
                img_data[:, i, :, :] -= x_mean[i]
                img_data[:, i, :, :] /= x_std[i]

    return img_data


### Training optimizer and setting

def lr_schedule(epoch):
    """Learning Rate Schedule
    Learning rate is scheduled to be reduced after 60, 120, 160 epochs.
    Called automatically every epoch as part of callbacks during training.
    # Arguments
        epoch (int): The number of epochs
    # Returns
        lr (float32): learning rate
    """
    lr = 0.1
    if epoch > 160:
        lr *= 0.2
    if epoch > 120:
        lr *= 0.2
    if epoch > 60:
        lr *= 0.2
    return lr


def get_optimizer():
    lr_scheduler = LearningRateScheduler(lr_schedule)

    lr_log = LRLog()

    callbacks = [lr_scheduler, lr_log]

    optimizer = SGD(momentum=0.9, lr=lr_schedule(0))

    epochs = 200

    return optimizer, epochs, callbacks


def visualize_data(imagedata):
    return imagedata


def visualize_output(model_output):
    # TODO: Visualize classification label
    return 0
