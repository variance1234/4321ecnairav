from __future__ import print_function
import keras
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.datasets import cifar10
from keras.utils.data_utils import Sequence
from keras import backend as K
import numpy as np
import os

from vgg_model import cifar10vgg


def lr_schedule(epoch):
    """Learning Rate Schedule
    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.
    # Arguments
        epoch (int): The number of epochs
    # Returns
        lr (float32): learning rate
    """
    ori_lr = 1.0
    epoch_step = 25
    a = np.floor(epoch/epoch_step)
    lr = float(ori_lr)/float(np.power(2, a))
    print('Learning rate: ', lr)
    return lr


def fgsm(model, x, y, batch_size=128):
    print(x.shape)
    inputs = K.placeholder(shape=(None, 32, 32, 3))
    print("model")
    preds = model(inputs)
    labels = K.placeholder(shape=(None, 10))
    print("labels: ", labels)
    cross_ent = keras.losses.categorical_crossentropy(labels, preds)
    print("cross_ent: ", cross_ent)
    grad, = K.gradients(cross_ent, inputs)
    print("grad: ", grad)
    x_adv = inputs + 0.3 * K.sign(grad)
    print("x_adv: ", x_adv)
    sess = K.get_session()
    nb_batches = int(np.ceil(float(x.shape[0])/batch_size))
    x_adv_list = np.zeros_like(x)
    y_adv_list = np.zeros_like(y)
    for i in range(nb_batches):
        begin = i * batch_size
        end = min((i + 1) * batch_size, x.shape[0])
        x_batch = x[begin:end]
        y_batch = y[begin:end]
        x_adv_list[begin:end], y_adv_list[begin:end] = sess.run([x_adv, preds], feed_dict={inputs: x_batch, labels: y_batch})
    return x_adv_list, y_adv_list


def adv_train(model, x_train, y_train, x_test, y_test, batch_size, epochs, callbacks):
    ind = np.array(range(len(x_train)))
    for i_epoch in range(epochs):
        print('Adversarial training epoch %i/%i', i_epoch, epochs)

        # Shuffle the examples
        np.random.shuffle(ind)

        # Generate adv examples
        x_adv, _ = fgsm(model, x_train, y_train, batch_size)

        model.fit(x_adv, y_train,
                  batch_size=batch_size,
                  epochs=1,
                  validation_data=(x_test, y_test),
                  shuffle=True,
                  callbacks=callbacks,
                  initial_epoch=i_epoch)


def train(model_path, train_flag=True):
    # Training parameters
    batch_size = 128  # orig paper trained all networks with batch_size=128
    weight_decay = 0.0005
    momentum = 0.9
    epochs = 300
    data_augmentation = True
    num_classes = 10

    # Subtracting pixel mean improves accuracy
    subtract_pixel_mean = True

    # Model name, depth and version
    model_type = 'VGG16'

    # Load the CIFAR10 data.
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Input image dimensions.
    input_shape = x_train.shape[1:]

    # Normalize data.
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    # If subtract pixel mean is enabled
    if subtract_pixel_mean:
        x_train_mean = np.mean(x_train, axis=0)
        x_train -= x_train_mean
        x_test -= x_train_mean

    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')
    print('y_train shape:', y_train.shape)

    # Convert class vectors to binary class matrices.
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    model = cifar10vgg().build_model()
    sgd = SGD(lr=lr_schedule(0), decay=weight_decay, momentum=momentum)
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])
    model.summary()
    print(model_type)

    print("We are here!")
    # Prepare model model saving directory.
    save_dir = os.path.join(os.getcwd(), 'saved_models')
    model_name = 'cifar10_%s_model.{epoch:03d}.h5' % model_type
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    filepath = os.path.join(save_dir, model_name)

    # Prepare callbacks for model saving and for learning rate adjustment.
    checkpoint = ModelCheckpoint(filepath=filepath,
                                 monitor='val_acc',
                                 verbose=1,
                                 save_best_only=True)

    lr_scheduler = LearningRateScheduler(lr_schedule)

    # lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
    #                                cooldown=0,
    #                                patience=5,
    #                                min_lr=0.5e-6)

    # callbacks = [checkpoint, lr_reducer, lr_scheduler]
    callbacks = [checkpoint, lr_scheduler]

    ind = np.array(range(len(x_train)))
    for i_epoch in range(epochs):
        print('Adversarial training epoch %i/%i', i_epoch, epochs)

        # Shuffle the examples
        np.random.shuffle(ind)

        # Generate adv examples
        x_adv, _ = fgsm(model, x_train, y_train, batch_size)

        model.fit(x_adv, y_train,
                  batch_size=batch_size,
                  epochs=1,
                  verbose=1,
                  validation_data=(x_test, y_test),
                  shuffle=True,
                  callbacks=callbacks,
                  initial_epoch=i_epoch + 1)

    # Score trained model.
    scores = model.evaluate(x_test, y_test, verbose=1)
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])

    model.save(model_path)
    print('Finished training model and saved model at:', model_path)


if __name__ == "__main__":
    if (K.backend() == 'tensorflow'):
        import tensorflow as tf

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        K.set_session(sess)

    train("cifar10vgg_adv_training.h5")
