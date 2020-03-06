"""
#Trains a ResNet on the CIFAR10 dataset.
ResNet v1:
[Deep Residual Learning for Image Recognition
](https://arxiv.org/pdf/1512.03385.pdf)
ResNet v2:
[Identity Mappings in Deep Residual Networks
](https://arxiv.org/pdf/1603.05027.pdf)
Model|n|200-epoch accuracy|Original paper accuracy |sec/epoch GTX1080Ti
:------------|--:|-------:|-----------------------:|---:
ResNet20   v1|  3| 92.16 %|                 91.25 %|35
ResNet32   v1|  5| 92.46 %|                 92.49 %|50
ResNet44   v1|  7| 92.50 %|                 92.83 %|70
ResNet56   v1|  9| 92.71 %|                 93.03 %|90
ResNet110  v1| 18| 92.65 %|            93.39+-.16 %|165
ResNet164  v1| 27|     - %|                 94.07 %|  -
ResNet1001 v1|N/A|     - %|                 92.39 %|  -
&nbsp;
Model|n|200-epoch accuracy|Original paper accuracy |sec/epoch GTX1080Ti
:------------|--:|-------:|-----------------------:|---:
ResNet20   v2|  2|     - %|                     - %|---
ResNet32   v2|N/A| NA    %|            NA         %| NA
ResNet44   v2|N/A| NA    %|            NA         %| NA
ResNet56   v2|  6| 93.01 %|            NA         %|100
ResNet110  v2| 12| 93.15 %|            93.63      %|180
ResNet164  v2| 18|     - %|            94.54      %|  -
ResNet1001 v2|111|     - %|            95.08+-.14 %|  -
"""

from __future__ import print_function
import keras
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.datasets import cifar10
import numpy as np
import os

import foolbox
from datetime import datetime
from timeit import default_timer as timer

from keras import backend as K

from wideresnet_original import WideResNet34_10


class NumpyArrayIteratorWrapper(keras.utils.Sequence):
    def __init__(self, numpyArrayIterator):
        self.numpyArrayIterator = numpyArrayIterator

    def __len__(self):
        return len(self.numpyArrayIterator)

    def __getitem__(self, batch_i):
        return self.numpyArrayIterator[batch_i]

    def on_epoch_end(self):
        self.numpyArrayIterator.on_epoch_end()


class DataSubset(object):
    def __init__(self, xs, ys):
        self.xs = xs
        self.n = xs.shape[0]
        self.ys = ys
        self.batch_start = 0
        self.cur_order = np.random.permutation(self.n)

    def get_next_batch(self, batch_size, multiple_passes=False, reshuffle_after_pass=True):
        if self.n < batch_size:
            raise ValueError('Batch size can be at most the dataset size')
        if not multiple_passes:
            actual_batch_size = min(batch_size, self.n - self.batch_start)
            if actual_batch_size <= 0:
                raise ValueError('Pass through the dataset is complete.')
            batch_end = self.batch_start + actual_batch_size
            batch_xs = self.xs[self.cur_order[self.batch_start : batch_end], ...]
            batch_ys = self.ys[self.cur_order[self.batch_start : batch_end], ...]
            self.batch_start += actual_batch_size
            return batch_xs, batch_ys
        actual_batch_size = min(batch_size, self.n - self.batch_start)
        if actual_batch_size < batch_size:
            if reshuffle_after_pass:
                self.cur_order = np.random.permutation(self.n)
            self.batch_start = 0
        batch_end = self.batch_start + batch_size
        batch_xs = self.xs[self.cur_order[self.batch_start : batch_end], ...]
        batch_ys = self.ys[self.cur_order[self.batch_start : batch_end], ...]
        self.batch_start += batch_size
        return batch_xs, batch_ys


def get_adversarial_loss(model):
    def adv_loss(y, preds):
        import keras
        cross_ent = keras.losses.categorical_crossentropy(y, preds)

        from keras import backend as K
        grad, = K.gradients(cross_ent, model.input)
        x_adv = model.input + 0.3 * K.sign(grad)
        x_adv = K.stop_gradient(x_adv)
        preds_adv = model(x_adv)
        cross_ent_adv = keras.losses.categorical_crossentropy(y, preds_adv)
        return 0.5 * cross_ent + 0.5 * cross_ent_adv

    return adv_loss


def fgsm(model, x, y, batch_size=128):
    # print(x.shape)
    inputs = K.placeholder(shape=(None, 32, 32, 3))
    # print("model")
    preds = model(inputs)
    labels = K.placeholder(shape=(None, 10))
    # print("labels: ", labels)
    cross_ent = keras.losses.categorical_crossentropy(labels, preds)
    # print("cross_ent: ", cross_ent)
    grad, = K.gradients(cross_ent, inputs)
    # print("grad: ", grad)
    x_adv = inputs + 0.3 * K.sign(grad)
    # print("x_adv: ", x_adv)
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


def fgsm_batch(model, sess, x, y, epsilon=0.3):
    # print(x.shape)
    inputs = K.placeholder(shape=(None, 32, 32, 3))
    # print("model")
    preds = model(inputs)
    labels = K.placeholder(shape=(None, 10))
    K.stop_gradient(labels)
    # print("labels: ", labels)
    cross_ent = keras.losses.categorical_crossentropy(labels, preds)
    # print("cross_ent: ", cross_ent)
    grad, = K.gradients(cross_ent, inputs)
    # print("grad: ", grad)
    x_adv = inputs + epsilon * K.sign(grad)
    # print("x_adv: ", x_adv)
    x_adv_list, y_adv_list = sess.run([x_adv, preds], feed_dict={inputs: x, labels: y})
    return np.array(x_adv_list), np.array(y_adv_list)


def lr_schedule(step):
    lr = 0.1
    if step > 40000:
        lr = 0.01
    elif step > 60000:
        lr = 0.001
    print('Learning rate: ', lr)
    return lr


def train(model_path, version, n,  attack='pgd'):
    # Training parameters
    max_steps = 80000
    batch_size = 128  # orig paper trained all networks with batch_size=128
    data_augmentation = True
    num_classes = 10
    depth = 32
    weight_decay = 0.0002
    momentum = 0.9
    num_output_steps = 10
    num_summary_steps = 100
    num_checkpoint_steps = 1000

    # Attack parameters
    epsilon = 0.03
    step_size = 0.008
    num_steps = 7

    sess = K.get_session()

    # Subtracting pixel mean improves accuracy
    subtract_pixel_mean = False

    depth = 34
    k = 10

    # Model name, depth and version
    model_type = 'WideResNet%d_%d' % (depth, k)

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
    print(type(y_train))
    print(y_train.dtype)

    model = WideResNet34_10()

    sgd = SGD(lr=lr_schedule(0), momentum=momentum)

    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])
    model.summary()
    print(model_type)

    # Prepare model model saving directory.
    save_dir = os.path.join(os.getcwd(), 'saved_models')
    model_name = 'cifar10_%s_model.{epoch:03d}.h5' % model_type
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    filepath = os.path.join(save_dir, model_name)

    dataset = DataSubset(x_train, y_train)

    # Prepare callbacks for model saving and for learning rate adjustment.
    checkpoint = ModelCheckpoint(filepath=filepath,
                                 monitor='val_acc',
                                 verbose=1,
                                 save_best_only=True)

    lr_scheduler = LearningRateScheduler(lr_schedule)

    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                                   cooldown=0,
                                   patience=5,
                                   min_lr=0.5e-6)

    # callbacks = [checkpoint, lr_reducer, lr_scheduler]

    # symbolic fgsm
    y_input = K.placeholder(shape=(None, 10), dtype='float32')
    cross_ent = keras.losses.categorical_crossentropy(y_input, model.output)
    grad_tensor, = K.gradients(cross_ent, model.input)
    x_adv = model.input + epsilon * K.sign(grad_tensor)
    x_adv = K.stop_gradient(x_adv)
    # y_adv = model(x_adv)

    generating_time_accumulate = 0
    for ii in range(max_steps):
        print('Adversarial training steps %i/%i' % (ii, max_steps))
        x_batch, y_batch = dataset.get_next_batch(batch_size, multiple_passes=True)
        # print(x_batch.shape, y_batch.shape)

        start = timer()
        # fmodel = foolbox.models.KerasModel(model, bounds=(0, 1))
        # pgd_attack = foolbox.attacks.PGD(fmodel, distance=foolbox.distances.Linf)
        #
        # x_batch_adv = []
        # for i in range(len(x_batch)):
        #     try:
        #         adversarial = pgd_attack(x_batch[i], np.argmax(y_batch[i]), epsilon=0.03, stepsize=0.008,
        #                                  iterations=7, random_start=True)
        #     except AssertionError as ae:
        #         print(" message: " + str(ae))
        #         adversarial = x_batch[i]
        #     if np.array(adversarial).shape != (32, 32, 3):
        #         print(np.array(adversarial).shape)
        #     x_batch_adv.append(adversarial)

        # x_batch_adv = pgd_attack(x_batch, np.argmax(y_batch, axis=1), epsilon=0.03, stepsize=0.008, iterations=7,
        #                          random_start=True)
        # x_batch_adv_nan = np.isnan(x_batch_adv)
        # x_batch_adv[x_batch_adv_nan] = x_batch[x_batch_adv_nan]
        # print(x_batch_adv.shape)

        if ii == 0:
            x_batch_adv, y_batch_adv = x_batch, y_batch
        else:
            # x_batch_adv, y_batch_adv = fgsm_batch(model, sess, x_batch, y_batch, epsilon=0.03)
            # x_batch_adv, y_batch_adv = sess.run([x_adv, y_adv], feed_dict={model.input: x_batch, y_input: y_batch})
            if attack == 'fgsm':
                x_batch_adv = sess.run(x_adv, feed_dict={model.input: x_batch, y_input: y_batch})
            elif attack == 'pgd':
                x_batch_adv = x_batch + np.random.uniform(-epsilon, epsilon, x_batch.shape)
                y_batch_adv = y_batch
                x_batch_adv = np.clip(x_batch_adv, 0.0, 1.0)
                for i in range(num_steps):
                    grad = sess.run(grad_tensor, feed_dict={model.input: x_batch_adv, y_input: y_batch_adv})
                    x_batch_adv = np.add(x_batch_adv, step_size * np.sign(grad), out=x_batch_adv, casting='unsafe')

                    x_batch_adv = np.clip(x_batch_adv, x_batch - epsilon, x_batch + epsilon)
                    x_batch_adv = np.clip(x_batch_adv, 0.0, 1.0)  # ensure valid pixel range

            else:
                raise Exception('Illegal attack name. ')

        x_batch_adv = np.array(x_batch_adv)
        end = timer()
        print(len(x_batch_adv))
        generating_time = end - start
        print("generating time: ", generating_time)
        generating_time_accumulate += generating_time

        if ii == 40000:
            K.set_value(model.optimizer.lr, 0.01)
        if ii == 60000:
            K.set_value(model.optimizer.lr, 0.001)

        if ii % num_output_steps == 0:
            nat_pred = model.predict(x_batch)
            nat_acc = np.sum(np.equal(np.argmax(y_batch, axis=1), np.argmax(nat_pred, axis=1)))*1.0/len(y_batch)
            adv_pred = model.predict(x_batch_adv)
            adv_acc = np.sum(np.equal(np.argmax(y_batch, axis=1), np.argmax(adv_pred, axis=1)))*1.0/len(y_batch)

            print('Step {}:    ({})'.format(ii, datetime.now()))
            print('    training nat accuracy {:.4}%'.format(nat_acc * 100))
            print('    training adv accuracy {:.4}%'.format(adv_acc * 100))
            if ii != 0:
                print('    {} examples per second'.format(
                    num_output_steps * batch_size / generating_time_accumulate))
                generating_time_accumulate = 0.0
        callbacks = [checkpoint]
        start = timer()

        model.train_on_batch(x_batch_adv, y_batch_adv)
        end = timer()
        training_time = end - start
        print("training time: ", training_time)

    model.save(model_path)
    print('Finished training model and saved model at:', model_path)

    # Score trained model.
    scores = model.evaluate(x_test, y_test, verbose=1)
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])

    # generate fgsm examples.
    testdataset = DataSubset(x_test, y_test)
    nb_batchs = np.ceil(len(x_test)*1.0/batch_size)

    x_test_fgsm = []
    x_test_pgd = []
    y_test_adv = []
    for i in range(int(nb_batchs)):
        x_batch, y_batch = testdataset.get_next_batch(batch_size)

        #fgsm
        x_batch_fgsm = sess.run(x_adv, feed_dict={model.input: x_batch, y_input: y_batch})
        x_test_fgsm.extend(x_batch_fgsm.tolist())

        #pgd
        x_batch_pgd = x_batch + np.random.uniform(-epsilon, epsilon, x_batch.shape)
        y_batch_pgd = y_batch
        x_batch_pgd = np.clip(x_batch_pgd, 0.0, 1.0)
        for j in range(num_steps):
            grad = sess.run(grad_tensor, feed_dict={model.input: x_batch_pgd, y_input: y_batch_pgd})
            x_batch_pgd = np.add(x_batch_pgd, step_size * np.sign(grad), out=x_batch_pgd, casting='unsafe')

            x_batch_pgd = np.clip(x_batch_pgd, x_batch - epsilon, x_batch + epsilon)
            x_batch_pgd = np.clip(x_batch_pgd, 0.0, 1.0)  # ensure valid pixel range
        x_test_pgd.extend(x_batch_pgd.tolist())

        y_test_adv.extend(y_batch)

    x_test_fgsm = np.array(x_test_fgsm)
    x_test_pgd = np.array(x_test_pgd)
    y_test_adv = np.array(y_test_adv)


    # Score trained model on adversarial input.
    scores = model.evaluate(x_test_fgsm, y_test_adv, verbose=1)
    print('Test fgsm loss:', scores[0])
    print('Test fgsm accuracy:', scores[1])

    scores = model.evaluate(x_test_pgd, y_test_adv, verbose=1)
    print('Test pgd loss:', scores[0])
    print('Test pgd accuracy:', scores[1])


def main():
    # Model params n
    # ----------------------------------------------------------------------------
    #           |      | 200-epoch | Orig Paper| 200-epoch | Orig Paper| sec/epoch
    # Model     |  n   | ResNet v1 | ResNet v1 | ResNet v2 | ResNet v2 | GTX1080Ti
    #           |v1(v2)| %Accuracy | %Accuracy | %Accuracy | %Accuracy | v1 (v2)
    # ----------------------------------------------------------------------------
    # ResNet20  | 3 (2)| 92.16     | 91.25     | -----     | -----     | 35 (---)
    # ResNet32  | 5(NA)| 92.46     | 92.49     | NA        | NA        | 50 ( NA)
    # ResNet44  | 7(NA)| 92.50     | 92.83     | NA        | NA        | 70 ( NA)
    # ResNet56  | 9 (6)| 92.71     | 93.03     | 93.01     | NA        | 90 (100)
    # ResNet110 |18(12)| 92.65     | 93.39+-.16| 93.15     | 93.63     | 165(180)
    # ResNet164 |27(18)| -----     | 94.07     | -----     | 94.54     | ---(---)
    # ResNet1001| (111)| -----     | 92.39     | -----     | 95.08+-.14| ---(---)
    # ---------------------------------------------------------------------------

    # Model version
    # Orig paper: version = 1 (ResNet v1), Improved ResNet: version = 2 (ResNet v2)

    program_start = timer()

    # train(model_path='./ResNet56v1.h5', version=1, n=9)
    # train(model_path='./ResNet38v1.h5', version=1, n=6)
    # train(model_path='./ResNet56v2.h5', version=2, n=6)
    train(model_path='./WideResnet34_10_pgd.h5', version=1, n=5, attack='pgd')

    program_end = timer()
    print("This run takes :", program_end-program_start)


if __name__ == "__main__":
    if (K.backend() == 'tensorflow'):
        import tensorflow as tf

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        K.set_session(sess)
    main()
