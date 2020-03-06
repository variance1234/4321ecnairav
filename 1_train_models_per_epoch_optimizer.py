import numpy.random
import random
import argparse
import time
import os
import numpy as np
import traceback
import sys
import psutil

from keras.callbacks import Callback, ModelCheckpoint, CSVLogger

import keras

from keras.utils import to_categorical
from keras.utils.multi_gpu_utils import multi_gpu_model
from keras.metrics import top_k_categorical_accuracy

from models import models_list
import running_utils

from keras import backend as K

import scipy.ndimage


def top_5_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=5)


def top_1_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=1)


class NumpyArrayIteratorWrapper(keras.utils.Sequence):
    def __init__(self, numpyArrayIterator):
        self.numpyArrayIterator = numpyArrayIterator

    def __len__(self):
        return len(self.numpyArrayIterator)

    def __getitem__(self, batch_i):
        return self.numpyArrayIterator[batch_i]

    def on_epoch_end(self):
        self.numpyArrayIterator.on_epoch_end()


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, model_name, data_names, data_source, labels, batch_size=5):
        'Initialization'
        self.batch_size = batch_size
        self.model_name = model_name
        self.data_names = data_names
        self.data_source = data_source
        self.labels = labels

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(len(self.data_names) / self.batch_size)
        # return 2

    def __getitem__(self, batch_i):
        'Generate one batch of data'
        # Load the batch
        batch_data_names = self.data_names[batch_i * self.batch_size:(batch_i + 1) * self.batch_size]
        batch_input_datas = list()
        for data_name in batch_data_names:
            data = models_list.get_property(self.model_name, models_list.LOAD_DATA)(self.data_source, data_name)
            batch_input_datas.append(data)

        data = np.concatenate(batch_input_datas, axis=0)

        labels = self.labels[batch_i * self.batch_size: (batch_i + 1) * self.batch_size]

        return data, labels

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        # shuffle here if needed


class TestAndTimeCallback(Callback):
    def __init__(self, generator):
        self.generator = generator
        self.begin_time = 0

    def on_epoch_begin(self, epoch, logs=None):
        self.begin_time = time.time()

    def on_epoch_end(self, epoch, logs={}):
        logs['time'] = time.time() - self.begin_time
        loss, acc = self.model.evaluate_generator(generator=self.generator, verbose=0)
        logs['test_loss'] = loss
        logs['test_acc'] = acc


def cleanup_weight_files(epochs, filename):
    found_best = False
    for epoch in range(epochs, 0, -1):
        fn = (filename + '_e%03d.h5') % epoch
        if os.path.isfile(fn):
            if not found_best:
                # print('RENAME: ' + fn)
                # print('TO: ' + rename_filename)
                if os.path.isfile(filename + '.h5'):
                    os.remove(filename + '.h5')
                os.rename(fn, filename + '.h5')
                found_best = True
            else:
                # print('REMOVE: ' + fn)
                os.remove(fn)


def main():
    # read the parameter
    # argument parsing
    parser = argparse.ArgumentParser(
        description='Train model')
    parser.add_argument('keras_version', help="the keras version used")
    parser.add_argument('backend', help="the back end name used in this run",
                        choices=['theano', 'tensorflow', 'cntk'])
    parser.add_argument('backend_version', help="the back end name used in this run")
    parser.add_argument('cuda_version', help="cuda version")
    parser.add_argument('cudnn_version', help="cudnn version")
    parser.add_argument('model_name', help="the model name")
    parser.add_argument('no_gpu', help="the number of gpus")
    parser.add_argument('training_type', help="the training type",
                        choices=['from_scratch', 'fine_tuning', 'transfer'])
    parser.add_argument('random_seed', help="set the random seed",
                        default=-1)
    parser.add_argument('data_dir', nargs='?', help="the path of the data folder",
                        default='./data')
    parser.add_argument('result_dir', nargs='?', help="the path of the result folder",
                        default='./result')
    parser.add_argument('iTry', nargs='?', help="run id", default=0)
    parser.add_argument('done_filename', help="the path of the result folder")

    args = parser.parse_args()

    model_name = args.model_name
    training_type = args.training_type

    iTry = int(args.iTry)

    random_seed = int(args.random_seed)

    isRandom = random_seed < 0

    if random_seed >= 0:
        os.environ['PYTHONHASHSEED'] = str(random_seed)
        random.seed(random_seed)
        numpy.random.seed(random_seed)

        # Deal with different backend seed
        if (K.backend() == 'tensorflow'):
            if random_seed == 0:
                from tfdeterminism import patch
                patch()
            
            # Deal with tensorflow
            import tensorflow as tf
            tf.set_random_seed(random_seed)

        elif (K.backend() == 'cntk'):
            # Deal with cntk
            from cntk.cntk_py import set_fixed_random_seed
            set_fixed_random_seed(random_seed)
            pass
        else:
            # deal with theano
            # Does not seem to have its own
            pass

    print('DONE SETTING SEED')

    no_gpu = int(args.no_gpu)
    if no_gpu <= 0:
        computation = 'cpu'
    elif no_gpu == 1:
        computation = '1_gpu'
    else:
        computation = str(no_gpu) + '_gpu'

    # Setup done file
    done_path = args.result_dir + '/' + args.done_filename
    running_utils.create_directory(done_path)
    done_out_f = open(done_path, "a")

    # Import models and print import error
    models_list.import_model()

    if len(models_list.import_erros) > 0:
        except_filename = 'import_errors_%s_%s_%s.txt' % (args.keras_version, args.backend, args.backend_version)
        e_path = args.result_dir + '/' + except_filename
        running_utils.create_directory(e_path)
        e_out_f = open(e_path, "w")
        for erros in models_list.import_erros:
            e_out_f.write(erros + '\n\n')
        e_out_f.close()

    print('DONE IMPORT')

    try:
        print('Running %s,%s,%s,%s,%s,%s,%d,%s,%d\n' % (
            args.keras_version, args.backend, args.backend_version, args.cuda_version, args.cudnn_version, model_name,
            random_seed, training_type, iTry))

        begin_run = time.time()

        partial_weight_filename = 'weight_%s_%s_%s_%s_%s_%s_%d_%s_%d' % (
            args.keras_version, args.backend, args.backend_version, args.cuda_version, args.cudnn_version, model_name,
            random_seed, training_type, iTry)

        weight_filename = partial_weight_filename + '_e{epoch:03d}.h5'
        perfomance_filename = 'train_performance_%s_%s_%s_%s_%s_%s_%d_%s_%d.csv' % (
            args.keras_version, args.backend, args.backend_version, args.cuda_version, args.cudnn_version, model_name,
            random_seed, training_type, iTry)
        exception_filename = 'train_exception_%s_%s_%s_%s_%s_%s_%d_%s.txt' % (
            args.keras_version, args.backend, args.backend_version, args.cuda_version, args.cudnn_version, model_name,
            random_seed, training_type)

        # Delete previous run (need to rerun)
        os.system("find '" + args.result_dir + "' -name " + exception_filename + " -type f -delete")

        dataset = models_list.get_property(model_name, models_list.DATA_DIR)

        # open csv to record training performance
        p_path = args.result_dir + '/' + dataset + '/' + perfomance_filename
        running_utils.create_directory(p_path)
        p_out_f = open(p_path, "w")
        p_out_f.write('epoch,train_acc,train_loss,lr,test_acc,test_loss,time,val_acc,val_loss\n')

        batch_size = models_list.get_property(model_name, models_list.TRAINING_BATCH)
        # batch_size=2

        if isinstance(batch_size, dict):
            batch_size = batch_size[args.backend]

        optimizer, epochs, callbacks = models_list.get_property(model_name, models_list.OPTIMIZER)
        # epochs = 2

        if no_gpu > 0:
            bs = batch_size * no_gpu
        else:
            bs = batch_size

        # Load Data List
        data_dir_path = args.data_dir + '/' + dataset
        if training_type == 'from_scratch':
            train_data_generator = NumpyArrayIteratorWrapper(models_list.get_property(model_name, models_list.LOAD_DATA_LIST)(data_dir_path,
                                                                                                    type=models_list.TRAIN,
                                                                                                    datagen=True,
                                                                                                    batch_size=bs))

            print("TYPE OF train_data_generator:" + str(type(train_data_generator)))

            val_data_names, val_data_source, val_labels = \
                models_list.get_property(model_name, models_list.LOAD_DATA_LIST)(
                    data_dir_path, type=models_list.VALIDATE)
            val_data_generator = DataGenerator(model_name, val_data_names, val_data_source,
                                               to_categorical(val_labels), bs)

            test_data_names, test_data_source, test_labels = \
                models_list.get_property(model_name, models_list.LOAD_DATA_LIST)(
                    data_dir_path, type=models_list.TEST)
            test_data_generator = DataGenerator(model_name, test_data_names, test_data_source,
                                                to_categorical(test_labels), bs)

            # val_data_generator = models_list.get_property(model_name, models_list.LOAD_DATA_LIST)(data_dir_path, type=models_list.VALIDATE, datagen=True, batch_size=bs)
            # test_data_generator = models_list.get_property(model_name, models_list.LOAD_DATA_LIST)(data_dir_path, type=models_list.TEST, datagen=True, batch_size=bs)
        elif training_type == 'fine_tuning':
            train_data_names, train_data_source, train_labels = models_list.get_property(model_name,
                                                                                         models_list.LOAD_DATA_LIST)(
                data_dir_path, type=models_list.VALIDATE)
            t_data_names, t_data_source, t_labels = models_list.get_property(model_name, models_list.LOAD_DATA_LIST)(
                data_dir_path, type=models_list.TEST)

            half_t_size = int(len(t_labels) / 2)

            val_data_names = t_data_names[:half_t_size]
            val_labels = t_labels[:half_t_size]
            test_data_names = t_data_names[half_t_size:]
            test_labels = t_labels[half_t_size:]

            if isinstance(t_data_source, str):
                val_data_source = test_data_source = t_data_source
            else:
                if isinstance(t_data_source, list):
                    val_data_source = (t_data_source[0][:half_t_size, ...], t_data_source[1])
                    test_data_source = (t_data_source[0][half_t_size:, ...], t_data_source[1])
                else:
                    val_data_source = t_data_source[:half_t_size, ...]
                    test_data_source = t_data_source[half_t_size:, ...]

            augmented_data_path = args.result_dir + '/data/' + dataset
            train_data_generator = DataGenerator(model_name, train_data_names, augmented_data_path,
                                                 to_categorical(train_labels), bs)
            val_data_generator = DataGenerator(model_name, val_data_names, augmented_data_path,
                                               to_categorical(val_labels), bs)
            test_data_generator = DataGenerator(model_name, test_data_names, augmented_data_path,
                                                to_categorical(test_labels), bs)
        else:
            raise Exception('Not yet supported!')

        load_weight = training_type != 'from_scratch'

        if no_gpu <= 1:
            # Load the model
            model = models_list.get_property(model_name, models_list.INIT_METHOD)(load_weight=load_weight)
        else:
            with tf.device('/cpu:0'):
                cpu_model = models_list.get_property(model_name, models_list.INIT_METHOD)(load_weight=load_weight)
            model = multi_gpu_model(cpu_model, gpus=no_gpu)

        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        print("Done load model " + model_name)

        checkpoint_acc = ModelCheckpoint(filepath=args.result_dir + '/' + dataset + '/best_acc_' + weight_filename,
                                         monitor='val_acc',
                                         mode='max',
                                         verbose=1,
                                         save_best_only=True,
                                         save_weights_only=True)
        checkpoint_loss = ModelCheckpoint(filepath=args.result_dir + '/' + dataset + '/best_loss_' + weight_filename,
                                          monitor='val_loss',
                                          mode='min',
                                          verbose=1,
                                          save_best_only=True,
                                          save_weights_only=True)

        csv_logger = CSVLogger(p_path, append=True)

        test_time = TestAndTimeCallback(test_data_generator)

        callbacks = [test_time, checkpoint_acc, checkpoint_loss] + callbacks + [csv_logger]

        # Get loss and accuracy value before the first epoch
        test_loss, test_acc = model.evaluate_generator(test_data_generator)
        val_loss, val_acc = model.evaluate_generator(val_data_generator)
        train_loss, train_acc = model.evaluate_generator(train_data_generator)

        p_out_f.write('%d, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f\n' % (
            -1, train_acc, train_loss, 0.0, test_acc, test_loss, 0.0, val_acc, val_loss))
        p_out_f.close()

        # Train the model
        if random_seed == 0:
            workers = 1
            use_multiprocessing = False
        else:
            workers = 4
            use_multiprocessing = True

        history = model.fit_generator(train_data_generator, validation_data=val_data_generator, epochs=epochs,
                                      shuffle=isRandom, use_multiprocessing=use_multiprocessing, workers=workers, callbacks=callbacks)

        end_run = time.time()

        end_epoch = max(history.epoch)
        end_loss = history.history['val_loss'][end_epoch]
        end_acc = history.history['val_acc'][end_epoch]
        best_epoch = np.argmax(history.history['val_acc'])
        best_loss = history.history['val_loss'][best_epoch]
        best_acc = history.history['val_acc'][best_epoch]
        best_test_acc = history.history['test_acc'][best_epoch]

        done_out_f.write('%s,%s,%s,%s,%s,%s,%s,%d,%s,%d' % (
            computation, args.keras_version, args.backend, args.backend_version, args.cuda_version, args.cudnn_version,
            model_name, random_seed, training_type, iTry))
        done_out_f.write(',True,%.5f,%d,%.5f,%.5f,%d,%.5f,%.5f,%.5f\n' % (
            end_run - begin_run, end_epoch, end_loss, end_acc, best_epoch, best_loss, best_acc, best_test_acc))

        print("DONE RUN")

        # clean up weight files
        cleanup_weight_files(epochs, args.result_dir + '/' + dataset + '/best_acc_' + partial_weight_filename)
        cleanup_weight_files(epochs, args.result_dir + '/' + dataset + '/best_loss_' + partial_weight_filename)

        print("DONE CLEANUP WEIGHT")

    except Exception:
        exception_filename = 'train_exception_%s_%s_%s_%s_%s_%s_%d_%s.txt' % (
            args.keras_version, args.backend, args.backend_version, args.cuda_version, args.cudnn_version, model_name,
            random_seed, training_type)
        e_path = args.result_dir + '/' + dataset + '/' + except_filename
        running_utils.create_directory(e_path)
        e_out_f = open(e_path, "w")
        e_out_f.write(traceback.format_exc())
        e_out_f.close()

        done_out_f.write(
            '%s,%s,%s,%s,%s,%s,%s,%s,%s,%d' % (
                computation, args.keras_version, args.backend, args.backend_version, args.cuda_version,
                args.cudnn_version, model_name, random_seed, training_type, iTry))
        done_out_f.write(',False\n')

    if 'p_out_f' in locals():
        if p_out_f is not None:
            p_out_f.close()

    # done_out_f.write('%s,%s,%s,%s,%s,%s,%s,%d,%s,%d\n' % (
    #    computation, args.keras_version, args.backend, args.backend_version, args.cuda_version, args.cudnn_version, model_name, random_seed, training_type, iTry))

    # Close done file
    done_out_f.close()

    print("DONE DONE")

    def clean_up():
        parent_pid = os.getpid()
        parent = psutil.Process(parent_pid)
        for child in parent.children(recursive=True):  # or parent.children() for recursive=False
            child.kill()
        parent.kill()

    clean_up()

    print("DONE KILLING")


if __name__ == "__main__":
    if (K.backend() == 'tensorflow'):
        import tensorflow as tf

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        K.set_session(sess)
    main()
