import numpy.random
import random
import argparse
import ntpath
import time
import os
import gc
import numpy as np
import h5py
import traceback
import sys

import keras.optimizers

import keras

from keras.utils import to_categorical
from keras.utils.multi_gpu_utils import multi_gpu_model
from keras.metrics import top_k_categorical_accuracy

from models import models_list
import running_utils

from keras import backend as K


def set_weights_to_randoms(model):
    layers = model.layers
    for i in range(len(layers)):
        w = model.layers[i].get_weights()
        ran_w = []
        for wm in w:
            ran_w.append(np.random.sample(wm.shape) - 0.5)
        model.layers[i].set_weights(ran_w)


def top_5_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=5)


def top_1_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=1)


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

    def __getitem__(self, batch_i):
        'Generate one batch of data'
        # Load the batch
        batch_data_names = self.data_names[batch_i * self.batch_size:(batch_i + 1) * self.batch_size]
        batch_input_datas = list()
        for data_name in batch_data_names:
            data = models_list.get_property(self.model_name, models_list.LOAD_DATA)(self.data_source, str(data_name) + '.png')
            batch_input_datas.append(data)

        data = np.concatenate(batch_input_datas, axis=0)

        labels = self.labels[batch_i * self.batch_size: (batch_i + 1) * self.batch_size]

        return data, labels

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        # shuffle here if needed


def main():
    # read the parameter
    # argument parsing
    parser = argparse.ArgumentParser(
        description='Generate prediction result for each backend')
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
        numpy.random.seed(random_seed)
        random.seed(random_seed)

        # Deal with different backend seed
        if (K.backend() == 'tensorflow'):
            # Deal with tensorflow
            import tensorflow as tf
            tf.set_random_seed(random_seed)
        elif (K.backend() == 'cntk'):
            # Deal with cntk
            from cntk._cntk_py import set_fixed_random_seed
            set_fixed_random_seed(random_seed)
            pass
        else:
            # deal with theano
            # Does not seem to have its own
            pass

    no_gpu = int(args.no_gpu)
    if no_gpu <= 0:
        computation = 'cpu'
    elif no_gpu == 1:
        computation = '1_gpu'
    else:
        computation = str(no_gpu) + '_gpu'

    # Setup done file
    p_path = args.result_dir + '/' + args.done_filename
    running_utils.create_directory(p_path)
    done_out_f = open(p_path, "w")

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

    '''

    try:
        print('Running %s,%s,%s,%s,%s,%s,%s,%d,%s,%d\n' % (
            computation, args.keras_version, args.backend, args.backend_version, args.cuda_version, args.cudnn_version, model_name, random_seed, training_type, iTry))
        begin_run = time.time()

        
        weight_filename = 'weight_%s_%s_%s_%s_%s_%s_%d_%s_%d.h5' % (
            args.keras_version, args.backend, args.backend_version, args.cuda_version, args.cudnn_version, model_name, random_seed, training_type, iTry)
        perfomance_filename = 'train_performance_%s_%s_%s_%s_%s_%s_%d_%s_%d.csv' % (
            args.keras_version, args.backend, args.backend_version, args.cuda_version, args.cudnn_version, model_name, random_seed, training_type, iTry)
        exception_filename = 'train_exception_%s_%s_%s_%s_%s_%s_%d_%s.txt' % (
            args.keras_version, args.backend, args.backend_version, args.cuda_version, args.cudnn_version, model_name, random_seed, training_type)

        # Delete previous run (need to rerun)
        os.system("find '" + args.result_dir + "' -name " + weight_filename + " -type f -delete")
        os.system("find '" + args.result_dir + "' -name " + perfomance_filename + " -type f -delete")
        os.system("find '" + args.result_dir + "' -name " + exception_filename + " -type f -delete")

        dataset = models_list.get_property(model_name, models_list.DATA_DIR)

        # open csv to record training performance
        p_path = args.result_dir + '/' + dataset + '/' + perfomance_filename
        running_utils.create_directory(p_path)
        p_out_f = open(p_path, "w")
        p_out_f.write('backend,comp,network,epoch,time,val_loss,val_accu,train_loss,train_accu\n')

        # Load Data List
        data_dir_path = args.data_dir + '/' + dataset
        if training_type == 'from_scratch':
            train_data_names, train_data_source, train_labels = models_list.get_property(model_name, models_list.LOAD_DATA_LIST)(data_dir_path, type=models_list.TRAIN)
            val_data_names, val_data_source, val_labels = models_list.get_property(model_name, models_list.LOAD_DATA_LIST)(data_dir_path, type=models_list.VALIDATE)
            test_data_names, test_data_source, test_labels = models_list.get_property(model_name, models_list.LOAD_DATA_LIST)(data_dir_path, type=models_list.TEST)
        elif training_type == 'fine_tuning':
            train_data_names, train_data_source, train_labels = models_list.get_property(model_name, models_list.LOAD_DATA_LIST)(data_dir_path, type=models_list.VALIDATE)
            t_data_names, t_data_source, t_labels = models_list.get_property(model_name, models_list.LOAD_DATA_LIST)(data_dir_path, type=models_list.TEST)

            half_t_size = int(len(t_labels) / 2)

            val_data_names = t_data_names[:half_t_size]
            val_labels = t_labels[:half_t_size]
            test_data_names = t_data_names[half_t_size:]
            test_labels = t_labels[half_t_size:]

            if isinstance(t_data_source, str):
                val_data_source = test_data_source = t_data_source
            else:
                val_data_source = t_data_source[:half_t_size, ...]
                test_data_source = t_data_source[half_t_size:, ...]


        else:
            raise Exception('Not yet supported!')

        if no_gpu <= 1:
            # Load the model
            model = models_list.get_property(model_name, models_list.INIT_METHOD)()

            if training_type == 'from_scratch':
                set_weights_to_randoms(model)
        else:
            with tf.device('/cpu:0'):
                cpu_model = models_list.get_property(model_name, models_list.INIT_METHOD)()

            if training_type == 'from_scratch':
                set_weights_to_randoms(cpu_model)

            model = multi_gpu_model(cpu_model, gpus=no_gpu)

        model.compile(optimizer="sgd", loss='categorical_crossentropy', metrics=[top_1_accuracy])
        print("Done load model " + model_name)

        batch_size = models_list.get_property(model_name, models_list.TRAINING_BATCH)

        # NO_EPOCHS = 1
        # NO_EPOCHS = 100
        NO_EPOCHS = 100

        augmented_data_path = args.result_dir + '/data/' + dataset

        train_data_generator = DataGenerator(model_name, train_data_names, augmented_data_path, to_categorical(train_labels), batch_size * no_gpu)
        val_data_generator = DataGenerator(model_name, val_data_names, augmented_data_path, to_categorical(val_labels), batch_size * no_gpu)
        test_data_generator = DataGenerator(model_name, test_data_names, augmented_data_path, to_categorical(test_labels), batch_size * no_gpu)

        # previous_accu = 0
        best_epoch = 0
        best_accu = 0
        best_test_accu = 0

        MAX_EPOCH_DELAY = 5
        delay = 0

        test_loss, test_accu = model.evaluate_generator(test_data_generator)
        val_loss, val_accu = model.evaluate_generator(val_data_generator)
        if training_type == 'fine_tuning':
            train_loss, train_accu = model.evaluate_generator(train_data_generator)
            p_out_f.write(
                '%s, %s, %s, %d, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f\n' % (args.backend, computation, model_name, -1, 0, val_loss, val_accu, test_loss, test_accu, train_loss, train_accu))
        else:
            p_out_f.write('%s, %s, %s, %d, %.5f, %.5f, %.5f, %.5f, %.5f\n' % (args.backend, computation, model_name, -1, 0, val_loss, val_accu, test_loss, test_accu))

        p_out_f.flush()

        for e in range(NO_EPOCHS):
            # Train and time the process
            begin = time.time()

            model.fit_generator(train_data_generator, epochs=1, shuffle=isRandom, use_multiprocessing=True, workers=12)

            end = time.time()

            test_loss, test_accu = model.evaluate_generator(test_data_generator)
            val_loss, val_accu = model.evaluate_generator(val_data_generator)
            print("EPOCH: %d VAL_LOSS: %.5f VAL_ACCU: %.5f" % (e, val_loss, val_accu))

            # store performance
            if training_type == 'fine_tuning':
                train_loss, train_accu = model.evaluate_generator(train_data_generator)
                p_out_f.write('%s, %s, %s, %d, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f\n' % (
                    args.backend, computation, model_name, e, (end - begin), val_loss, val_accu, test_loss, test_accu, train_loss, train_accu))
            else:
                p_out_f.write('%s, %s, %s, %d, %.5f, %.5f, %.5f, %.5f, %.5f\n' % (args.backend, computation, model_name, e, (end - begin), val_loss, val_accu, test_loss, test_accu))

            p_out_f.flush()

            # if val_accu < previous_accu:
            #
            if val_accu > best_accu:
                delay = 0
                best_loss = val_loss
                best_accu = val_accu
                best_test_accu = test_accu
                best_epoch = e
                # store the weight
                model.save_weights(args.result_dir + '/' + dataset + '/' + weight_filename)
            elif delay < MAX_EPOCH_DELAY:
                delay = delay + 1
            else:
                break

            # previous_accu = val_accu

        end_loss = val_loss
        end_accu = val_accu
        end_epoch = e

        end_run = time.time()
        
        

        done_out_f.write('%s,%s,%s,%s,%s,%s,%s,%d,%s,%d' % (
            computation, args.keras_version, args.backend, args.backend_version, args.cuda_version, args.cudnn_version, model_name, random_seed, training_type, iTry))
        done_out_f.write(',True,%.5f,%d,%.5f,%.5f,%d,%.5f,%.5f,%.5f\n' % (end_run - begin_run, end_epoch, end_loss, end_accu, best_epoch, best_loss, best_accu, best_test_accu))

        # Delete model and close stream
        if no_gpu > 1:
            del cpu_model
        del model
        del train_data_generator
        del val_data_generator
        gc.collect()

    except Exception:
        except_filename = 'train_exception_%s_%s_%s_%s_%s_%s_%d_%s.txt' % (
            args.keras_version, args.backend, args.backend_version, args.cuda_version, args.cudnn_version, model_name, random_seed, training_type)
        e_path = args.result_dir + '/' + dataset + '/' + except_filename
        running_utils.create_directory(e_path)
        e_out_f = open(e_path, "w")
        e_out_f.write(traceback.format_exc())
        e_out_f.close()

        done_out_f.write(
            '%s,%s,%s,%s,%s,%s,%s,%s,%s,%d' % (
            computation, args.keras_version, args.backend, args.backend_version, args.cuda_version, args.cudnn_version, model_name, random_seed, training_type, iTry))
        done_out_f.write(',False\n')
    finally:
        if p_out_f is not None:
            p_out_f.close()
            
    '''

    done_out_f.write('%s,%s,%s,%s,%s,%s,%s,%d,%s,%d\n' % (
        computation, args.keras_version, args.backend, args.backend_version, args.cuda_version, args.cudnn_version, model_name, random_seed, training_type, iTry))

    # Close done file
    done_out_f.close()


if __name__ == "__main__":
    if (K.backend() == 'tensorflow'):
        import tensorflow as tf

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        K.set_session(sess)
    main()
