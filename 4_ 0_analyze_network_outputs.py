import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path

from models import models_list

import running_utils
import analyze_utils

import h5py

import time

import statistics as st


def cal_accuracy(outputs, labels):
    predicted_labels = list(np.argmax(outputs, axis=1))

    accuracy = sum(np.equal(predicted_labels, labels)) / len(labels)

    per_label_accuracy = []
    per_label_no_samples = []
    for l in range(max(labels)):
        l_predicteds = [predicted_labels[l_index] for l_index in range(len(labels)) if labels[l_index] == l]
        l_labels = [l] * len(l_predicteds)
        l_accuracy = sum(np.equal(l_predicteds, l_labels)) / len(l_labels)
        per_label_accuracy.append(l_accuracy)
        per_label_no_samples.append(len(l_labels))

    return accuracy, per_label_accuracy, per_label_no_samples


def cal_runtime(stopping_type, per_data):
    epochs = per_data['epoch'].values
    times = per_data['time'].values

    if stopping_type == 'best_acc':
        monitors = per_data['val_acc'].values
        best_monitor_index = np.argmax(monitors[1:]) + 1
    elif stopping_type == 'best_loss':
        monitors = per_data['val_loss'].values
        best_monitor_index = np.argmin(monitors[1:]) + 1
    else:
        best_monitor_index = len(epochs) - 1

    convergent_running_time = 0
    per_epoch_running_time = 0
    first_epoch_running_time = 0

    for i in range(len(epochs)):
        epoch = epochs[i]

        if i <= best_monitor_index:
            convergent_running_time += times[i]

        if epoch == 0:
            first_epoch_running_time = times[i]

        if epoch > 0:
            per_epoch_running_time += times[i]

    per_epoch_running_time /= (len(epochs) - 2)

    return best_monitor_index, convergent_running_time, per_epoch_running_time, first_epoch_running_time


def get_last_epoch_accuracy(per_data):
    test_accs = per_data['test_acc'].values

    return test_accs[-1]


def find_max_accuracy_diff(acc_raw_f, acc_per_raw_f, result_dir_path, no_try, labels, comp, kv, back, back_v, cu_v, cudnn_v, model_name, random_seed, training_type, stopping_type):
    accuracies = []
    per_label_accuracies = []

    convergent_epochs = []
    convergent_running_times = []
    per_epoch_running_times = []
    first_epoch_running_times = []

    for iTry in range(no_try):
        # Calculate running time
        performance_filename = 'train_performance_%s_%s_%s_%s_%s_%s_%d_%s_%d.csv' % (
            kv, back, back_v, cu_v, cudnn_v, model_name, random_seed, training_type, iTry)

        per_p = result_dir_path + '/' + performance_filename
        per_path = Path(per_p)
        if not per_path.is_file():
            break
        per_data = pd.read_csv(per_p, skipinitialspace=True)

        if len(per_data['test_acc'].values) == 0:
            return None

        if stopping_type != 'epoch':
            # Calculate accuracy
            output_filename = stopping_type + '_output_%s_%s_%s_%s_%s_%s_%d_%s_%d.h5' % (
                kv, back, back_v, cu_v, cudnn_v, model_name, random_seed, training_type, iTry)

            out_p = result_dir_path + '/' + output_filename
            out_path = Path(out_p)
            if not out_path.is_file():
                break

            hf = h5py.File(out_p, 'r')
            outputs = np.array(hf['Outputs'])
            hf.close()

            accuracy, per_label_accuracy, per_label_no_samples = cal_accuracy(outputs, labels)
            accuracies.append(accuracy)
            per_label_accuracies.append(per_label_accuracy)

        else:
            accuracy = get_last_epoch_accuracy(per_data)
            accuracies.append(accuracy)

        convergent_epoch, convergent_running_time, per_epoch_running_time, first_epoch_running_time = cal_runtime(stopping_type, per_data)
        convergent_epochs.append(convergent_epoch)
        convergent_running_times.append(convergent_running_time)
        per_epoch_running_times.append(per_epoch_running_time)
        first_epoch_running_times.append(first_epoch_running_time)

        acc_raw_f.write('%s,%s,%s,%s,%s,%s,%s,%d,%s,%s,%d' %
                        (comp, kv, back, back_v, cu_v, cudnn_v, model_name, random_seed, training_type, stopping_type, iTry))
        acc_raw_f.write(',%.5f,%.5f,%d,%.5f,%.5f\n' %
                        (accuracy,
                         convergent_running_time,
                         convergent_epoch,
                         per_epoch_running_time,
                         first_epoch_running_time))

    if len(accuracies) == 0:
        return None

    # calculate stat for accuracy
    max_accuracy_diff = max(accuracies) - min(accuracies)
    max_accuracy = max(accuracies)
    min_accuracy = min(accuracies)
    std_dev_accuracy = st.stdev(accuracies)
    mean_accuracy = st.mean(accuracies)

    # calculate stat for running time
    max_convergent_diff_epoch = max(convergent_epochs) - min(convergent_epochs)
    max_convergent_epoch = max(convergent_epochs)
    min_convergent_epoch = min(convergent_epochs)
    std_dev_convergent_epoch = st.stdev(map(float, convergent_epochs))
    mean_convergent_epoch = st.mean(map(float, convergent_epochs))

    max_convergent_diff = max(convergent_running_times) - min(convergent_running_times)
    max_convergent = max(convergent_running_times)
    min_convergent = min(convergent_running_times)
    std_dev_convergent = st.stdev(map(float, convergent_running_times))
    mean_convergent = st.mean(map(float, convergent_running_times))

    max_per_epoch_diff = max(per_epoch_running_times) - min(per_epoch_running_times)
    max_per_epoch = max(per_epoch_running_times)
    min_per_epoch = min(per_epoch_running_times)
    std_dev_per_epoch = st.stdev(per_epoch_running_times)
    mean_per_epoch = st.mean(per_epoch_running_times)

    max_first_epoch_diff = max(first_epoch_running_times) - min(first_epoch_running_times)
    max_first_epoch = max(first_epoch_running_times)
    min_first_epoch = min(first_epoch_running_times)
    std_dev_first_epoch = st.stdev(first_epoch_running_times)
    mean_first_epoch = st.mean(first_epoch_running_times)

    if stopping_type != 'epoch':
        per_label_accuracies = np.array(per_label_accuracies)

        max_diff_label = -1
        max_per_label_acc_diff = 0
        max_label_accuracy = 0
        min_label_accuracy = 0

        max_std_label = -1
        max_per_label_acc_std = 0

        for l in range(max(labels)):
            label_accu_diff = max(per_label_accuracies[:, l]) - min(per_label_accuracies[:, l])
            label_accu_std = st.stdev(per_label_accuracies[:, l])
            if label_accu_diff > max_per_label_acc_diff:
                max_per_label_acc_diff = label_accu_diff
                max_diff_label = l
                max_label_accuracy = max(per_label_accuracies[:, l])
                min_label_accuracy = min(per_label_accuracies[:, l])

            if label_accu_std > max_per_label_acc_std:
                max_per_label_acc_std = label_accu_std
                max_std_label = l

        per_class_accuracies = per_label_accuracies[:, max_std_label]
        for iTry in range(len(per_class_accuracies)):
            acc_per_raw_f.write('%s,%s,%s,%s,%s,%s,%s,%d,%s,%s,%d' %
                                (comp, kv, back, back_v, cu_v, cudnn_v, model_name, random_seed, training_type, stopping_type, iTry))
            acc_per_raw_f.write(',%d,%.5f\n' %
                                (max_std_label, per_class_accuracies[iTry]))

        return (len(accuracies),
                max_accuracy_diff, max_accuracy, min_accuracy, std_dev_accuracy, mean_accuracy,
                max_diff_label, max_per_label_acc_diff, max_label_accuracy, min_label_accuracy, per_label_no_samples[max_diff_label],
                max_std_label, max_per_label_acc_std, per_label_no_samples[max_std_label],
                max_convergent_diff, max_convergent, min_convergent, std_dev_convergent, mean_convergent,
                max_convergent_diff_epoch, max_convergent_epoch, min_convergent_epoch, std_dev_convergent_epoch, mean_convergent_epoch,
                max_per_epoch_diff, max_per_epoch, min_per_epoch, std_dev_per_epoch, mean_per_epoch,
                max_first_epoch_diff, max_first_epoch, min_first_epoch, std_dev_first_epoch, mean_first_epoch)

    return (len(accuracies),
            max_accuracy_diff, max_accuracy, min_accuracy, std_dev_accuracy, mean_accuracy,
            -1, -1, -1, -1, -1,
            -1, -1, -1,
            max_convergent_diff, max_convergent, min_convergent, std_dev_convergent, mean_convergent,
            max_convergent_diff_epoch, max_convergent_epoch, min_convergent_epoch, std_dev_convergent_epoch, mean_convergent_epoch,
            max_per_epoch_diff, max_per_epoch, min_per_epoch, std_dev_per_epoch, mean_per_epoch,
            max_first_epoch_diff, max_first_epoch, min_first_epoch, std_dev_first_epoch, mean_first_epoch)


def main():
    # read the parameter
    # argument parsing
    result_dir = "result_per_epoch_re_run"
    data_dir = "../crossmodelchecking/data"

    no_try = 16

    computation = '1_gpu'
    # backends = ['tensorflow']
    # backends = ['theano', 'tensorflow', 'cntk']

    # lower_lib_config_list = [{'cuda_version': '9.0', 'cudnn_version': '7.6'}]
    lower_lib_config_list = [{'cuda_version': '9.0', 'cudnn_version': '7.4'},
                             {'cuda_version': '9.0', 'cudnn_version': '7.3'},
                             {'cuda_version': '9.0', 'cudnn_version': '7.5'},
                             {'cuda_version': '9.0', 'cudnn_version': '7.6'},
                             {'cuda_version': '10.0', 'cudnn_version': '7.4'},
                             {'cuda_version': '10.0', 'cudnn_version': '7.3'},
                             {'cuda_version': '10.0', 'cudnn_version': '7.5'},
                             {'cuda_version': '10.0', 'cudnn_version': '7.6'}
                             ]
    lib_config_list = [{'keras_version': '2.2.2', 'backend': 'tensorflow', 'backend_version': '1.10.0'},
                       {'keras_version': '2.2.2', 'backend': 'tensorflow', 'backend_version': '1.12.0'},
                       {'keras_version': '2.2.2', 'backend': 'tensorflow', 'backend_version': '1.14.0'},
                       {'keras_version': '2.2.2', 'backend': 'cntk', 'backend_version': '2.7'},
                       {'keras_version': '2.2.2', 'backend': 'theano', 'backend_version': '1.0.4'}]

    random_seed_list = [-1, 0, 1]
    random_seed_list = [-1, 1]
    # random_seed_list = [0]
    no_gpu_list = [1]
    # training_type_list = ['fine_tuning']
    training_type_list = ['from_scratch']

    #stopping_type_list = ['best_loss', 'best_acc', 'epoch']
    stopping_type_list = ['best_loss', 'best_acc']
    # stopping_type_list = ['best_acc']

    model_names = ['LeNet1', 'LeNet4', 'LeNet5',
                   'ResNet38v1', 'ResNet56v1',
                   'WRN-28-10']

    # model_names = ['ResNet38v1']

    acc_raw_path = result_dir + '/analysis_raw.csv'
    acc_raw_f = open(acc_raw_path, "w")
    acc_raw_f.write('comp,keras,backend,backend_version,cuda_version,cudnn_version,network,random_seed,training_type,stopping_type,try,' +
                    'accuracy,' +
                    'convergent,' +
                    'convergent_epoch,' +
                    'per_epoch_average,'
                    'first_epoch_average,\n')

    acc_per_raw_path = result_dir + '/analysis_per_raw.csv'
    acc_per_raw_f = open(acc_per_raw_path, "w")
    acc_per_raw_f.write('comp,keras,backend,backend_version,cuda_version,cudnn_version,network,random_seed,training_type,stopping_type,try,' +
                        'class,' +
                        'accuracy\n')

    acc_path = result_dir + '/analysis_result.csv'
    acc_f = open(acc_path, "w")
    acc_f.write('comp,keras,backend,backend_version,cuda_version,cudnn_version,network,random_seed,training_type,stopping_type,' +
                'no_try,' +
                'max_accuracy_diff,max_accuracy,min_accuracy,std_dev_accuracy,mean_accuracy,' +
                'max_diff_label,max_per_label_acc_diff,max_label_accuracy,min_label_accuracy,no_samples_max_diff,' +
                'max_std_label,max_per_label_acc_std,no_samples_max_std,' +
                'max_convergent_diff,max_convergent,min_convergent,std_dev_convergent,mean_convergent,' +
                'max_convergent_diff_epoch,max_convergent_epoch,min_convergent_epoch,std_dev_convergent_epoch,mean_convergent_epoch,' +
                'max_per_epoch_diff,max_per_epoch,min_per_epoch,std_dev_per_epoch,mean_per_epoch,'
                'max_first_epoch_diff,max_first_epoch,min_first_epoch,std_dev_first_epoch,mean_first_epoch\n')

    models_list.import_model()

    for random_seed in random_seed_list:
        for model_name in model_names:

            dataset = models_list.get_property(model_name, models_list.DATA_DIR)

            data_dir_path = data_dir + '/' + dataset
            result_dir_path = result_dir + '/' + dataset

            # Load Data List
            data_names, data_source, labels = models_list.get_property(model_name, models_list.LOAD_DATA_LIST)(
                data_dir_path, type=models_list.TEST)

            for stopping_type in stopping_type_list:
                for lower_lib_config in lower_lib_config_list:
                    for lib_config in lib_config_list:
                        for training_type in training_type_list:
                            if lib_config['backend'] != "tensorflow" and random_seed == -1:
                                continue
                            print('Analyzing %s,%s,%s,%s,%s,%s,%s,%d,%s,%s\n' % (
                                computation, lib_config['keras_version'], lib_config['backend'], lib_config['backend_version'],
                                lower_lib_config['cuda_version'], lower_lib_config['cudnn_version'],
                                model_name, random_seed, training_type, stopping_type))

                            result = \
                                find_max_accuracy_diff(acc_raw_f, acc_per_raw_f, result_dir_path, no_try, labels,
                                                       computation, lib_config['keras_version'], lib_config['backend'], lib_config['backend_version'],
                                                       lower_lib_config['cuda_version'], lower_lib_config['cudnn_version'],
                                                       model_name, random_seed, training_type, stopping_type)

                            if result is None:
                                continue

                            (actual_no_try,
                             max_accuracy_diff, max_accuracy, min_accuracy, std_dev_accuracy, mean_accuracy,
                             max_diff_label, max_per_label_acc_diff, max_label_accuracy, min_label_accuracy, no_samples_max_diff,
                             max_std_label, max_per_label_acc_std, no_samples_max_std,
                             max_convergent_diff, max_convergent, min_convergent, std_dev_convergent, mean_convergent,
                             max_convergent_diff_epoch, max_convergent_epoch, min_convergent_epoch, std_dev_convergent_epoch, mean_convergent_epoch,
                             max_per_epoch_diff, max_per_epoch, min_per_epoch, std_dev_per_epoch, mean_per_epoch,
                             max_first_epoch_diff, max_first_epoch, min_first_epoch, std_dev_first_epoch, mean_first_epoch) = result

                            acc_f.write('%s,%s,%s,%s,%s,%s,%s,%d,%s,%s' % (
                                computation, lib_config['keras_version'], lib_config['backend'], lib_config['backend_version'],
                                lower_lib_config['cuda_version'], lower_lib_config['cudnn_version'],
                                model_name, random_seed, training_type, stopping_type))
                            acc_f.write(',%d,%.5f,%.5f,%.5f,%.5f,%.5f,%d,%.5f,%.5f,%.5f,%d,%d,%.5f,%d,%.5f,%.5f,%.5f,%.5f,%.5f,%d,%d,%d,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f\n' %
                                        (actual_no_try,
                                         max_accuracy_diff, max_accuracy, min_accuracy, std_dev_accuracy, mean_accuracy,
                                         max_diff_label, max_per_label_acc_diff, max_label_accuracy, min_label_accuracy, no_samples_max_diff,
                                         max_std_label, max_per_label_acc_std, no_samples_max_std,
                                         max_convergent_diff, max_convergent, min_convergent, std_dev_convergent, mean_convergent,
                                         max_convergent_diff_epoch, max_convergent_epoch, min_convergent_epoch, std_dev_convergent_epoch, mean_convergent_epoch,
                                         max_per_epoch_diff, max_per_epoch, min_per_epoch, std_dev_per_epoch, mean_per_epoch,
                                         max_first_epoch_diff, max_first_epoch, min_first_epoch, std_dev_first_epoch, mean_first_epoch))

    acc_per_raw_f.close()
    acc_raw_f.close()
    acc_f.close()


if __name__ == "__main__":
    main()
