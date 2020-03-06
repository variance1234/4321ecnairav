import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import scipy.stats as scst

from pathlib import Path

from models import models_list

import running_utils
import analyze_utils

import h5py

import time

import statistics as st

from numpy.testing import dec


def main():
    result_dir = './result_per_epoch_re_run'

    random_seed_list = [-1, 1]

    stopping_type_list = ['best_loss', 'best_acc', 'epoch']

    model_names = ['LeNet1', 'LeNet4', 'LeNet5',
                   'ResNet38v1', 'ResNet56v1',
                   'WRN-28-10']

    lib_config_list = [{'keras_version': '2.2.2', 'backend': 'tensorflow', 'backend_version': '1.10.0',
                        'lower_libs': [{'cuda_version': '9.0', 'cudnn_version': '7.3'},
                                       {'cuda_version': '9.0', 'cudnn_version': '7.4'},
                                       {'cuda_version': '9.0', 'cudnn_version': '7.5'},
                                       {'cuda_version': '9.0', 'cudnn_version': '7.6'}],
                        'random_seed_list': [1, -1]
                        },
                       {'keras_version': '2.2.2', 'backend': 'tensorflow', 'backend_version': '1.12.0',
                        'lower_libs': [{'cuda_version': '9.0', 'cudnn_version': '7.3'},
                                       {'cuda_version': '9.0', 'cudnn_version': '7.4'},
                                       {'cuda_version': '9.0', 'cudnn_version': '7.5'},
                                       {'cuda_version': '9.0', 'cudnn_version': '7.6'}],
                        'no_try': 16,
                        'random_seed_list': [1, -1]
                        },
                       {'keras_version': '2.2.2', 'backend': 'tensorflow', 'backend_version': '1.14.0',
                        'lower_libs': [{'cuda_version': '10.0', 'cudnn_version': '7.4'},
                                       {'cuda_version': '10.0', 'cudnn_version': '7.5'},
                                       {'cuda_version': '10.0', 'cudnn_version': '7.6'}],
                        'no_try': 16,
                        'random_seed_list': [1, -1]
                        }]

    # RQ2 overall stat test
    result_path = result_dir + '/' + 'RQ1_stat.csv'
    result_f = open(result_path, "w")

    # RQ2 stat test between network
    result_f.write('Levene test overall\n')
    print_levene_test('/analysis_raw.csv', model_names, lib_config_list, result_dir, result_f)

    result_f.write('\n\nLevene test per_class\n')
    print_levene_test('/analysis_per_raw.csv', model_names, lib_config_list, result_dir, result_f)

    # RQ1 vs 2 between seed
    result_f.write('\n\nLevene test overall between seed (-1 vs 1)\n')
    print_levene_test_between_random_seed('/analysis_raw.csv', model_names, lib_config_list, result_dir, result_f)

    result_f.write('\n\nLevene test per_class between seed (-1 vs 1)\n')
    print_levene_test_between_random_seed('/analysis_per_raw.csv', model_names, lib_config_list, result_dir, result_f)

    # convergent time, random seed=1
    result_f.write('\n\nLevene test between stopping overall + convergent + random seed=1\n')
    print_leveneor_u_test_between_stopping_type('/analysis_raw.csv', model_names, lib_config_list, result_dir, result_f,
                                                time_or_epoch='convergent', levene_or_U='levene', random_seed=1)

    # convergent epoch, random seed=1
    result_f.write('\n\nLevene test between stopping overall + convergent epoch + ranndom seed=1\n')
    print_leveneor_u_test_between_stopping_type('/analysis_raw.csv', model_names, lib_config_list, result_dir, result_f,
                                                time_or_epoch='convergent_epoch', levene_or_U='levene', random_seed=1)

    # convergent time, random seed=-1
    result_f.write('\n\nLevene test between stopping overall + convergent + random seed=-1\n')
    print_leveneor_u_test_between_stopping_type('/analysis_raw.csv', model_names, lib_config_list, result_dir, result_f,
                                                time_or_epoch='convergent', levene_or_U='levene', random_seed=-1)

    # convergent epoch, random seed=-1
    result_f.write('\n\nLevene test between stopping overall + convergent epoch + ranndom seed=-1\n')
    print_leveneor_u_test_between_stopping_type('/analysis_raw.csv', model_names, lib_config_list, result_dir, result_f,
                                                time_or_epoch='convergent_epoch', levene_or_U='levene', random_seed=-1)

    # convergent time, random seed=1
    result_f.write('\n\nU test between stopping overall + convergent + random seed=1\n')
    print_leveneor_u_test_between_stopping_type('/analysis_raw.csv', model_names, lib_config_list, result_dir, result_f,
                                                time_or_epoch='convergent', levene_or_U='U', random_seed=1)

    # convergent epoch, random seed=1
    result_f.write('\n\nU test between stopping overall + convergent epoch + ranndom seed=1\n')
    print_leveneor_u_test_between_stopping_type('/analysis_raw.csv', model_names, lib_config_list, result_dir, result_f,
                                                time_or_epoch='convergent_epoch', levene_or_U='U', random_seed=1)

    # convergent time, random seed=-1
    result_f.write('\n\nU test between stopping overall + convergent + random seed=-1\n')
    print_leveneor_u_test_between_stopping_type('/analysis_raw.csv', model_names, lib_config_list, result_dir, result_f,
                                                time_or_epoch='convergent', levene_or_U='U', random_seed=-1)

    # convergent epoch, random seed=-1
    result_f.write('\n\nU test between stopping overall + convergent epoch + ranndom seed=-1\n')
    print_leveneor_u_test_between_stopping_type('/analysis_raw.csv', model_names, lib_config_list, result_dir, result_f,
                                                time_or_epoch='convergent_epoch', levene_or_U='U', random_seed=-1)

    # accuracy, random seed=1, levene
    result_f.write('\n\nLevene test overall + accuracy + ranndom seed=1\n')
    print_levene_or_U_test_between_backend('/analysis_raw.csv', model_names, lib_config_list, result_dir, result_f,
                                           accuracy_or_epoch='accuracy', levene_or_U='levene', random_seed=1)

    # accuracy, random seed=1, U
    result_f.write('\n\nU test overall + accuracy + ranndom seed=1\n')
    print_levene_or_U_test_between_backend('/analysis_raw.csv', model_names, lib_config_list, result_dir, result_f,
                                           accuracy_or_epoch='accuracy', levene_or_U='U', random_seed=1)

    # accuracy, random seed=1, levene
    result_f.write('\n\nLevene test overall + convergent_epoch + ranndom seed=1\n')
    print_levene_or_U_test_between_backend('/analysis_raw.csv', model_names, lib_config_list, result_dir, result_f,
                                           accuracy_or_epoch='convergent_epoch', levene_or_U='levene', random_seed=1)

    # accuracy, random seed=1, U
    result_f.write('\n\nU test overall + convergent_epoch + ranndom seed=1\n')
    print_levene_or_U_test_between_backend('/analysis_raw.csv', model_names, lib_config_list, result_dir, result_f,
                                           accuracy_or_epoch='convergent_epoch', levene_or_U='U', random_seed=1)

    result_f.close()


def print_levene_test(raw_file, model_names, lib_config_list, result_dir, result_f):
    data = pd.read_csv(result_dir + raw_file, skipinitialspace=True)
    data['backend_version'] = data['backend_version'].astype(str)
    data['cuda_version'] = data['cuda_version'].astype(str)
    data['cudnn_version'] = data['cudnn_version'].astype(str)

    result_f.write('network')
    for baseline in model_names:
        result_f.write(',' + baseline)
    result_f.write('\n')
    tensorflow_data = data[data['backend'] == 'tensorflow']
    random_seed = 1
    for subject in model_names:
        result_f.write(subject)
        for baseline in model_names:
            subject_data = tensorflow_data[(tensorflow_data['network'].values == subject) & (tensorflow_data['random_seed'].values == random_seed)]
            subject_accuracies = get_accuracies(subject_data, lib_config_list)

            baseline_data = tensorflow_data[(tensorflow_data['network'].values == baseline) & (tensorflow_data['random_seed'].values == random_seed)]
            baseline_accuracies = get_accuracies(baseline_data, lib_config_list)

            w, p_value = scst.levene(subject_accuracies, baseline_accuracies)

            result_f.write(',' + str(p_value))

        result_f.write('\n')


def print_levene_test_between_random_seed(raw_file, model_names, lib_config_list, result_dir, result_f):
    data = pd.read_csv(result_dir + raw_file, skipinitialspace=True)
    data['backend_version'] = data['backend_version'].astype(str)
    data['cuda_version'] = data['cuda_version'].astype(str)
    data['cudnn_version'] = data['cudnn_version'].astype(str)

    result_f.write('network')
    for baseline in model_names:
        result_f.write(',' + baseline)
    result_f.write('\n')
    tensorflow_data = data[data['backend'] == 'tensorflow']
    result_f.write('p value')
    for subject in model_names:
        network_data = tensorflow_data[(tensorflow_data['network'].values == subject)]
        print(subject)
        random_seed = 1
        print(random_seed)
        subject_data = network_data[(network_data['random_seed'].values == random_seed)]
        subject_accuracies = get_accuracies(subject_data, lib_config_list)

        random_seed = -1
        print(random_seed)
        baseline_data = network_data[(network_data['random_seed'].values == random_seed)]
        baseline_accuracies = get_accuracies(baseline_data, lib_config_list)

        w, p_value = scst.levene(subject_accuracies, baseline_accuracies)

        result_f.write(',' + str(p_value))
    result_f.write('\n')


def print_leveneor_u_test_between_stopping_type(raw_file, model_names, lib_config_list, result_dir, result_f,
                                                time_or_epoch='convergent', levene_or_U='levene', random_seed=1):
    data = pd.read_csv(result_dir + raw_file, skipinitialspace=True)
    data['backend_version'] = data['backend_version'].astype(str)
    data['cuda_version'] = data['cuda_version'].astype(str)
    data['cudnn_version'] = data['cudnn_version'].astype(str)

    result_f.write('network')
    for baseline in model_names:
        result_f.write(',' + baseline)
    result_f.write('\n')
    tensorflow_data = data[data['backend'] == 'tensorflow']
    result_f.write('p value')
    for subject in model_names:
        network_data = tensorflow_data[(tensorflow_data['network'].values == subject)]
        print(subject)
        stopping_type = 'best_loss'
        print(stopping_type)
        subject_data = network_data[(network_data['stopping_type'].values == stopping_type)]
        subject_conv, subject_acc = get_convergent(subject_data, lib_config_list, time_or_epoch, random_seed)

        stopping_type = 'best_acc'
        print(stopping_type)
        baseline_data = network_data[(network_data['stopping_type'].values == stopping_type)]
        baseline_conv, baseline_acc = get_convergent(baseline_data, lib_config_list, time_or_epoch, random_seed)

        if levene_or_U == 'levene':
            _, conv_p_value = scst.levene(subject_conv, baseline_conv)
            _, acc_p_value = scst.levene(subject_acc, baseline_acc)
        elif levene_or_U == 'U':
            if st.variance(subject_conv) == 0 and st.variance(baseline_conv) == 0 and subject_conv[0] == baseline_conv[0]:
                conv_p_value = np.nan
            elif st.variance(subject_acc) == 0 and st.variance(baseline_acc) == 0 and subject_conv[0] == baseline_acc[0]:
                acc_p_value = np.nan
            else:
                _, conv_p_value = scst.mannwhitneyu(subject_conv, baseline_conv)
                _, acc_p_value = scst.mannwhitneyu(subject_acc, baseline_acc)
        else:
            raise Exception('Unknown test!')

        result_f.write(',%.2f/%.2f' %(conv_p_value, acc_p_value))
    result_f.write('\n')


def print_levene_or_U_test_between_backend(raw_file, model_names, lib_config_list, result_dir, result_f,
                                           accuracy_or_epoch='accuracy', levene_or_U='levene', random_seed=1):
    assert accuracy_or_epoch in ['accuracy', 'convergent_epoch']

    data = pd.read_csv(result_dir + raw_file, skipinitialspace=True)
    data['backend_version'] = data['backend_version'].astype(str)
    data['cuda_version'] = data['cuda_version'].astype(str)
    data['cudnn_version'] = data['cudnn_version'].astype(str)

    result_f.write('p value ' + levene_or_U + ' test')
    for baseline in model_names:
        result_f.write(',' + baseline)
    result_f.write('\n')
    backend_pairs = [['tensorflow', 'cntk'],
                     ['cntk', 'theano'],
                     ['theano', 'tensorflow']]

    lib_data = data[(data['cuda_version'] == '10.0') & (data['cudnn_version'] == '7.6') & (data['random_seed'] == random_seed)]
    for backend_pair in backend_pairs:
        backend1 = backend_pair[0]
        backend2 = backend_pair[1]
        result_f.write(backend1 + ' vs ' + backend2)
        for subject in model_names:
            network_data = lib_data[(lib_data['network'].values == subject)]
            print(subject)
            subject_data = network_data[(network_data['backend'].values == backend1)]
            # subject_accuracies = get_accuracy_or_convergent(subject_data, lib_config_list, accuracy_or_epoch)
            subject_data = subject_data[(subject_data['stopping_type'].values == 'best_loss')]
            subject_accuracies = subject_data[accuracy_or_epoch].values

            baseline_data = network_data[(network_data['backend'].values == backend2)]
            # baseline_accuracies = get_accuracy_or_convergent(baseline_data, lib_config_list, accuracy_or_epoch)
            baseline_data = baseline_data[(baseline_data['stopping_type'].values == 'best_loss')]
            baseline_accuracies = baseline_data[accuracy_or_epoch].values

            if levene_or_U == 'levene':
                w, p_value = scst.levene(subject_accuracies, baseline_accuracies)
            elif levene_or_U == 'U':
                if st.variance(subject_accuracies) == 0 and st.variance(baseline_accuracies) == 0 and subject_accuracies[0] == baseline_accuracies[0]:
                    p_value = np.nan
                else:
                    w, p_value = scst.mannwhitneyu(subject_accuracies, baseline_accuracies)
            else:
                raise Exception('Unknown test!')

            result_f.write(',' + str(p_value))
        result_f.write('\n')


def get_accuracies(data, lib_config_list):
    best_var = -1
    best_acc = []  # accuracy of the batch with max standard deviation

    best_config = {}

    for lib_config in lib_config_list:
        lower_libs = lib_config['lower_libs']
        for lower_lib_config in lower_libs:
            for stopping_type in ['best_loss', 'best_acc']:
                data_batch = data[(data['backend_version'].values == lib_config['backend_version']) &
                                  (data['cuda_version'].values == lower_lib_config['cuda_version']) &
                                  (data['cudnn_version'].values == lower_lib_config['cudnn_version']) &
                                  (data['stopping_type'].values == stopping_type)]
                accuracy = data_batch['accuracy'].values
                variance = st.variance(accuracy)
                # print(variance)
                if variance > best_var:
                    best_var = variance
                    best_acc = accuracy
                    best_config['backend_version'] = lib_config['backend_version']
                    best_config['cuda_version'] = lower_lib_config['cuda_version']
                    best_config['cudnn_version'] = lower_lib_config['cudnn_version']
                    best_config['stopping_type'] = stopping_type

    # print(best_config)
    # print(best_var)
    # print(best_acc)
    return best_acc


def get_convergent(data, lib_config_list, time_or_epoch='convergent', random_seed=1):
    assert time_or_epoch in ['convergent', 'convergent_epoch']
    best_var = -1
    best_conv = []  # accuracy of the batch with max standard deviation
    best_accuracy = []

    best_config = {}

    for lib_config in lib_config_list:
        lower_libs = lib_config['lower_libs']
        for lower_lib_config in lower_libs:
            data_batch = data[(data['backend_version'].values == lib_config['backend_version']) &
                              (data['cuda_version'].values == lower_lib_config['cuda_version']) &
                              (data['cudnn_version'].values == lower_lib_config['cudnn_version']) &
                              (data['random_seed'].values == random_seed)]
            convergent = data_batch[time_or_epoch].values
            accuracy = data_batch['accuracy'].values
            variance = st.variance(convergent)
            if variance > best_var:
                best_var = variance
                best_conv = convergent
                best_accuracy = accuracy
    print(best_var)
    print(best_conv)
    return best_conv, best_accuracy


def get_accuracy_or_convergent(data, lib_config_list, accuracy_or_epoch='convergent'):
    assert accuracy_or_epoch in ['accuracy', 'convergent_epoch']
    best_var = -1
    best_conv = []  # accuracy of the batch with max standard deviation

    best_config = {}

    for stopping_type in ['best_loss', 'best_acc']:
        data_batch = data[(data['stopping_type'].values == stopping_type)]
        convergent = data_batch[accuracy_or_epoch].values
        variance = st.variance(convergent)
        if variance > best_var:
            best_var = variance
            best_conv = convergent
    print(best_var)
    print(best_conv)
    return best_conv


if __name__ == "__main__":
    main()
