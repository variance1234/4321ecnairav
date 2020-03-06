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


def extract_best(filtered_data, headers, model_result):
    if len(filtered_data[headers[0]].values) > 0:
        max_index = np.argmax(filtered_data[headers[0]].values)
        model_result.append([filtered_data[header].values[max_index] for header in headers])
    else:
        model_result.append([-1 for header in headers])


def print_values(inx, no_tables, model_names, random_seed_list, stopping_type_list, result_f, result):
    for random_seed in random_seed_list:
        for model_name in model_names:
            key = str(random_seed) + '_' + model_name

            result_f.write('%s' % key)

            model_result = result[key]
            for i in range(len(stopping_type_list)):
                single_result = model_result[i * no_tables + inx]
                for v in single_result:
                    result_f.write(',' + str(v))

            result_f.write('\n')


def extract_result(file_name, model_names, random_seed_list, result_dir, stopping_type_list, data):
    result = {}
    for random_seed in random_seed_list:
        for model_name in model_names:
            model_result = []
            result[str(random_seed) + '_' + model_name] = model_result

            for stopping_type in stopping_type_list:
                filtered_data = data[(data['network'].values == model_name) & (data['random_seed'].values == random_seed) & (data['stopping_type'].values == stopping_type)]

                headers = ['max_accuracy_diff', 'max_accuracy', 'min_accuracy', 'mean_accuracy']
                extract_best(filtered_data, headers, model_result)

                headers = ['max_per_label_acc_diff', 'max_label_accuracy', 'min_label_accuracy', 'max_diff_label']
                extract_best(filtered_data, headers, model_result)

                headers = ['max_convergent_diff', 'max_convergent', 'min_convergent', 'std_dev_convergent', 'mean_convergent']
                extract_best(filtered_data, headers, model_result)

                headers = ['max_convergent_diff_epoch', 'max_convergent_epoch', 'min_convergent_epoch', 'std_dev_convergent_epoch', 'mean_convergent_epoch']
                extract_best(filtered_data, headers, model_result)

                headers = ['max_per_epoch_diff', 'max_per_epoch', 'min_per_epoch', 'std_dev_per_epoch', 'mean_per_epoch']
                extract_best(filtered_data, headers, model_result)

                headers = ['max_first_epoch_diff', 'max_first_epoch', 'min_first_epoch', 'std_dev_first_epoch', 'mean_first_epoch']
                extract_best(filtered_data, headers, model_result)

                headers = ['std_dev_accuracy', 'mean_accuracy']
                extract_best(filtered_data, headers, model_result)

                headers = ['max_per_label_acc_std', 'max_std_label']
                extract_best(filtered_data, headers, model_result)

                headers = ['std_dev_convergent', 'mean_convergent']
                extract_best(filtered_data, headers, model_result)

                headers = ['std_dev_convergent_epoch', 'mean_convergent_epoch']
                extract_best(filtered_data, headers, model_result)

                headers = ['std_dev_per_epoch', 'mean_per_epoch']
                extract_best(filtered_data, headers, model_result)

                headers = ['std_dev_first_epoch', 'mean_first_epoch']
                extract_best(filtered_data, headers, model_result)

    result_path = result_dir + '/' + file_name
    result_f = open(result_path, "w")

    NO_TABLES = 12

    result_f.write('Overall accuracy result\n')
    result_f.write('Seed_Network,Loss,,,,Acc,,,,Epoch,,,\n')
    result_f.write(',Diff,Max,Min,Mean,Diff,Max,Min,Mean,Diff,Max,Min,Mean\n')
    print_values(0, NO_TABLES, model_names, random_seed_list, stopping_type_list, result_f, result)

    result_f.write('\nPer-class accuracy result\n')
    result_f.write('Seed_Network,Loss,,,,Acc,,,,Epoch,,,\n')
    result_f.write(',Diff,Max,Min,Label,Diff,Max,Min,Label,Diff,Max,Min,Label\n')
    print_values(1, NO_TABLES, model_names, random_seed_list, stopping_type_list, result_f, result)

    result_f.write('\nConv time result\n')
    result_f.write('Seed_Network,Loss,,,,,Acc,,,,,Epoch,,,,\n')
    result_f.write(',Diff,Max,Min,SDev,Mean,Diff,Max,Min,SDev,Mean,Diff,Max,Min,Sdev,Mean\n')
    print_values(2, NO_TABLES, model_names, random_seed_list, stopping_type_list, result_f, result)

    result_f.write('\nConv epoch result\n')
    result_f.write('Seed_Network,Loss,,,,,Acc,,,,,Epoch,,,,\n')
    result_f.write(',Diff,Max,Min,SDev,Mean,Diff,Max,Min,SDev,Mean,Diff,Max,Min,Sdev,Mean\n')
    print_values(3, NO_TABLES, model_names, random_seed_list, stopping_type_list, result_f, result)

    result_f.write('\nPer-epoch time result\n')
    result_f.write('Seed_Network,Loss,,,,,Acc,,,,,Epoch,,,,\n')
    result_f.write(',Diff,Max,Min,SDev,Mean,Diff,Max,Min,SDev,Mean,Diff,Max,Min,Sdev,Mean\n')
    print_values(4, NO_TABLES, model_names, random_seed_list, stopping_type_list, result_f, result)

    result_f.write('\nFirst-epoch time result\n')
    result_f.write('Seed_Network,Loss,,,,,Acc,,,,,Epoch,,,,\n')
    result_f.write(',Diff,Max,Min,SDev,Mean,Diff,Max,Min,SDev,Mean,Diff,Max,Min,Sdev,Mean\n')
    print_values(5, NO_TABLES, model_names, random_seed_list, stopping_type_list, result_f, result)

    result_f.write('\nOverall std dev acc result\n')
    result_f.write('Seed_Network,Loss,,Acc,,Epoch,\n')
    result_f.write(',StdDev,Mean,StdDev,Mean,StdDev,Mean\n')
    print_values(6, NO_TABLES, model_names, random_seed_list, stopping_type_list, result_f, result)

    result_f.write('\nPer-class std dev acc result\n')
    result_f.write('Seed_Network,Loss,,Acc,,Epoch,\n')
    result_f.write(',StdDev,Label,StdDev,Label,StdDev,Label\n')
    print_values(7, NO_TABLES, model_names, random_seed_list, stopping_type_list, result_f, result)

    result_f.write('\nConv time std dev result\n')
    result_f.write('Seed_Network,Loss,,Acc,,Epoch,\n')
    result_f.write(',StdDev,Mean,StdDev,Mean,StdDev,Mean\n')
    print_values(8, NO_TABLES, model_names, random_seed_list, stopping_type_list, result_f, result)

    result_f.write('\nConv epoch std dev result\n')
    result_f.write('Seed_Network,Loss,,Acc,,Epoch,\n')
    result_f.write(',StdDev,Mean,StdDev,Mean,StdDev,Mean\n')
    print_values(9, NO_TABLES, model_names, random_seed_list, stopping_type_list, result_f, result)

    result_f.write('\nPer epoch std dev result\n')
    result_f.write('Seed_Network,Loss,,Acc,,Epoch,\n')
    result_f.write(',StdDev,Mean,StdDev,Mean,StdDev,Mean\n')
    print_values(10, NO_TABLES, model_names, random_seed_list, stopping_type_list, result_f, result)

    result_f.write('\nFirst epoch std dev result\n')
    result_f.write('Seed_Network,Loss,,Acc,,Epoch,\n')
    result_f.write(',StdDev,Mean,StdDev,Mean,StdDev,Mean\n')
    print_values(11, NO_TABLES, model_names, random_seed_list, stopping_type_list, result_f, result)

    result_f.close()


def main():
    result_dir = './result_per_epoch_re_run'

    random_seed_list = [-1, 1]

    stopping_type_list = ['best_loss', 'best_acc', 'epoch']

    model_names = ['LeNet1', 'LeNet4', 'LeNet5',
                   'ResNet38v1', 'ResNet56v1',
                   'WRN-28-10']

    data = pd.read_csv(result_dir + '/analysis_result.csv', skipinitialspace=True)
    data['backend_version'] = data['backend_version'].astype(str)
    data['cuda_version'] = data['cuda_version'].astype(str)
    data['cudnn_version'] = data['cudnn_version'].astype(str)

    # RQ1-RQ2: Accuracy and running time gap with only tensorflow
    tensorflow_data = data[data['backend'] == 'tensorflow']
    file_name = 'detail_analysis_result.csv'
    extract_result(file_name, model_names, random_seed_list, result_dir, stopping_type_list, tensorflow_data)

    # RQ3: Compare TensorFlow, CNTK, and Theano on fixed libraries
    tensorflow_compare_data = data[(data['backend'] == 'tensorflow') &
                                   (data['backend_version'] == '1.14.0') &
                                   (data['cuda_version'] == '10.0') &
                                   (data['cudnn_version'] == '7.6')]
    file_name = 'detail_analysis_result_tensorflow.csv'
    extract_result(file_name, model_names, random_seed_list, result_dir, stopping_type_list, tensorflow_compare_data)

    cntk_compare_data = data[(data['backend'] == 'cntk') &
                             (data['backend_version'] == '2.7') &
                             (data['cuda_version'] == '10.0') &
                             (data['cudnn_version'] == '7.6')]
    file_name = 'detail_analysis_result_cntk.csv'
    extract_result(file_name, model_names, random_seed_list, result_dir, stopping_type_list, cntk_compare_data)

    theano_compare_data = data[(data['backend'] == 'theano') &
                               (data['backend_version'] == '1.0.4') &
                               (data['cuda_version'] == '10.0') &
                               (data['cudnn_version'] == '7.6')]
    file_name = 'detail_analysis_result_theano.csv'
    extract_result(file_name, model_names, random_seed_list, result_dir, stopping_type_list, theano_compare_data)


if __name__ == "__main__":
    main()
