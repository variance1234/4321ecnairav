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


def cal_runtime(per_data):
    batch_names = per_data['batch_name'].values
    times = per_data['time'].values
    batch_sizes = per_data['size'].values

    per_batch_average_running_time = 0
    first_batch_average_running_time = 0

    for i in range(len(batch_names)):
        batch_name = batch_names[i]

        if batch_name == 0:
            first_batch_average_running_time = (times[i] / batch_sizes[i])

        if batch_name > 0:
            per_batch_average_running_time += (times[i] / batch_sizes[i])

    per_batch_average_running_time /= (len(batch_names) - 1)

    return first_batch_average_running_time, per_batch_average_running_time

def find_max_accuracy_diff(acc_raw_f, result_dir_path, no_try, labels, comp, kv, back, back_v, cu_v, cudnn_v, model_name, random_seed, training_type):
    accuracies = []
    per_label_accuracies = []

    first_batch_running_times = []
    per_batch_running_times = []

    for iTry in range(no_try):
        # Calculate running time
        performance_filename = 'inference_performance_%s_%s_%s_%s_%s_%s_%d_%s_%d.csv' % (
            kv, back, back_v, cu_v, cudnn_v, model_name, random_seed, training_type, iTry)

        per_p = result_dir_path + '/' + performance_filename
        per_path = Path(per_p)
        if not per_path.is_file():
            break
        per_data = pd.read_csv(per_p, skipinitialspace=True)

        if len(per_data['batch_name'].values) == 0:
            return None


        # Calculate accuracy
        output_filename = 'inference_output_%s_%s_%s_%s_%s_%s_%d_%s_%d.h5' % (
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

        first_batch_average_running_time, per_batch_average_running_time = cal_runtime(per_data)
        first_batch_running_times.append(first_batch_average_running_time)
        per_batch_running_times.append(per_batch_average_running_time)

        acc_raw_f.write('%s,%s,%s,%s,%s,%s,%s,%d,%s,%d' %
                        (comp, kv, back, back_v, cu_v, cudnn_v, model_name, random_seed, training_type, iTry))
        acc_raw_f.write(',%.5f,%.5f,%.5f\n' %
                        (accuracy,
                         first_batch_average_running_time,
                         per_batch_average_running_time))

    if len(accuracies) == 0:
        return None

    # calculate stat for accuracy
    max_accuracy_diff = max(accuracies) - min(accuracies)
    max_accuracy = max(accuracies)
    min_accuracy = min(accuracies)
    std_dev_accuracy = st.stdev(accuracies)
    mean_accuracy = st.mean(accuracies)

    # calculate stat for running time
    max_per_batch_diff = max(per_batch_running_times) - min(per_batch_running_times)
    max_per_batch = max(per_batch_running_times)
    min_per_batch = min(per_batch_running_times)
    std_dev_per_batch = st.stdev(per_batch_running_times)
    mean_per_batch = st.mean(per_batch_running_times)

    max_first_batch_diff = max(first_batch_running_times) - min(first_batch_running_times)
    max_first_batch = max(first_batch_running_times)
    min_first_batch = min(first_batch_running_times)
    std_dev_first_batch = st.stdev(first_batch_running_times)
    mean_first_batch = st.mean(first_batch_running_times)

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

    return (len(accuracies),
            max_accuracy_diff, max_accuracy, min_accuracy, std_dev_accuracy, mean_accuracy,
            max_diff_label, max_per_label_acc_diff, max_label_accuracy, min_label_accuracy, per_label_no_samples[max_diff_label],
            max_std_label, max_per_label_acc_std, per_label_no_samples[max_std_label],
            max_per_batch_diff, max_per_batch, min_per_batch, std_dev_per_batch, mean_per_batch,
            max_first_batch_diff, max_first_batch, min_first_batch, std_dev_first_batch, mean_first_batch)

def main():
    # read the parameter
    # argument parsing
    result_dir = "result_inference"
    data_dir = "../crossmodelchecking/data"

    no_try = 16

    computation = '1_gpu'
    # backends = ['tensorflow']
    # backends = ['theano', 'tensorflow', 'cntk']

    lower_lib_config_list = [{'cuda_version': '10.0', 'cudnn_version': '7.6'}]
    '''
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
    
    '''

    lib_config_list = [{'keras_version': '2.2.2', 'backend': 'tensorflow', 'backend_version': '1.14.0'}]

    random_seed_list = [-1 ,0, 1]
    # random_seed_list = [0]
    no_gpu_list = [1]
    # training_type_list = ['fine_tuning']
    training_type_list = ['pre_train']

    stopping_type_list = ['best_loss', 'best_acc', 'epoch']
    # stopping_type_list = ['best_acc']


    model_names = ['LeNet1', 'LeNet4', 'LeNet5',
                   'ResNet38v1', 'ResNet56v1',
                   'WRN-28-10']


    #model_names = ['ResNet38v1']

    acc_raw_path = result_dir + '/analysis_raw.csv'
    acc_raw_f = open(acc_raw_path, "w")
    acc_raw_f.write('comp,keras,backend,backend_version,cuda_version,cudnn_version,network,random_seed,training_type,try,' +
                    'accuracy,' +
                    'first_batch_average,'
                    'per_batch_average,\n')

    acc_path = result_dir + '/analysis_result.csv'
    acc_f = open(acc_path, "w")
    acc_f.write('comp,keras,backend,backend_version,cuda_version,cudnn_version,network,random_seed,training_type,' +
                'no_try,' +
                'max_accuracy_diff,max_accuracy,min_accuracy,std_dev_accuracy,mean_accuracy,' +
                'max_diff_label,max_per_label_acc_diff,max_label_accuracy,min_label_accuracy,no_samples_max_diff,' +
                'max_std_label,max_per_label_acc_std,no_samples_max_std,' +
                'max_per_batch_diff,max_per_batch,min_per_batch,std_dev_per_batch,mean_per_batch,'
                'max_first_batch_diff,max_first_batch,min_first_batch,std_dev_first_batch,mean_first_batch\n')

    models_list.import_model()

    for random_seed in random_seed_list:
        for model_name in model_names:

            dataset = models_list.get_property(model_name, models_list.DATA_DIR)

            data_dir_path = data_dir + '/' + dataset
            result_dir_path = result_dir + '/' + dataset

            # Load Data List
            data_names, data_source, labels = models_list.get_property(model_name, models_list.LOAD_DATA_LIST)(
                data_dir_path, type=models_list.TEST)


            for lower_lib_config in lower_lib_config_list:
                for lib_config in lib_config_list:
                    for training_type in training_type_list:
                        print('Analyzing %s,%s,%s,%s,%s,%s,%s,%d,%s\n' % (
                            computation, lib_config['keras_version'], lib_config['backend'], lib_config['backend_version'],
                            lower_lib_config['cuda_version'], lower_lib_config['cudnn_version'],
                            model_name, random_seed, training_type))

                        result = \
                            find_max_accuracy_diff(acc_raw_f, result_dir_path, no_try, labels,
                                                   computation, lib_config['keras_version'], lib_config['backend'], lib_config['backend_version'],
                                                   lower_lib_config['cuda_version'], lower_lib_config['cudnn_version'],
                                                   model_name, random_seed, training_type)

                        if result is None:
                            continue

                        (actual_no_try,
                         max_accuracy_diff, max_accuracy, min_accuracy, std_dev_accuracy, mean_accuracy,
                         max_diff_label, max_per_label_acc_diff, max_label_accuracy, min_label_accuracy, no_samples_max_diff,
                         max_std_label, max_per_label_acc_std, no_samples_max_std,
                         max_per_epoch_diff, max_per_epoch, min_per_epoch, std_dev_per_epoch, mean_per_epoch,
                         max_first_epoch_diff, max_first_epoch, min_first_epoch, std_dev_first_epoch, mean_first_epoch) = result

                        acc_f.write('%s,%s,%s,%s,%s,%s,%s,%d,%s' % (
                            computation, lib_config['keras_version'], lib_config['backend'], lib_config['backend_version'],
                            lower_lib_config['cuda_version'], lower_lib_config['cudnn_version'],
                            model_name, random_seed, training_type))
                        acc_f.write(',%d,%.5f,%.5f,%.5f,%.5f,%.5f,%d,%.5f,%.5f,%.5f,%d,%d,%.5f,%d,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f\n' %
                                    (actual_no_try,
                                     max_accuracy_diff, max_accuracy, min_accuracy, std_dev_accuracy, mean_accuracy,
                                     max_diff_label, max_per_label_acc_diff, max_label_accuracy, min_label_accuracy, no_samples_max_diff,
                                     max_std_label, max_per_label_acc_std, no_samples_max_std,
                                     max_per_epoch_diff, max_per_epoch, min_per_epoch, std_dev_per_epoch, mean_per_epoch,
                                     max_first_epoch_diff, max_first_epoch, min_first_epoch, std_dev_first_epoch, mean_first_epoch))

    acc_raw_f.close()
    acc_f.close()


if __name__ == "__main__":
    main()
