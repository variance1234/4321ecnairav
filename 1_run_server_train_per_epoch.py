import argparse
import subprocess
from multiprocessing import Lock, Queue, Process
from datetime import datetime
import os
import traceback

import running_utils


def execute_cpu_runs(runs):
    pass


def execute_multi_gpu_runs(runs):
    pass


def execute_one_single_gpu_run(time_stamp, temp_done_filename, result_dir, gpu_index, run_queue):
    try:
        temp_done_filename = temp_done_filename % gpu_index

        while True:
            if run_queue.empty():
                return
            run = run_queue.get()
            docker_command = run[0] % (gpu_index, gpu_index, gpu_index)
            container_name = run[1] % (gpu_index)
            docker_image = run[2]
            env = run[3]
            com = run[4]
            log_filename = run[5]

            log_filename = 'log/' + time_stamp + '_' + log_filename
            running_utils.create_directory(result_dir + '/' + log_filename)

            run_command = 'bash /working/1_0_run_single_server_train_per_epoch.sh ' + env + ' "' + com + ' ' + temp_done_filename + '" ' + '/result/' + log_filename

            full_command = docker_command + ' ' + docker_image + ' ' + 'su -l -c \'' + run_command + '\' user2'

            print('RUNNING: ' + full_command)

            subprocess.call(full_command, shell=True)

            subprocess.call('docker container rm ' + container_name, shell=True)
    except Exception:
        print(traceback.format_exc())


def execute_gpu_runs(time_stamp, done_f, result_dir, available_gpus, runs):
    run_queue = Queue()
    for run in runs:
        run_queue.put(run)

    temp_done_filename = 'temp_done_%d.csv'

    processes = []
    for gpu_index in available_gpus:
        process = Process(target=execute_one_single_gpu_run,
                          args=(time_stamp, temp_done_filename, result_dir, gpu_index, run_queue))
        process.start()
        processes.append(process)

    for process in processes:
        process.join()

    for gpu_index in available_gpus:
        td_filename = temp_done_filename % gpu_index
        temp_done_path = result_dir + '/' + td_filename
        with open(temp_done_path) as sd_f:
            lines = sd_f.readlines()

        for line in lines:
            done_f.write(line)
        done_f.flush()

        os.remove(temp_done_path)


def main():
    # read the parameter argument parsing
    parser = argparse.ArgumentParser(
        description='Run multiple training run on each of available gpus')
    parser.add_argument('data_dir', help="the path of the data folder")
    parser.add_argument('result_dir', help="the path of the result folder")
    parser.add_argument('done_filename', help="the path of the result folder")
    args = parser.parse_args()

    data_dir = args.data_dir
    result_dir = args.result_dir
    done_filename = args.done_filename
    # no_try = 16

    cuda_vers = ['9.0', '10.0']
    cudnn_vers = ['7.3', '7.4', '7.5', '7.6']

    '''
    lower_lib_config_list = [{'cuda_version': '9.0', 'cudnn_version': '7.4'},
                             {'cuda_version': '9.0', 'cudnn_version': '7.3'},
                             {'cuda_version': '9.0', 'cudnn_version': '7.5'},
                            {'cuda_version': '9.0', 'cudnn_version': '7.6'}]
    '''

    lib_config_list = [{'keras_version': '2.2.2', 'backend': 'tensorflow', 'backend_version': '1.10.0',
                        'lower_libs': [{'cuda_version': '9.0', 'cudnn_version': '7.3'},
                                       {'cuda_version': '9.0', 'cudnn_version': '7.4'},
                                       {'cuda_version': '9.0', 'cudnn_version': '7.5'},
                                       {'cuda_version': '9.0', 'cudnn_version': '7.6'}],
                        'no_try': 16,
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
                        },
                       {'keras_version': '2.2.2', 'backend': 'tensorflow', 'backend_version': '1.14.0',
                        'lower_libs': [{'cuda_version': '10.0', 'cudnn_version': '7.6'}],
                        'no_try': 16,
                        'random_seed_list': [0]
                        },
                       {'keras_version': '2.2.2', 'backend': 'cntk', 'backend_version': '2.7',
                        'lower_libs': [{'cuda_version': '10.0', 'cudnn_version': '7.6'}],
                        'no_try': 16,
                        'random_seed_list': [1, -1]
                        },
                       {'keras_version': '2.2.2', 'backend': 'theano', 'backend_version': '1.0.4',
                        'lower_libs': [{'cuda_version': '10.0', 'cudnn_version': '7.6'}],
                        'no_try': 16,
                        'random_seed_list': [1, -1]
                        }]

    # random_seed_list = [1, 0, -1]
    # random_seed_list = [1, -1]
    # random_seed_list = [1]
    # random_seed_list = [0]
    no_gpu_list = [1]
    # training_type_list = ['fine_tuning']
    training_type_list = ['from_scratch']

    '''
    model_names = ['LeNet1', 'LeNet4', 'LeNet5',
                   'ThaiMnist',
                   'ResNet56v1', 'ResNet38v1',
                   'TrafficSignsModel1', 'TrafficSignsModel2', 'TrafficSignsModel3',
                   'MobileNetV2', 'MobileNet', 'NASNetMobile', 'DenseNet121', 'DenseNet169', 'DenseNet201', 'Xception', 'InceptionV3', 'ResNet50', 'InceptionResNetV2', 'NASNetLarge', 'VGG16', 'VGG19']
    '''

    # model_names = ['InceptionResNetV2', 'MobileNetV2', 'DenseNet121', 'ResNet50', 'DenseNet169']

    model_names = ['LeNet1', 'LeNet4', 'LeNet5',
                   'ResNet56v1', 'ResNet38v1',
                   'WRN-28-10']

    # model_names = ['ResNet38v1']

    # model_names = ['WRN-28-10']

    # model_names = ['LeNet1', 'LeNet4', 'LeNet5']

    running_utils.create_directory(result_dir + '/' + done_filename)

    working_dir = os.getcwd()

    run_set, done_set, done_out_f = running_utils.setup_done_file(result_dir, done_filename,
                                                                  ',end_epoch,end_val_loss,end_val_accu,best_epoch,best_val_loss,best_val_accu,best_test_accu')
    done_out_f.flush()

    cpu_runs = []
    gpu_runs = []
    multi_gpu_runs = []

    for no_gpu in no_gpu_list:
        if no_gpu <= 0:
            computation = 'cpu'
            runs = cpu_runs
        elif no_gpu == 1:
            computation = '1_gpu'
            runs = gpu_runs
        else:
            computation = str(no_gpu) + '_gpu'
            runs = multi_gpu_runs

        for lib_config in lib_config_list:
            lower_libs = lib_config['lower_libs']
            random_seed_list = lib_config['random_seed_list']
            for lower_lib_config in lower_libs:
                for random_seed in random_seed_list:
                    for model_name in model_names:
                        for training_type in training_type_list:
                            no_try = lib_config['no_try']
                            for iTry in range(no_try):
                                # Create full config set
                                config = (computation, lib_config['keras_version'], lib_config['backend'],
                                          lib_config['backend_version'],
                                          lower_lib_config['cuda_version'], lower_lib_config['cudnn_version'],
                                          model_name, random_seed, training_type, iTry)

                                # Check if this config has been done before
                                if config in done_set:
                                    continue

                                # Main command
                                command = 'KERAS_BACKEND=%s python 1_train_models_per_epoch_optimizer.py %s %s %s %s %s %s %d %s %d %s %s %d' \
                                          % (lib_config['backend'], lib_config['keras_version'], lib_config['backend'],
                                             lib_config['backend_version'],
                                             lower_lib_config['cuda_version'], lower_lib_config['cudnn_version'],
                                             model_name, no_gpu, training_type, random_seed, '/data', '/result', iTry)
                                # Add special flag for theano to make it use GPU
                                if lib_config['backend'] == 'theano':
                                    command = "THEANO_FLAGS=device=cuda0 " + command

                                if random_seed == 0 and lib_config['backend'] == 'tensorflow':
                                    command = "TF_CUDNN_DETERMINISTIC=1 " + command

                                if random_seed >= 0:
                                    command = "PYTHONHASHSEED=" + str(random_seed) + ' ' + command

                                # Log file
                                log_filename = '1_train_models_per_epoch_%s_%s_%s_%s_%s_%s_%d_%s_%d_%d.log' \
                                               % (lib_config['keras_version'], lib_config['backend'],
                                                  lib_config['backend_version'],
                                                  lower_lib_config['cuda_version'], lower_lib_config['cudnn_version'],
                                                  model_name, no_gpu, training_type, random_seed, iTry)

                                # Conda environment
                                env = 'K_' + lib_config['keras_version'] + '_' + lib_config['backend'] + '_' + \
                                      lib_config['backend_version']

                                # Docker environment
                                docker_command = 'docker run ' + \
                                                 '-w /working ' + \
                                                 '-v /usr/local/anaconda3:/usr/local/anaconda3 ' + \
                                                 '-v /home/user2/.conda:/home/user2/.conda ' + \
                                                 '-v /local/user2/env:/env ' + \
                                                 '-v ' + result_dir + '/cache/keras%d:/home/user2/.keras ' + \
                                                 '-v  ' + working_dir + ':/working ' + \
                                                 '-v ' + data_dir + ':/data ' + \
                                                 '-v ' + result_dir + ':/result ' + \
                                                 '--name CUDA_%s_CUDNN_%s' % (
                                                     lower_lib_config['cuda_version'],
                                                     lower_lib_config['cudnn_version']) + \
                                                 '_GPU_%d --gpus '"device=%d"''

                                # Docker container and images
                                container_name = 'CUDA_%s_CUDNN_%s' % (
                                    lower_lib_config['cuda_version'], lower_lib_config['cudnn_version']) + '_GPU_%d'
                                docker_image = 'cuda:cuda_%s_cudnn_%s' % (
                                    lower_lib_config['cuda_version'], lower_lib_config['cudnn_version'])

                                # Add the run to the list
                                runs.append((docker_command, container_name, docker_image, env, command, log_filename))

    now = datetime.now()
    timestamp = now.strftime("%Y_%m_%d_%H_%M_%S")

    available_gpus = [0, 1, 2, 3, 4, 5, 6, 7]

    execute_gpu_runs(timestamp, done_out_f, result_dir, available_gpus, gpu_runs)
    # execute_multi_gpu_runs(multi_gpu_runs)
    # execute_cpu_runs(cpu_runs)

    done_out_f.close()


if __name__ == "__main__":
    main()
