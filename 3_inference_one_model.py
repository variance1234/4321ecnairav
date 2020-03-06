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
from pathlib import Path

import keras
from models import models_list
import running_utils


def create_directory(filepath):
    dir = ntpath.dirname(filepath)
    os.makedirs(dir, exist_ok=True)


def main():
    # read the parameter
    # argument parsing
    parser = argparse.ArgumentParser(
        description='Generate model output')
    parser.add_argument('keras_version', help="the keras version used")
    parser.add_argument('backend', help="the back end name used in this run",
                        choices=['theano', 'tensorflow', 'cntk'])
    parser.add_argument('backend_version', help="the back end name used in this run")
    parser.add_argument('cuda_version', help="cuda version")
    parser.add_argument('cudnn_version', help="cudnn version")
    parser.add_argument('model_name', help="the model name")
    parser.add_argument('no_gpu', help="the number of gpus")
    parser.add_argument('inference_type', help="the inference type")
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
    inference_type = args.inference_type

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
        create_directory(e_path)
        e_out_f = open(e_path, "w")
        for erros in models_list.import_erros:
            e_out_f.write(erros + '\n\n')
        e_out_f.close()

    dataset = models_list.get_property(model_name, models_list.DATA_DIR)

    try:
        print('Running %s,%s,%s,%s,%s,%s,%d,%s,%d\n' % (
            args.keras_version, args.backend, args.backend_version, args.cuda_version, args.cudnn_version, model_name,
            random_seed, inference_type, iTry))

        exception_filename = 'inference_exception_%s_%s_%s_%s_%s_%s_%d_%s.txt' % (
            args.keras_version, args.backend, args.backend_version, args.cuda_version, args.cudnn_version,
            model_name, random_seed, inference_type)

        # Delete previous run (need to rerun)
        os.system("find '" + args.result_dir + "' -name " + exception_filename + " -type f -delete")

        begin_run = time.time()

        output_filename = 'inference_output_%s_%s_%s_%s_%s_%s_%d_%s_%d.h5' % (
            args.keras_version, args.backend, args.backend_version, args.cuda_version, args.cudnn_version,
            model_name, random_seed, inference_type, iTry)
        perfomance_filename = 'inference_performance_%s_%s_%s_%s_%s_%s_%d_%s_%d.csv' % (
            args.keras_version, args.backend, args.backend_version, args.cuda_version, args.cudnn_version,
            model_name, random_seed, inference_type, iTry)

        # open h5 file to record output
        o_path = args.result_dir + '/' + dataset + '/' + output_filename
        o_out_hf = h5py.File(o_path, 'w')

        # open csv to record prediction performance
        p_path = args.result_dir + '/' + dataset + '/' + perfomance_filename
        p_out_f = open(p_path, "w")
        p_out_f.write('batch_name,size,time\n')

        # Load Data List
        data_dir_path = args.data_dir + '/' + dataset
        data_names, data_source, labels = \
            models_list.get_property(model_name, models_list.LOAD_DATA_LIST)(data_dir_path, type=models_list.TEST)

        # Load the model
        model = models_list.get_property(model_name, models_list.INIT_METHOD)()
        print("Done load model " + model_name)

        batch_size = models_list.get_property(model_name, models_list.TRAINING_BATCH)

        if isinstance(batch_size, dict):
            batch_size = batch_size[args.backend]

        outputs = []

        batch_i = 0
        # while batch_i < 10:
        while batch_i < len(data_names) / batch_size:
            # Load the batch
            batch_data_names = data_names[batch_i * batch_size: (batch_i + 1) * batch_size]
            batch_input_datas = list()
            for data_name in batch_data_names:
                single_data = models_list.get_property(model_name, models_list.LOAD_DATA)(data_source, data_name)
                batch_input_datas.append(single_data)
                del single_data

            data = np.concatenate(batch_input_datas, axis=0)

            # Get outputs and time the process
            begin = time.time()
            output = model.predict(data, batch_size=len(batch_data_names))
            end = time.time()

            del data

            # store performance
            p_out_f.write('%s, %d, %.5f\n' %
                          (batch_i, len(batch_data_names), (end - begin)))

            output_type = models_list.get_property(model_name, models_list.OUTPUT_TYPE)

            # add output

            outputs.append(output)

            print(
                "Done a prediction model: " + model_name +
                " batch: " + str(batch_i) + " in " +
                ("%.4f" % (end - begin)) + " s")

            batch_i += 1

        outputs = np.concatenate(outputs, axis=0)

        o_out_hf.create_dataset('Outputs', data=outputs)

        del model

        o_out_hf.close()
        p_out_f.close()

        end_run = time.time()

        done_out_f.write('%s,%s,%s,%s,%s,%s,%s,%d,%s,%d' % (
            computation, args.keras_version, args.backend, args.backend_version, args.cuda_version,
            args.cudnn_version,
            model_name, random_seed, inference_type, iTry))
        done_out_f.write(',True,%.5f\n' % (end_run - begin_run))
    except Exception:
        except_filename = 'inference_exception_%s_%s_%s_%s_%s_%s_%d_%s.txt' % (
            args.keras_version, args.backend, args.backend_version, args.cuda_version, args.cudnn_version, model_name,
            random_seed, inference_type)
        e_path = args.result_dir + '/' + dataset + '/' + except_filename
        create_directory(e_path)
        e_out_f = open(e_path, "w")
        e_out_f.write(traceback.format_exc())
        e_out_f.close()
        done_out_f.write('%s,%s,%s,%s,%s,%s,%s,%d,%s,%d' % (
            computation, args.keras_version, args.backend, args.backend_version, args.cuda_version, args.cudnn_version,
            model_name, random_seed, inference_type, iTry))
        done_out_f.write(',False\n')

    # Close done file
    done_out_f.close()


if __name__ == "__main__":
    from keras import backend as K

    if (K.backend() == 'tensorflow'):
        import tensorflow as tf

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        K.set_session(sess)
    main()
