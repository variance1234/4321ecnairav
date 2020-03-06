from pathlib import Path
import pandas as pd
import ntpath
import os


def create_directory(filepath):
    dir = ntpath.dirname(filepath)
    os.makedirs(dir, exist_ok=True)


def get_done_set(result_dir, filename):
    done_path = result_dir + '/' + filename
    done_data = pd.read_csv(done_path, skipinitialspace=True)

    # Run list
    comps = done_data['comp'].values
    kerass = done_data['keras'].values
    backends = done_data['backend'].values
    backend_versions = done_data['backend_version'].values
    cuda_versions = done_data['cuda_version'].values
    cudnn_versions = done_data['cudnn_version'].values
    networks = done_data['network'].values
    random_seeds = done_data['random_seed'].values
    training_types = done_data['training_type'].values
    runs = done_data['run'].values

    run_set = [(comps[i], kerass[i], backends[i], backend_versions[i], str(cuda_versions[i]), str(cudnn_versions[i]), networks[i], random_seeds[i], training_types[i], runs[i]) for i in range(len(comps))]

    print('Run ' + str(len(comps)) + ' projects!')

    # Finish list
    finished_data = done_data.loc[done_data['done']]

    comps = finished_data['comp'].values
    kerass = finished_data['keras'].values
    backends = finished_data['backend'].values
    backend_versions = finished_data['backend_version'].values
    cuda_versions = finished_data['cuda_version'].values
    cudnn_versions = finished_data['cudnn_version'].values
    networks = finished_data['network'].values
    random_seeds = finished_data['random_seed'].values
    training_types = finished_data['training_type'].values

    runs = finished_data['run'].values

    done_set = [(comps[i], kerass[i], backends[i], backend_versions[i], str(cuda_versions[i]), str(cudnn_versions[i]), networks[i], random_seeds[i], training_types[i], runs[i]) for i in range(len(comps))]

    print('Done ' + str(len(comps)) + ' projects!')

    return run_set, done_set


def setup_done_file(result_dir, filename, extraheader=''):
    done_path = result_dir + '/' + filename
    done_file = Path(done_path)
    if not done_file.is_file():
        done_set = []
        run_set = []
        create_directory(done_path)
        done_out_f = open(done_path, "w")
        done_out_f.write('comp,keras,backend,backend_version,cuda_version,cudnn_version,network,random_seed,training_type,run,done,time' + extraheader + '\n')
    else:
        run_set, done_set = get_done_set(result_dir, filename)

        done_out_f = open(done_path, "a")

    return run_set, done_set, done_out_f


def get_done_get_set(result_dir, filename):
    done_path = result_dir + '/' + filename
    done_data = pd.read_csv(done_path, skipinitialspace=True)

    # Run list
    comps = done_data['comp'].values
    kerass = done_data['keras'].values
    train_backends = done_data['train_backend'].values
    train_backend_versions = done_data['train_backend_version'].values
    backends = done_data['backend'].values
    backend_versions = done_data['backend_version'].values
    networks = done_data['network'].values
    runs = done_data['run'].values

    run_set = [(comps[i], kerass[i], train_backends[i], train_backend_versions[i], backends[i], backend_versions[i], networks[i], runs[i]) for i in range(len(comps))]

    print('Run ' + str(len(comps)) + ' projects!')

    # Finish list
    finished_data = done_data.loc[done_data['done']]

    comps = finished_data['comp'].values
    kerass = finished_data['keras'].values
    train_backends = finished_data['train_backend'].values
    train_backend_versions = finished_data['train_backend_version'].values
    backends = finished_data['backend'].values
    backend_versions = finished_data['backend_version'].values
    networks = finished_data['network'].values
    runs = finished_data['run'].values

    done_set = [(comps[i], kerass[i], train_backends[i], train_backend_versions[i], backends[i], backend_versions[i], networks[i], runs[i]) for i in range(len(comps))]

    print('Done ' + str(len(comps)) + ' projects!')

    return run_set, done_set


def setup_done_get_file(result_dir, filename):
    done_path = result_dir + '/' + filename
    done_file = Path(done_path)
    if not done_file.is_file():
        done_set = []
        run_set = []
        create_directory(done_path)
        done_out_f = open(done_path, "w")
        done_out_f.write('comp,keras,train_backend,train_backend_version,backend,backend_version,network,run,done,time\n')
    else:
        run_set, done_set = get_done_get_set(result_dir, filename)

        done_out_f = open(done_path, "a")

    return run_set, done_set, done_out_f


def get_inconsistent_set(result_dir):
    done_path = result_dir + '/network_inconsistency.csv'

    done_file = Path(done_path)
    if not done_file.is_file():
        return [], []

    done_data = pd.read_csv(done_path, skipinitialspace=True)

    # analyzed list
    comps = done_data['comp'].values
    kerass = done_data['keras'].values
    back1s = done_data['back1'].values
    back2s = done_data['back2'].values
    back1_versions = done_data['backend_version1'].values
    back2_versions = done_data['backend_version2'].values
    networks = done_data['network'].values
    metrics = done_data['metric'].values

    run_set = [
        (comps[i], kerass[i], back1s[i], back1_versions[i], back2s[i], back2_versions[i], networks[i], metrics[i]) for i
        in range(len(comps))]

    print('Analyzed ' + str(len(comps)) + ' projects!')

    # inconsistent list
    finished_data = done_data.loc[done_data['inconsistent']]

    comps = finished_data['comp'].values
    kerass = finished_data['keras'].values
    back1s = finished_data['back1'].values
    back2s = finished_data['back2'].values
    back1_versions = finished_data['backend_version1'].values
    back2_versions = finished_data['backend_version2'].values
    networks = finished_data['network'].values
    metrics = finished_data['metric'].values

    incon_set = [
        (comps[i], kerass[i], back1s[i], back1_versions[i], back2s[i], back2_versions[i], networks[i], metrics[i]) for i
        in range(len(comps))]

    print('Found ' + str(len(comps)) + ' inconsistencies!')

    return run_set, incon_set


def get_accuracy_set(result_dir):
    done_path = result_dir + '/network_accuracy.csv'

    done_file = Path(done_path)
    if not done_file.is_file():
        return [], []

    done_data = pd.read_csv(done_path, skipinitialspace=True)

    # analyzed list
    comps = done_data['comp'].values
    kerass = done_data['keras'].values
    backs = done_data['backend'].values
    backend_versions = done_data['backend_version'].values
    networks = done_data['network'].values

    run_set = [(comps[i], kerass[i], backs[i], backend_versions[i], networks[i]) for i in range(len(comps))]

    print('Done accuracy in ' + str(len(comps)) + ' projects!')

    return run_set


def get_accuracy_train_set(result_dir):
    done_path = result_dir + '/train_network_accuracy.csv'

    done_file = Path(done_path)
    if not done_file.is_file():
        return [], []

    done_data = pd.read_csv(done_path, skipinitialspace=True)

    # analyzed list
    comps = done_data['comp'].values
    kerass = done_data['keras'].values
    train_backs = done_data['train_backend'].values
    train_backend_versions = done_data['train_backend_version'].values
    test_backs = done_data['test_backend'].values
    test_backend_versions = done_data['test_backend_version'].values
    networks = done_data['network'].values
    runs = done_data['run'].values

    run_set = [(comps[i], kerass[i], train_backs[i], train_backend_versions[i], test_backs[i], test_backend_versions[i], networks[i], runs[i]) for i in range(len(comps))]

    print('Done accuracy in ' + str(len(comps)) + ' projects!')

    return run_set
