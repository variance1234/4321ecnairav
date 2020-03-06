import numpy as np
import pandas as pd
import h5py


def filter_data(data, acc, back, network):
    filtered_data = data.loc[(data['comp'] == acc) & (data['network'] == network)]
    back_data = filtered_data.loc[filtered_data['backend'] == back]
    back_data = back_data.sort_values(by=['data_name'])
    return back_data


def load_run_result(result_dir, keras_version, network, back, version):
    data = pd.read_csv(result_dir + '/result_' + keras_version + '_' + back + '_' + version + '_' + network + '.csv', skipinitialspace=True)
    return data


def load_train_run_result(result_dir, keras_version, network, train_back, train_version, test_back, test_version):
    data = pd.read_csv(result_dir + '/result_' + keras_version + '_' + train_back + '_' + train_version + '_' + test_back + '_' + test_version + '_' + network + '.csv', skipinitialspace=True)
    return data


def load_run_output(result_dir, keras_version, network, back, version, data_names):
    hf = h5py.File(result_dir + '/output_' + keras_version + '_' + back + '_' + version + '_' + network + '.h5', 'r')
    out = [np.array(hf[str(data_name)]) for data_name in data_names]
    out[0] = np.reshape(out[0], out[1].shape)
    out = np.stack(out, axis=0)
    hf.close()
    return out

def load_train_run_output(result_dir, keras_version, network, train_back, train_version, test_back, test_version, data_names):
    hf = h5py.File(result_dir + '/output_' + keras_version + '_' + train_back + '_' + train_version + '_' + test_back + '_' + test_version + '_' + network + '.h5', 'r')
    out = [np.array(hf[str(data_name)]) for data_name in data_names]
    out[0] = np.reshape(out[0], out[1].shape)
    out = np.stack(out, axis=0)
    hf.close()
    return out


def load_run_output_without_version(result_dir, keras_version, network, back, data_names):
    hf = h5py.File(result_dir + '/output_' + keras_version + '_' + back + '_' + network + '.h5', 'r')
    out = [np.array(hf[str(data_name)]) for data_name in data_names]
    out[0] = np.reshape(out[0], out[1].shape)
    out = np.stack(out, axis=0)
    hf.close()
    return out


def match_label(data_names, truth):
    actual_labels = list()
    for name in data_names:
        actual_labels.append(truth[name])
    actual_labels = np.array(actual_labels).reshape((-1, 1))

    return actual_labels


# TODO: seem to be buggy
def regression_based_metric(actual_labels, back1_data, back2_data):
    predicted_labels_1 = back1_data.iloc[:, 4].values
    predicted_labels_2 = back2_data.iloc[:, 4].values

    diff1 = predicted_labels_1 - actual_labels
    diff2 = predicted_labels_2 - actual_labels

    diffs = np.abs(diff1 - diff2)

    counts, bins = np.histogram(diffs, bins=6)

    label_range = np.max(actual_labels) - np.min(actual_labels)

    got_inconsistency = np.max(diffs) > label_range * 0.1

    return diffs, got_inconsistency, counts, bins


def mad_based_metric(actual_labels, out1, out2):
    diff1 = [mad(out1[i], actual_labels[i]) for i in range(actual_labels.shape[0])]
    diff2 = [mad(out2[i], actual_labels[i]) for i in range(actual_labels.shape[0])]

    diff1 = np.stack(diff1, axis=0)
    diff2 = np.stack(diff2, axis=0)

    diffs = np.divide(np.abs(diff1 - diff2), diff1 + diff2)
    diffs = np.nan_to_num(diffs)

    got_inconsistency = np.max(diffs) >= 0.25

    return diffs, got_inconsistency


def mad(X, Y):
    return np.sum(np.abs(X - Y)) / len(X)


def class_based_metric(actual_labels, out1, out2):
    predicted_labels_1 = decode_class(out1)
    predicted_labels_2 = decode_class(out2)

    match1 = np.equal(predicted_labels_1, actual_labels)
    match2 = np.equal(predicted_labels_2, actual_labels)

    match1 = match1.astype('int')
    match2 = match2.astype('int')

    match1 = np.multiply(match1, np.array([16, 8, 4, 2, 1]))
    match2 = np.multiply(match2, np.array([16, 8, 4, 2, 1]))

    match1 = match1.sum(axis=1)
    match2 = match2.sum(axis=1)

    diffs = np.abs(match1 - match2)

    got_inconsistency = np.max(diffs) >= 1

    return diffs, got_inconsistency


# get top predicted class
def decode_class(preds, top=5):
    results = []

    for pred in preds:
        if len(pred) > 2:
            result = pred.argsort()[-top:][::-1]
        else:
            p = float(pred[0])
            if p >= 0.5:
                result = [0, 1, 2, 3, 4]
            else:
                result = [1, 0, 2, 3, 4]

        results.append(result)

    results = np.stack(results, axis=0)
    return results
