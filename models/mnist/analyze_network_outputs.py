import pandas as pd
import matplotlib.pyplot as plt
from utils import *
from keras.datasets import mnist


def get_image(imagename):
    index = int(imagename[6:])
    return x_data[index]


def summary_diff_backend(data, back1, back2, acc, network):
    filtered_data = data.loc[(data['acc'] == acc) & (data['network'] == network)]

    back1_data = filtered_data.loc[filtered_data['backend'] == back1]
    back1_data = back1_data.sort_values(by=['image'])
    back2_data = filtered_data.loc[filtered_data['backend'] == back2]
    back2_data = back2_data.sort_values(by=['image'])

    diffs = back1_data.iloc[:, list(range(3, 13, 2))].values == back2_data.iloc[:, list(range(3, 13, 2))].values
    diffs = 1 - diffs.astype('int')
    diffs = np.multiply(diffs, np.array([16, 8, 4, 2, 1]))
    diffs = diffs.sum(axis=1)

    plt.hist(diffs, 100)
    plt.savefig(dir + '/' + back1 + '_vs_' + back2 + '_' + acc + '_' + network + '.png')
    plt.clf()
    plt.close()

    return diffs


def summary_diff_acc(data, back, network):
    acc1 = 'cpu'
    acc2 = 'gpu'

    filtered_data = data.loc[(data['backend'] == back) & (data['network'] == network)]

    acc1_data = filtered_data.loc[filtered_data['acc'] == acc1]
    acc1_data = acc1_data.sort_values(by=['image'])
    acc2_data = filtered_data.loc[filtered_data['acc'] == acc2]
    acc2_data = acc2_data.sort_values(by=['image'])

    diffs = acc1_data.iloc[:, list(range(3, 13, 2))].values == acc2_data.iloc[:, list(range(3, 13, 2))].values
    diffs = 1 - diffs.astype('int')
    diffs = np.multiply(diffs, np.array([16, 8, 4, 2, 1]))
    diffs = diffs.sum(axis=1)

    plt.hist(diffs, 100)
    plt.savefig(dir + '/' + acc1 + '_vs_' + acc2 + '_' + back + '_' + network + '.png')
    plt.clf()
    plt.close()

    return diffs


def display_top_k_backend(data, k, back1, back2, acc, network):
    filtered_data = data.loc[(data['acc'] == acc) & (data['network'] == network)]

    back1_data = filtered_data.loc[filtered_data['backend'] == back1]
    back1_data = back1_data.sort_values(by=['image'])
    back2_data = filtered_data.loc[filtered_data['backend'] == back2]
    back2_data = back2_data.sort_values(by=['image'])

    diffs = back1_data.iloc[:, list(range(3, 13, 2))].values == back2_data.iloc[:, list(range(3, 13, 2))].values
    diffs = 1 - diffs.astype('int')
    diffs = np.multiply(diffs, np.array([16, 8, 4, 2, 1]))
    diffs = diffs.sum(axis=1)

    sorted_i = np.argsort(-diffs)
    sorted_diff = diffs[sorted_i]
    sorted_image = back1_data['image'].values
    sorted_image = sorted_image[sorted_i]
    sorted_percent1 = back1_data.iloc[:, list(range(4, 13, 2))].values
    sorted_percent1 = sorted_percent1[sorted_i]
    sorted_label1 = back1_data.iloc[:, list(range(3, 13, 2))].values
    sorted_label1 = sorted_label1[sorted_i]
    sorted_percent2 = back2_data.iloc[:, list(range(4, 13, 2))].values
    sorted_percent2 = sorted_percent2[sorted_i]
    sorted_label2 = back2_data.iloc[:, list(range(3, 13, 2))].values
    sorted_label2 = sorted_label2[sorted_i]

    ind = np.arange(1, 6)

    fig, axeslist = plt.subplots(ncols=k, nrows=3, figsize=(50, 12), sharey='row')
    for i in range(k):
        axeslist.ravel()[i].bar(ind, sorted_percent1[i, :])
        axeslist.ravel()[i].set_xticks(ind)
        axeslist.ravel()[i].set_xticklabels(sorted_label1[i, :])
        axeslist.ravel()[i].set_title(back1)
        plt.setp(axeslist.ravel()[i].get_xticklabels(), rotation=90)

    for i in range(k):
        axeslist.ravel()[k + i].bar(ind, sorted_percent2[i, :])
        axeslist.ravel()[k + i].set_xticks(ind)
        axeslist.ravel()[k + i].set_xticklabels(sorted_label2[i, :])
        axeslist.ravel()[k + i].set_title(back2)
        plt.setp(axeslist.ravel()[k + i].get_xticklabels(), rotation=90)

    for i in range(k):
        image_data = get_image(sorted_image[i])
        image_data = deprocess_image(image_data)
        axeslist.ravel()[2 * k + i].imshow(image_data, interpolation='nearest')
        # axeslist.ravel()[2*k+i].set_title(sorted_image[i])
        axeslist.ravel()[2 * k + i].set_axis_off()

    plt.tight_layout()
    plt.savefig(dir + '/top_' + str(k) + '_' + back1 + '_vs_' + back2 + '_' + acc + '_' + network + '.png')
    plt.clf()
    plt.close()


def main():
    global dir, x_data
    dir = '../result/mnist'

    # input image dimensions
    img_rows, img_cols = 28, 28

    (x_train, _), (x_test, _) = mnist.load_data()

    x_data = np.concatenate((x_train, x_test), axis=0)

    x_data = x_data.reshape(x_data.shape[0], img_rows, img_cols, 1)

    x_data = x_data.astype('float32')
    x_data /= 255

    cpu = pd.read_csv(dir + '/cpu_mnist.csv', skipinitialspace=True)
    cpu['acc'] = 'cpu'
    gpu = pd.read_csv(dir + '/gpu_mnist.csv', skipinitialspace=True)
    gpu['acc'] = 'gpu'

    data = pd.concat([cpu, gpu])

    networks = ['LeNet-1', 'LeNet-4', 'LeNet-5']
    cgpu = ['cpu', 'gpu']
    backends = ['theano', 'tensorflow', 'cntk']
    backendpairs = [('theano', 'tensorflow'), ('tensorflow', 'cntk'), ('cntk', 'theano')]

    for network in networks:
        for backend in backends:
            summary_diff_acc(data, backend, network)

        for cg in cgpu:
            for backendpair in backendpairs:
                summary_diff_backend(data, backendpair[0], backendpair[1], cg, network)
                display_top_k_backend(data, 10, backendpair[0], backendpair[1], cg, network)


if __name__ == "__main__":
    main()
