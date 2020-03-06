import pandas as pd
import numpy as np

import matplotlib as mpl

mpl.use('agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import statistics as st


def draw_max_accuracy_diff(result_path, lib_config_list, data_file='analysis_result.csv'):
    data = pd.read_csv(result_path + '/' + data_file, skipinitialspace=True)
    data['backend_version'] = data['backend_version'].astype(str)
    data['cuda_version'] = data['cuda_version'].astype(str)
    data['cudnn_version'] = data['cudnn_version'].astype(str)

    for random_seed in [1, -1]:
        data_seed = data[(data['random_seed'] == random_seed)]
        list_lenet = print_max_accuracy_diff(data_seed, ['LeNet1', 'LeNet4', 'LeNet5'], lib_config_list)
        list_resnet = print_max_accuracy_diff(data_seed, ['ResNet38v1', 'ResNet56v1'], lib_config_list)
        list_wrn = print_max_accuracy_diff(data_seed, ['WRN-28-10'], lib_config_list)

        fig = plt.figure(figsize=(12, 3))
        ax1, ax2, ax3 = fig.subplots(1, 3, gridspec_kw={'width_ratios': [3, 2, 1]})

        def to_percent(temp, position):
            return '%1.1f' % (100 * temp) + '%'

        # Create the boxplot
        ax1.boxplot(list_lenet, widths=0.4, showmeans=True)
        ax1.set_xticks([1, 2, 3])
        ax1.set_xlim(0.5, 3.5)
        ax1.set_xticklabels(['LeNet1', 'LeNet4', 'LeNet5'])
        ax1.tick_params(axis='both', which='major', labelsize=15)
        ax1.yaxis.set_major_formatter(FuncFormatter(to_percent))

        # Create the boxplot
        ax2.boxplot(list_resnet, widths=0.4, showmeans=True)
        ax2.set_xticks([1, 2])
        ax2.set_xlim(0.5, 2.5)
        ax2.set_xticklabels(['ResNet38', 'ResNet56'])
        ax2.tick_params(axis='both', which='major', labelsize=15)
        ax2.yaxis.set_major_formatter(FuncFormatter(to_percent))

        # Create the boxplot
        ax3.boxplot(list_wrn, widths=0.4, showmeans=True)
        ax3.set_xticks([1])
        ax3.set_xlim(0.5, 1.5)
        ax3.set_xticklabels(['WRN-28-10'])
        ax3.tick_params(axis='both', which='major', labelsize=15)
        ax3.yaxis.set_major_formatter(FuncFormatter(to_percent))

        fig.subplots_adjust(wspace=0.6)

        # Save the figure
        fig.savefig(result_path + '/plots/max_accuracy_diff_%d.png' % (random_seed), dpi=600, bbox_inches='tight')
        plt.close(fig)


def print_max_accuracy_diff(data, model_names, lib_config_list):
    model_list = []
    for subject in model_names:
        network_data = data[(data['network'].values == subject)]
        list = get_max_accuracy_diff(network_data, lib_config_list)
        model_list.append(list)
    return model_list


def get_max_accuracy_diff(data, lib_config_list):
    max_acc_diff_list = []
    for lib_config in lib_config_list:
        lower_libs = lib_config['lower_libs']
        for lower_lib_config in lower_libs:
            data_batch = data[(data['backend_version'].values == lib_config['backend_version']) &
                              (data['cuda_version'].values == lower_lib_config['cuda_version']) &
                              (data['cudnn_version'].values == lower_lib_config['cudnn_version'])]
            if len(data_batch) <= 0:
                continue
            data_loss = data_batch[data_batch['stopping_type'] == 'best_loss']
            max_acc_diff_loss = data_loss['max_accuracy_diff'].values[0]

            data_acc = data_batch[data_batch['stopping_type'] == 'best_acc']
            max_acc_diff_acc = data_acc['max_accuracy_diff'].values[0]

            max_acc_diff_list.append(max(max_acc_diff_loss, max_acc_diff_acc))
    return np.array(max_acc_diff_list)


def draw_max_diff_mean_criterion(result_path, lib_config_list, data_file='analysis_result.csv'):
    data = pd.read_csv(result_path + '/' + data_file, skipinitialspace=True)
    data['backend_version'] = data['backend_version'].astype(str)
    data['cuda_version'] = data['cuda_version'].astype(str)
    data['cudnn_version'] = data['cudnn_version'].astype(str)

    for criterion in ['max_accuracy_diff', 'max_convergent_diff', 'max_convergent_diff_epoch']:
        for stopping_type in ['best_loss', 'best_acc']:
            for random_seed in [1, -1]:
                data_seed = data[(data['random_seed'] == random_seed)]
                list_lenet = print_max_diff_mean_criterion(data_seed, ['LeNet1', 'LeNet4', 'LeNet5'], lib_config_list, criterion, stopping_type)
                list_resnet = print_max_diff_mean_criterion(data_seed, ['ResNet38v1', 'ResNet56v1'], lib_config_list, criterion, stopping_type)
                list_wrn = print_max_diff_mean_criterion(data_seed, ['WRN-28-10'], lib_config_list, criterion, stopping_type)

                fig = plt.figure(figsize=(12, 3))
                ax1, ax2, ax3 = fig.subplots(1, 3, gridspec_kw={'width_ratios': [3, 2, 1]})

                # Create the boxplot
                ax1.boxplot(list_lenet, widths=0.4, showmeans=True)
                ax1.set_xticks([1, 2, 3])
                ax1.set_xlim(0.5, 3.5)
                ax1.set_xticklabels(['LeNet1', 'LeNet4', 'LeNet5'])
                ax1.tick_params(axis='both', which='major', labelsize=15)

                # Create the boxplot
                ax2.boxplot(list_resnet, widths=0.4, showmeans=True)
                ax2.set_xticks([1, 2])
                ax2.set_xlim(0.5, 2.5)
                ax2.set_xticklabels(['ResNet38', 'ResNet56'])
                ax2.tick_params(axis='both', which='major', labelsize=15)

                # Create the boxplot
                ax3.boxplot(list_wrn, widths=0.4, showmeans=True)
                ax3.set_xticks([1])
                ax3.set_xlim(0.5, 1.5)
                ax3.set_xticklabels(['WRN-28-10'])
                ax3.tick_params(axis='both', which='major', labelsize=15)

                fig.subplots_adjust(wspace=0.6)

                # Save the figure
                fig.savefig(result_path + '/plots/%s_d_mean_%d_%s.png' % (criterion, random_seed, stopping_type), dpi=600, bbox_inches='tight')
                plt.close(fig)


def draw_criterion(result_path, lib_config_list, data_file='analysis_raw.csv'):
    data = pd.read_csv(result_path + '/' + data_file, skipinitialspace=True)
    data['backend_version'] = data['backend_version'].astype(str)
    data['cuda_version'] = data['cuda_version'].astype(str)
    data['cudnn_version'] = data['cudnn_version'].astype(str)

    for criterion in ['accuracy', 'convergent_epoch', 'convergent']:
        for random_seed in [1, -1]:
            data_seed = data[(data['random_seed'] == random_seed)]
            list_lenet = print_criterion(data_seed, ['LeNet1', 'LeNet4', 'LeNet5'], lib_config_list, criterion)
            list_resnet = print_criterion(data_seed, ['ResNet38v1', 'ResNet56v1'], lib_config_list, criterion)
            list_wrn = print_criterion(data_seed, ['WRN-28-10'], lib_config_list, criterion)

            fig = plt.figure(figsize=(12, 3))
            ax1, ax2, ax3 = fig.subplots(1, 3, gridspec_kw={'width_ratios': [3, 2, 1]})

            def to_percent(temp, position):
                return '%1.1f' % (100 * temp) + '%'

            # Create the boxplot
            ax1.boxplot(list_lenet, widths=0.4, showmeans=True)
            ax1.set_xticks([1, 2, 3])
            ax1.set_xlim(0.5, 3.5)
            ax1.set_xticklabels(['LeNet1', 'LeNet4', 'LeNet5'])
            ax1.tick_params(axis='both', which='major', labelsize=15)
            if criterion == 'accuracy':
                ax1.yaxis.set_major_formatter(FuncFormatter(to_percent))

            # Create the boxplot
            ax2.boxplot(list_resnet, widths=0.4, showmeans=True)
            ax2.set_xticks([1, 2])
            ax2.set_xlim(0.5, 2.5)
            ax2.set_xticklabels(['ResNet38', 'ResNet56'])
            ax2.tick_params(axis='both', which='major', labelsize=15)
            if criterion == 'accuracy':
                ax2.yaxis.set_major_formatter(FuncFormatter(to_percent))

            # Create the boxplot
            ax3.boxplot(list_wrn, widths=0.4, showmeans=True)
            ax3.set_xticks([1])
            ax3.set_xlim(0.5, 1.5)
            ax3.set_xticklabels(['WRN-28-10'])
            ax3.tick_params(axis='both', which='major', labelsize=15)
            if criterion == 'accuracy':
                ax3.yaxis.set_major_formatter(FuncFormatter(to_percent))

            fig.subplots_adjust(wspace=0.6)

            # Save the figure
            fig.savefig(result_path + '/plots/%s_%d.png' % (criterion, random_seed), dpi=600, bbox_inches='tight')
            plt.close(fig)


def print_max_diff_mean_criterion(data, model_names, lib_config_list, criterion, stopping_type):
    model_list = []
    for subject in model_names:
        network_data = data[(data['network'].values == subject)]
        list = get_max_diff_mean_criterion(network_data, lib_config_list, criterion, stopping_type)
        model_list.append(list)
    return model_list


def print_criterion(data, model_names, lib_config_list, criterion):
    model_list = []
    for subject in model_names:
        network_data = data[(data['network'].values == subject)]
        list = get_criterion(network_data, lib_config_list, criterion)
        model_list.append(list)
    return model_list


def get_max_diff_mean_criterion(data, lib_config_list, criterion, stopping_type):
    assert criterion in ['max_accuracy_diff', 'max_convergent_diff', 'max_convergent_diff_epoch']
    assert stopping_type in ['best_loss', 'best_acc']
    if criterion == 'max_accuracy_diff':
        criterion_mean = 'mean_accuracy'
    elif criterion == 'max_convergent_diff':
        criterion_mean = 'mean_convergent'
    elif criterion == 'max_convergent_diff_epoch':
        criterion_mean = 'mean_convergent_epoch'
    else:
        raise Exception('Unknown criterion!')

    max_diff_mean_list = []
    for lib_config in lib_config_list:
        lower_libs = lib_config['lower_libs']
        for lower_lib_config in lower_libs:
            data_batch = data[(data['backend_version'].values == lib_config['backend_version']) &
                              (data['cuda_version'].values == lower_lib_config['cuda_version']) &
                              (data['cudnn_version'].values == lower_lib_config['cudnn_version'])]
            if len(data_batch) <= 0:
                continue
            data_stopping_type = data_batch[data_batch['stopping_type'] == stopping_type]
            max_acc_diff = data_stopping_type[criterion].values[0]
            mean_acc = data_stopping_type[criterion_mean].values[0]
            max_diff_mean = max_acc_diff / mean_acc

            max_diff_mean_list.append(max_diff_mean)
    return np.array(max_diff_mean_list)


def get_criterion(data, lib_config_list, criterion):
    assert criterion in ['accuracy', 'convergent_epoch', 'convergent']
    best_gap = -1
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
                accuracy = data_batch[criterion].values
                gap = max(accuracy) - min(accuracy)
                if gap > best_gap:
                    best_gap = gap
                    best_acc = accuracy
                    best_config['backend_version'] = lib_config['backend_version']
                    best_config['cuda_version'] = lower_lib_config['cuda_version']
                    best_config['cudnn_version'] = lower_lib_config['cudnn_version']
                    best_config['stopping_type'] = stopping_type

    # print(best_config)
    # print(best_var)
    # print(best_acc)
    return best_acc


def draw_acc_or_epoch_between_backend(result_path, data_file='analysis_raw.csv'):
    data = pd.read_csv(result_path + data_file, skipinitialspace=True)
    data['backend_version'] = data['backend_version'].astype(str)
    data['cuda_version'] = data['cuda_version'].astype(str)
    data['cudnn_version'] = data['cudnn_version'].astype(str)

    backends = ['tensorflow', 'cntk', 'theano']

    for acc_or_epoch in ['accuracy', 'convergent_epoch']:
        for random_seed in [1, -1]:
            list_lenet = print_acc_or_epoch_between_backend(data, ['LeNet1', 'LeNet4', 'LeNet5'], acc_or_epoch, random_seed)
            list_resnet = print_acc_or_epoch_between_backend(data, ['ResNet38v1', 'ResNet56v1'], acc_or_epoch, random_seed)
            list_wrn = print_acc_or_epoch_between_backend(data, ['WRN-28-10'], acc_or_epoch, random_seed)

            fig = plt.figure(figsize=(12, 3))
            ax1, ax2, ax3 = fig.subplots(1, 3, gridspec_kw={'width_ratios': [3, 2, 1]})

            def to_percent(temp, position):
                return '%1.1f' % (100 * temp) + '%'

            width = 0.4
            # Create the boxplot
            bp1 = ax1.boxplot(list_lenet[0], positions=[1, 5, 9], widths=width, patch_artist=True, showmeans=True,
                              boxprops=dict(facecolor="C0"), medianprops=dict(color='C3'),
                              meanprops=dict(markeredgecolor='purple', markerfacecolor='purple'))
            bp2 = ax1.boxplot(list_lenet[1], positions=[2, 6, 10], widths=width, patch_artist=True, showmeans=True,
                              boxprops=dict(facecolor="C1"), medianprops=dict(color='C3'),
                              meanprops=dict(markeredgecolor='purple', markerfacecolor='purple'))
            bp3 = ax1.boxplot(list_lenet[2], positions=[3, 7, 11], widths=width, patch_artist=True, showmeans=True,
                              boxprops=dict(facecolor="C2"), medianprops=dict(color='C3'),
                              meanprops=dict(markeredgecolor='purple', markerfacecolor='purple'))
            ax1.set_xticks([2, 6, 10])
            ax1.set_xlim(0.5, 11.5)
            ax1.set_xticklabels(['LeNet1', 'LeNet4', 'LeNet5'])
            ax1.legend([bp1["boxes"][0], bp2["boxes"][0], bp3["boxes"][0]], backends, loc='upper right')
            ax1.get_legend().set_visible(False)
            ax1.tick_params(axis='both', which='major', labelsize='x-large')
            if acc_or_epoch == 'accuracy':
                ax1.yaxis.set_major_formatter(FuncFormatter(to_percent))

            # Create the boxplot
            bp1 = ax2.boxplot(list_resnet[0], positions=[1, 5], widths=width, patch_artist=True, showmeans=True,
                              boxprops=dict(facecolor="C0"), medianprops=dict(color='C3'),
                              meanprops=dict(markeredgecolor='purple', markerfacecolor='purple'))
            bp2 = ax2.boxplot(list_resnet[1], positions=[2, 6], widths=width, patch_artist=True, showmeans=True,
                              boxprops=dict(facecolor="C1"), medianprops=dict(color='C3'),
                              meanprops=dict(markeredgecolor='purple', markerfacecolor='purple'))
            bp3 = ax2.boxplot(list_resnet[2], positions=[3, 7], widths=width, patch_artist=True, showmeans=True,
                              boxprops=dict(facecolor="C2"), medianprops=dict(color='C3'),
                              meanprops=dict(markeredgecolor='purple', markerfacecolor='purple'))
            ax2.set_xticks([2, 6])
            ax2.set_xlim(0.5, 7.5)
            ax2.set_xticklabels(['ResNet38', 'ResNet56'])
            ax2.tick_params(axis='both', which='major', labelsize='x-large')
            ax2.legend([bp1["boxes"][0], bp2["boxes"][0], bp3["boxes"][0]], backends, loc='upper right')
            ax2.get_legend().set_visible(False)
            if acc_or_epoch == 'accuracy':
                ax2.yaxis.set_major_formatter(FuncFormatter(to_percent))

            # Create the boxplot
            bp1 = ax3.boxplot(list_wrn[0], positions=[1], widths=width, patch_artist=True, showmeans=True,
                              boxprops=dict(facecolor="C0"), medianprops=dict(color='C3'),
                              meanprops=dict(markeredgecolor='purple', markerfacecolor='purple'))
            bp2 = ax3.boxplot(list_wrn[1], positions=[2], widths=width, patch_artist=True, showmeans=True,
                              boxprops=dict(facecolor="C1"), medianprops=dict(color='C3'),
                              meanprops=dict(markeredgecolor='purple', markerfacecolor='purple'))
            bp3 = ax3.boxplot(list_wrn[2], positions=[3], widths=width, patch_artist=True, showmeans=True,
                              boxprops=dict(facecolor="C2"), medianprops=dict(color='C3'),
                              meanprops=dict(markeredgecolor='purple', markerfacecolor='purple'))
            ax3.set_xticks([2])
            ax3.set_xlim(0.5, 3.5)
            ax3.set_xticklabels(['WRN-28-10'])
            ax3.tick_params(axis='both', which='major', labelsize='x-large')
            ax3.legend([bp1["boxes"][0], bp2["boxes"][0], bp3["boxes"][0]], backends, fontsize='x-large', loc='upper right')
            ax3.get_legend().set_visible(False)
            if acc_or_epoch == 'accuracy':
                ax3.yaxis.set_major_formatter(FuncFormatter(to_percent))

            fig.subplots_adjust(wspace=0.6)

            fig.legend([bp1["boxes"][0], bp2["boxes"][0], bp3["boxes"][0]], ['TensorFlow', 'CNTK', 'Theano'],
                       frameon=False, fontsize='x-large', ncol=3, loc='upper center')

            # Save the figure
            fig.savefig(result_path + '/plots/%s_between_backend_%d.png' % (acc_or_epoch, random_seed), dpi=600, bbox_inches='tight')
            plt.close(fig)


def print_acc_or_epoch_between_backend(data, model_names, accuracy_or_epoch='accuracy', random_seed=1):
    assert accuracy_or_epoch in ['accuracy', 'convergent_epoch']

    backends = ['tensorflow', 'cntk', 'theano']
    lib_data = data[(data['cuda_version'] == '10.0') & (data['cudnn_version'] == '7.6') & (data['random_seed'] == random_seed)]
    backend_acc_list = [[], [], []]
    backend_i = -1
    for backend in backends:
        backend_i += 1
        for subject in model_names:
            network_data = lib_data[(lib_data['network'].values == subject)]
            subject_data = network_data[(network_data['backend'].values == backend)]
            subject_data = subject_data[(subject_data['stopping_type'].values == 'best_loss')]
            subject_accuracies = subject_data[accuracy_or_epoch].values
            backend_acc_list[backend_i].append(subject_accuracies)

    return backend_acc_list


def main():
    result_dir = 'result_per_epoch_re_run/'

    training_type_list = ['from_scratch']

    backend_versions = ['1.10.0', '1.12.0', '1.14.0']

    cudnns = ['7.3', '7.4', '7.5', '7.6']

    # seed_list = [1, -1]
    seed_list = [1]

    stopping_type_list = ['best_loss', 'best_acc', 'epoch']

    model_names = ['LeNet1', 'LeNet4', 'LeNet5',
                   'ResNet38v1', 'ResNet56v1',
                   'WRN-28-10']

    plot_values = ['max_accuracy_diff', 'std_dev_accuracy', 'mean_accuracy']

    analysis_path = result_dir + '/analysis_result.csv'

    data = pd.read_csv(analysis_path, skipinitialspace=True)

    data['cudnn_version'] = data['cudnn_version'].astype(str)

    data = data.sort_values(by=['backend_version', 'cudnn_version', 'mean_accuracy'])

    for plot_value in plot_values:
        for seed in seed_list:
            seed_data = data[data['random_seed'] == seed]
            for stopping_type in stopping_type_list:
                stopping_data = seed_data[seed_data['stopping_type'] == stopping_type]
                for backend_version in backend_versions:
                    backend_version_data = stopping_data[stopping_data['backend_version'] == backend_version]

                    fig, ax = plt.subplots(figsize=(7, 7))

                    backend_version_data.groupby(['cudnn_version', 'network']).sum()[plot_value].unstack().plot(ax=ax)

                    fig.savefig(result_dir + "/plots/%s_%d_%s_%s.png" % (plot_value, seed, stopping_type, backend_version))
                    plt.close(fig)

                for cudnn in cudnns:
                    cudnn_data = stopping_data[stopping_data['cudnn_version'] == cudnn]

                    fig, ax = plt.subplots(figsize=(7, 7))

                    cudnn_data.groupby(['backend_version', 'network']).sum()[plot_value].unstack().plot(ax=ax)

                    fig.savefig(result_dir + "/plots/%s_%d_%s_%s.png" % (plot_value, seed, stopping_type, cudnn))
                    plt.close(fig)

                for model_name in model_names:
                    model_data = stopping_data[stopping_data['network'] == model_name]

                    fig, ax = plt.subplots(figsize=(5, 4))

                    plt.locator_params(axis='x', nbins=4)

                    model_data.groupby(['cudnn_version', 'backend_version']).sum()[plot_value].unstack().plot(ax=ax)

                    ax.set_xticklabels(['', '7.3', '7.4', '7.5', '7.6'])

                    fig.savefig(result_dir + "/plots/%s_%d_%s_%s.png" % (plot_value, seed, stopping_type, model_name))
                    plt.close(fig)


def draw_something_new():
    result_dir = 'result_per_epoch_re_run/'

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

    draw_max_accuracy_diff(result_dir, lib_config_list)
    draw_criterion(result_dir, lib_config_list)
    draw_acc_or_epoch_between_backend(result_dir)
    draw_max_diff_mean_criterion(result_dir, lib_config_list)


if __name__ == "__main__":
    # main()
    draw_something_new()
