import csv
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QProgressBar
from imblearn.datasets import fetch_datasets
from sklearn.decomposition import PCA



def write_dataset_to_csv(path, dataset=None, x_values_param=None, y_values_param=None):
    if dataset is not None:
        x_values_param = dataset['data']
        y_values_param = dataset['target']
    with open(path, "w", newline="\n") as csv_output_file:
        dataset_writer = csv.writer(csv_output_file, delimiter=",")
        for row_idx, dv in enumerate(x_values_param):
            dataset_writer.writerow(np.append(dv, y_values_param[row_idx]))


def load_dataset(dataset_loader):
    with open(dataset_loader.path, newline="") as csv_input_file:
        try:
            reader = csv.reader(csv_input_file, delimiter=",")
            reader_as_list = list(reader)
            dataset = {}
            dataset['name'] = dataset_loader.path.split("/")[-1].split(".csv")[0]
            dataset['x_values'] = np.array([])
            dataset['y_values'] = np.array([])
            appending_first_row = True
            count = 0
            dataset_loader.main_window.findChild(QProgressBar, "datasetProgressBar").setMaximum(len(reader_as_list))
            for row in reader_as_list:
                row_floats = list(map(float, row))
                count += 1
                if appending_first_row:
                    dataset['x_values'] = np.concatenate((dataset['x_values'], np.array(row_floats[:-1])))
                    appending_first_row = False
                else:
                    dataset['x_values'] = np.vstack((dataset['x_values'], np.array(row_floats[:-1])))
                dataset['y_values'] = np.append(dataset['y_values'], row_floats[-1])

                # time.sleep(0.01)
                dataset_loader.update_progress_bar.emit(count)
            dataset_loader.update_dataset.emit(dataset)
        except Exception as e:
            print(e)


def compute_some_statistics_for_the_dataset(dataset):
    dataset['number_of_examples'] = len(dataset['y_values'])
    unique_target_values = Counter(dataset['y_values'])
    first_mc_tuple, second_mc_tuple = unique_target_values.most_common(2)
    dataset['y_values_as_set'] = (first_mc_tuple[0], second_mc_tuple[0])
    dataset['target_class_percentage'] = second_mc_tuple[1] / first_mc_tuple[1]


def get_statistics_string(dataset):
    dataset_statistics = 'Number of examples: {}\nPercentage between target examples: {}'.format(
        dataset['number_of_examples'], dataset['target_class_percentage'])
    return dataset_statistics


def draw_comparision_picture(normal_dataset, resampled_dataset, sampling_algo_name):
    pca = PCA(n_components=2)
    x_visible = pca.fit_transform(normal_dataset['x_values'])
    x_resampled_vis = pca.transform(resampled_dataset['x_values'])

    y_values = normal_dataset['y_values']
    y_values_as_set = normal_dataset['y_values_as_set']

    y_resampled_values = resampled_dataset['y_values']

    tc_one, tc_two = y_values_as_set

    f, (ax1, ax2) = plt.subplots(1, 2)
    c0 = ax1.scatter(x_visible[y_values == tc_one, 0], x_visible[y_values == tc_one, 1], label="Class #0",
                     alpha=0.5)
    c1 = ax1.scatter(x_visible[y_values == tc_two, 0], x_visible[y_values == tc_two, 1], label="Class #1",
                     alpha=0.5)
    ax1.set_title('Original set')

    ax2.scatter(x_resampled_vis[y_resampled_values == tc_one, 0], x_resampled_vis[y_resampled_values == tc_one, 1],
                label="Class #0", alpha=.5)
    ax2.scatter(x_resampled_vis[y_resampled_values == tc_two, 0], x_resampled_vis[y_resampled_values == tc_two, 1],
                label="Class #1", alpha=.5)
    ax2.set_title(sampling_algo_name)

    # make nice plotting
    for ax in (ax1, ax2):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()
        ax.spines['left'].set_position(('outward', 10))
        ax.spines['bottom'].set_position(('outward', 10))
        ax.set_xlim([-6, 8])
        ax.set_ylim([-6, 6])

    plt.figlegend((c0, c1), ('Class #0', 'Class #1'), loc='lower center',
                  ncol=2, labelspacing=0.)
    plt.tight_layout(pad=3)
    plt.show()


def extract_binarized_imbalanced_datasets():
    for dataset_name, dataset_values in fetch_datasets().items():
        write_dataset_to_csv("./binarized-datasets/" + dataset_name + ".csv", dataset_values)

