import csv
import numpy as np
from collections import deque
from imblearn.datasets import fetch_datasets
from rs_types.widgets import Widgets


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
        reader = csv.reader(csv_input_file, delimiter=",")
        reader_as_list = list(reader)
        dataset = dict()
        dataset['name'] = dataset_loader.path.split("/")[-1].split(".csv")[0]
        x_values = deque()
        y_values = deque()
        rows_count = 0
        dataset_loader.main_window.widgets.\
            get_progress_bar(Widgets.ProgressBars.DatasetProgressBar.value).setMaximum(len(reader_as_list))
        for row in reader_as_list:
            # this here is very dangerous. should be refactored.
            row_floats = list(map(float, row))
            rows_count += 1
            x_values.append(row_floats[:-1])
            y_values.append(row_floats[-1])
            dataset_loader.update_dataset_load_progress_bar.emit(rows_count)
        dataset['x_values'] = np.array(x_values)
        dataset['y_values'] = np.array(y_values)
        dataset_loader.main_window.state.dataset = dataset


def __binarize_custom_dataset():
    with open("E:/python-workspace/resampler/non-binarized-dataset/eula.csv", newline="") as csv_input_file:
        reader = csv.reader(csv_input_file, delimiter=",")
        with open("E:/python-workspace/resampler/binarized-datasets/ecoli-custom.csv", "w", newline="\n") as csv_output_file:
            dataset_writer = csv.writer(csv_output_file, delimiter=",")
            for row in reader:
                row[-1] = 1 if row[-1] == 'om' else -1
                dataset_writer.writerow(row[1:])


def __extract_binarized_imbalanced_datasets():
    for dataset_name, dataset_values in fetch_datasets().items():
        write_dataset_to_csv("./binarized-datasets/" + dataset_name + ".csv", dataset_values)

# if __name__ == '__main__':
#     extract_binarized_imbalanced_datasets()