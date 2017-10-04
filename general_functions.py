import csv
import numpy as np
from imblearn.datasets import fetch_datasets


def write_dataset_to_csv(path, dataset=None, x_values_param=None, y_values_param=None):
    if dataset is not None:
        x_values_param = dataset['data']
        y_values_param = dataset['target']
    with open(path, "w", newline="\n") as csv_output_file:
        dataset_writer = csv.writer(csv_output_file, delimiter=",")
        for row_idx, dv in enumerate(x_values_param):
            dataset_writer.writerow(np.append(dv, y_values_param[row_idx]))

def load_dataset(path):
    with open(path, newline="") as csv_input_file:
        reader = csv.reader(csv_input_file, delimiter=",")
        dataset = {}
        dataset['x_values'] = np.array([])
        dataset['y_values'] = np.array([])
        appending_first_row = True
        for row in reader:
            if appending_first_row:
                dataset['x_values'] = np.concatenate((dataset['x_values'], np.array(row[:-1])))
                appending_first_row = False
            else:
                dataset['x_values'] = np.vstack((dataset['x_values'], np.array(row[:-1])))
            dataset['y_values'] = np.append(dataset['y_values'], row[-1])
        return dataset

def extract_binarized_imbalanced_datasets():
    for dataset_name, dataset_values in fetch_datasets().items():
        write_dataset_to_csv("./binarized-datasets/" + dataset_name + ".csv", dataset_values)

