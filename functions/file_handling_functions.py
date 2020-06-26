import csv
import numpy as np
from collections import deque

import pandas as pd
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
    first_row = pd.read_csv(dataset_loader.path, delimiter=',', header=0, nrows=1)
    header_row = has_header(first_row)
    tfr = pd.read_csv(dataset_loader.path, delimiter=',', iterator=True, header=header_row)
    ds_as_dataframe = pd.concat(tfr)
    dataset = dict()
    columns_length = len(ds_as_dataframe.columns)
    if header_row:
        dataset['header_row'] = first_row.columns.to_numpy().flatten()
    else:
        dataset['header_row'] = np.array(['X_{}'.format(i) for i in range(columns_length - 1)] + ['Y'])
    dataset['dataset_as_dataframe'] = ds_as_dataframe
    dataset['x_values'] = ds_as_dataframe.iloc[:, :columns_length-1].to_numpy()
    dataset['y_values'] = ds_as_dataframe.iloc[:, columns_length-1:].to_numpy().flatten()
    dataset['name'] = dataset_loader.path.split("/")[-1].split(".csv")[0]
    # print ('asd')

    # with open(dataset_loader.path, newline="") as csv_input_file:
    #     # try:
    #         reader = csv.reader(csv_input_file, delimiter=",")
    #         reader_as_list = list(reader)
    #
    #         dataset['name'] = dataset_loader.path.split("/")[-1].split(".csv")[0]
    #         x_values = deque()
    #         y_values = deque()
    #         rows_count = 0
    #         dataset_loader.main_window.widgets.\
    #             get_progress_bar(Widgets.ProgressBars.DatasetProgressBar.value).setMinimum(0)
    #         dataset_loader.main_window.widgets.\
    #             get_progress_bar(Widgets.ProgressBars.DatasetProgressBar.value).setMaximum(0)
    #         for row in reader_as_list:
    #             # this here is very dangerous. should be refactored.
    #             row_floats = list(map(float, row))
    #             rows_count += 1
    #             x_values.append(row_floats[:-1])
    #             y_values.append(row_floats[-1])
    #             # dataset_loader.update_dataset_load_progress_bar_signal.emit(rows_count)
    #         dataset['x_values'] = np.array(x_values)
    #         dataset['y_values'] = np.array(y_values)
    dataset_loader.main_window.state.dataset = dataset
        # except Exception:
        #     print ("wow")


def has_header(dataframe):
    for el in dataframe:
        try:
            if str(el).count(".") == 2:
                continue
            float(el)
        except ValueError:
            return 1
    return None


def __labelize_dataset(path, output_file_path):
    with open(path, newline="") as csv_input_file:
        reader = csv.reader(csv_input_file, delimiter=",")
        with open(output_file_path, "w", newline="\n") \
                as csv_output_file:
            dataset_writer = csv.writer(csv_output_file, delimiter=",")
            reader_as_list = list(reader)
            parse_header = True
            for row in reader_as_list:
                if parse_header:
                    dict_with_labelencoders = __parse_custom_header(row)
                    parse_header = False
                    continue
                labelized_row = []
                for idx, element in enumerate(row):
                    if dict_with_labelencoders.get(str(idx)) != None:
                        if element == "''":
                            labelized_row.append(dict_with_labelencoders.get(str(idx)).index(""))
                        else:
                            labelized_row.append(
                                dict_with_labelencoders.get(str(idx)).index(element.replace("'", "")))
                    else:
                        labelized_row.append(element)
                dataset_writer.writerow(labelized_row)


def __parse_custom_header(row):
    result_dict = dict()
    for element in row:
        el_idx, el_list = element.split('=')
        el_list = eval(el_list.replace(";", ","))
        # result_dict[el_idx] = preprocessing.LabelEncoder().fit(el_list)
        result_dict[el_idx] = el_list
    return result_dict


def __binarize_custom_dataset():
    with open("E:/python-workspace/resampler/non-binarized-dataset/eula.csv", newline="") as csv_input_file:
        reader = csv.reader(csv_input_file, delimiter=",")
        with open("E:/python-workspace/resampler/binarized-datasets/ecoli-custom.csv", "w", newline="\n")\
                as csv_output_file:
            dataset_writer = csv.writer(csv_output_file, delimiter=",")
            for row in reader:
                row[-1] = 1 if row[-1] == 'om' else -1
                dataset_writer.writerow(row[1:])


def __extract_binarized_imbalanced_datasets():
    for dataset_name, dataset_values in fetch_datasets().items():
        write_dataset_to_csv("./binarized-datasets/" + dataset_name + ".csv", dataset_values)

# if __name__ == '__main__':
    # __labelize_dataset("E:/python-workspace/resampler/binarized-datasets/"
    #                    "2_Class_Data_February_Cleaned_with_custom_header.csv",
    #                    "E:/python-workspace/resampler/binarized-datasets/custom_ds.csv")
    #__extract_binarized_imbalanced_datasets()