import csv
import numpy as np
from imblearn.datasets import fetch_datasets

if __name__ == "__main__":
    for dataset_name, dataset_values in fetch_datasets().items():
        with open("./binarized-datasets/" + dataset_name + ".csv", "w", newline="\n") as csv_output_file:
            dataset_writer = csv.writer(csv_output_file, delimiter=',')
            target_values = dataset_values['target']
            for row_idx, dv in enumerate(dataset_values['data']):
                dataset_writer.writerow(np.append(dv, target_values[row_idx]))
