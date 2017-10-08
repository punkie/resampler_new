import csv
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import QLabel, QComboBox, QPushButton

from resampling_methods import ResamplingAlgorithms


class DatasetLoader(QThread):

    update_progress_bar = pyqtSignal(int)
    update_dataset = pyqtSignal(dict)

    def __init__(self, main_window, path):
        super(DatasetLoader, self).__init__()
        self.path = path
        self.main_window = main_window
        main_window.findChild(QComboBox, "resamplingAlgorithms").addItems([ra.value[0] for ra in ResamplingAlgorithms])
        main_window.state.sampling_algorithm = ResamplingAlgorithms.RO

    def __update_other_widgets(self):
        self.main_window.findChild(QLabel, "datasetPickedLabel").setText(self.path)
        self.main_window.findChild(QComboBox, "resamplingAlgorithms").setEnabled(True)
        self.main_window.findChild(QPushButton, "outputDirectoryButton").setEnabled(True)

    def __load_dataset(self):
        with open(self.path, newline="") as csv_input_file:
            reader = csv.reader(csv_input_file, delimiter=",")
            dataset = {}
            dataset['name'] = self.path.split("/")[-1].split(".csv")[0]
            dataset['x_values'] = np.array([])
            dataset['y_values'] = np.array([])
            appending_first_row = True
            count = 0
            for row in reader:
                row_floats = list(map(float, row))
                count += 1
                if appending_first_row:
                    dataset['x_values'] = np.concatenate((dataset['x_values'], np.array(row_floats[:-1])))
                    appending_first_row = False
                else:
                    dataset['x_values'] = np.vstack((dataset['x_values'], np.array(row_floats[:-1])))
                dataset['y_values'] = np.append(dataset['y_values'], row_floats[-1])
                #time.sleep(0.01)
                self.update_progress_bar.emit(count)
            self.update_dataset.emit(dataset)

    def run(self):
        self.__update_other_widgets()
        self.__load_dataset()
