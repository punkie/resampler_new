from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import QLabel, QComboBox, QPushButton, QProgressBar
from general_functions import load_dataset, compute_some_statistics_for_the_dataset, get_statistics_string


class DatasetLoader(QThread):

    update_dataset_load_progress_bar = pyqtSignal(int)
    update_dataset = pyqtSignal(dict)

    def __init__(self, main_window, path):
        super(DatasetLoader, self).__init__()
        self.path = path
        self.main_window = main_window

    def __update_other_widgets(self):
        self.main_window.findChild(QLabel, "datasetPickedLabel").setText(self.path)
        self.main_window.findChild(QComboBox, "resamplingAlgorithms").setEnabled(True)
        self.main_window.findChild(QPushButton, "outputDirectoryButton").setEnabled(True)
        dataset_statistics = get_statistics_string(self.main_window.state.dataset)
        self.main_window.findChild(QLabel, "datasetStatisticsLabel").setText(dataset_statistics)
        self.main_window.findChild(QLabel, "resampledDatasetStatistics").setText(" ")
        self.main_window.findChild(QLabel, "resamplingStatusLabel").setText(" ")
        self.main_window.findChild(QPushButton, "imgDiffsButton").setEnabled(False)
        self.main_window.findChild(QPushButton, "classifyButton").setEnabled(True)
        self.main_window.findChild(QLabel, "classifyingStatusLabel").setText(" ")
        self.main_window.findChild(QProgressBar, "classifyProgressBar").setValue(0)

    def __load_dataset(self):
        load_dataset(self)

    def __compute_some_statistics_for_the_dataset(self):
        dataset = self.main_window.state.dataset
        compute_some_statistics_for_the_dataset(dataset)

    def run(self):
        self.__load_dataset()
        self.__compute_some_statistics_for_the_dataset()
        self.__update_other_widgets()

