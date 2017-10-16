from PyQt5.QtCore import QThread, pyqtSignal
from general_functions import load_dataset, compute_some_statistics_for_the_dataset, get_statistics_string
from widgets import Widgets


class DatasetLoader(QThread):

    update_dataset_load_progress_bar = pyqtSignal(object)

    def __init__(self, main_window, path):
        super(DatasetLoader, self).__init__()
        self.path = path
        self.main_window = main_window
        self.main_window.widgets.get_progress_bar(Widgets.ProgressBars.DatasetProgressBar.value).setValue(0)
        self.main_window.widgets.get_button(Widgets.Buttons.ShowROCGraphs.value).setEnabled(False)
        self.main_window.widgets.get_button(Widgets.Buttons.ShowPRGraphs.value).setEnabled(False)

    def __update_other_widgets(self):
        self.main_window.widgets.get_label(Widgets.Labels.DatasetPickedLabel.value).setText(self.path)
        self.main_window.widgets.get_combo_box(Widgets.ComboBoxes.ResamplingAlgorithms.value).setEnabled(True)
        self.main_window.widgets.get_button(Widgets.Buttons.OutputDirectoryButton.value).setEnabled(True)
        dataset_statistics = get_statistics_string(self.main_window.state.dataset)
        self.main_window.widgets.get_label(Widgets.Labels.DatasetStatisticsLabel.value).setText(dataset_statistics)
        self.main_window.widgets.get_label(Widgets.Labels.ResampledDatasetStatistics.value).setText(" ")
        self.main_window.widgets.get_label(Widgets.Labels.ResamplingStatusLabel.value).setText(" ")
        self.main_window.widgets.get_button(Widgets.Buttons.ImgDiffsButton.value).setEnabled(False)
        self.main_window.widgets.get_button(Widgets.Buttons.ClassifyButton.value).setEnabled(True)
        self.main_window.widgets.get_label(Widgets.Labels.ClassifyingStatusLabel.value).setText(" ")
        self.main_window.widgets.get_progress_bar(Widgets.ProgressBars.NormalClassifyProgressBar.value).setValue(0)
        self.main_window.widgets.get_progress_bar(Widgets.ProgressBars.ResampleClassifyProgressBar.value).setValue(0)

    def __load_dataset(self):
        load_dataset(self)

    def __compute_some_statistics_for_the_dataset(self):
        dataset = self.main_window.state.dataset
        compute_some_statistics_for_the_dataset(dataset)

    def run(self):
        self.__load_dataset()
        self.__compute_some_statistics_for_the_dataset()
        self.__update_other_widgets()
        # QApplication.processEvents()
