from PyQt5.QtCore import QThread, pyqtSignal
from functions.file_handling_functions import load_dataset
from functions.general_functions import compute_some_statistics_for_the_dataset
from rs_types.widgets import Widgets


class DatasetLoader(QThread):

    update_dataset_load_progress_bar = pyqtSignal(object)

    def __init__(self, main_window, path):
        super(DatasetLoader, self).__init__()
        self.path = path
        self.main_window = main_window
        self.__custom_pre_process()

    def run(self):
        load_dataset(self)
        compute_some_statistics_for_the_dataset(self.main_window.state.dataset)
        self.__custom_post_process()

    def __custom_post_process(self):
        self.__update_widgets()

    def __custom_pre_process(self):
        self.main_window.widgets.get_progress_bar(Widgets.ProgressBars.DatasetProgressBar.value).setValue(0)
        self.main_window.widgets.get_button(Widgets.Buttons.ShowROCGraphs.value).setEnabled(False)
        self.main_window.widgets.get_button(Widgets.Buttons.ShowPRGraphs.value).setEnabled(False)

    def __update_widgets(self):
        self.main_window.widgets.get_label(Widgets.Labels.DatasetPickedLabel.value).setText(self.path)
        self.main_window.widgets.\
            get_label(Widgets.Labels.DatasetStatisticsLabel.value).setText(self.main_window.state.dataset['dataset_statistics_string'])
        self.main_window.widgets.get_combo_box(Widgets.ComboBoxes.ResamplingAlgorithms.value).setEnabled(True)
        self.main_window.widgets.get_button(Widgets.Buttons.OutputDirectoryButton.value).setEnabled(True)
        self.main_window.widgets.get_label(Widgets.Labels.ResampledDatasetStatistics.value).setText(" ")
        self.main_window.widgets.get_label(Widgets.Labels.ResamplingStatusLabel.value).setText(" ")
        self.main_window.widgets.get_button(Widgets.Buttons.ImgDiffsButton.value).setEnabled(False)
        self.main_window.widgets.get_button(Widgets.Buttons.ClassifyButton.value).setEnabled(True)
        self.main_window.widgets.get_label(Widgets.Labels.ClassifyingStatusLabel.value).setText(" ")
        self.main_window.widgets.get_progress_bar(Widgets.ProgressBars.NormalClassifyProgressBar.value).setValue(0)
        self.main_window.widgets.get_progress_bar(Widgets.ProgressBars.ResampleClassifyProgressBar.value).setValue(0)