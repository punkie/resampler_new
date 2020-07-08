from PyQt5.QtCore import QThread, pyqtSignal

from functions.file_handling_functions import load_dataset
from functions.general_functions import compute_some_statistics_for_the_dataset
from rs_types.widgets import Widgets


class DatasetLoader(QThread):

    update_dataset_load_progress_bar_signal = pyqtSignal(object)
    update_gui_after_dataset_load_signal = pyqtSignal(str)
    reraise_non_mt_exception_signal = pyqtSignal(Exception)

    def __init__(self, main_window, path):
        super(DatasetLoader, self).__init__()
        self.path = path
        self.main_window = main_window
        self.__custom_pre_process()

    def run(self):
        try:
            load_dataset(self)
            compute_some_statistics_for_the_dataset(self.main_window.state.dataset, self.main_window.state, False)
        except Exception as e:
            self.reraise_non_mt_exception_signal.emit(e)
        # load_table(self.main_window)
        self.__custom_post_process()

    def __custom_post_process(self):
        self.update_gui_after_dataset_load_signal.emit(self.path)

    def __custom_pre_process(self):
        self.main_window.setEnabled(False)
        self.main_window.widgets.get_progress_bar(Widgets.ProgressBars.DatasetProgressBar.value).setVisible(True)
        self.main_window.widgets.get_progress_bar(Widgets.ProgressBars.DatasetProgressBar.value).setMaximum(100)
        self.main_window.widgets.get_label(Widgets.Labels.DatasetLoadingResultLabel.value).setText("")
        self.main_window.widgets.get_label(Widgets.Labels.FilePathLabel.value).setText("")
        self.main_window.widgets.get_label(Widgets.Labels.DatasetPickedLabel.value).setText("")
        self.main_window.widgets.get_label(Widgets.Labels.DatasetLoadingTextLabel.value).setText("Loading...")

        # self.main_window.widgets.get_button(Widgets.Buttons.ShowROCGraphs.value).setEnabled(False)
        # self.main_window.widgets.get_button(Widgets.Buttons.ShowPRGraphs.value).setEnabled(False)
        # self.main_window.widgets.get_button(Widgets.Buttons.ClassifyButton.value).setEnabled(False)
        # self.main_window.widgets.get_combo_box(Widgets.ComboBoxes.ClassificationAlgorithms.value).setEnabled(False)
        # self.main_window.widgets. \
        #     get_label(Widgets.Labels.AfterClassificationStatistics.value).setText(" ")