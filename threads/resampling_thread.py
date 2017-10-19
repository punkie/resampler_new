from PyQt5.QtCore import QThread
from functions.general_functions import compute_some_statistics_for_the_dataset
from functions.resampling_functions import do_resampling
from rs_types.widgets import Widgets


class Resampling(QThread):

    def __init__(self, main_window):
        super(Resampling, self).__init__()
        self.main_window = main_window
        self.__custom_pre_process()

    def run(self):
        resampled_dataset = do_resampling(self.main_window.state)
        compute_some_statistics_for_the_dataset(resampled_dataset)
        self.main_window.state.resampled_dataset = resampled_dataset
        self.__custom_post_process()

    def __custom_pre_process(self):
        self.main_window.widgets.get_label(Widgets.Labels.ResamplingStatusLabel.value).setText(
            "Started resampling. Please wait...")

    def __custom_post_process(self):
        self.main_window.widgets.get_label(Widgets.Labels.ResamplingStatusLabel.value).setText(
            "Done!")
        self.main_window.widgets.get_label(Widgets.Labels.ResampledDatasetStatistics.value).setText(
            self.main_window.state.resampled_dataset['dataset_statistics_string'])
        self.main_window.widgets.get_button(Widgets.Buttons.ImgDiffsButton.value).setEnabled(True)