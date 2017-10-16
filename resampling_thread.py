from PyQt5.QtCore import QThread
from PyQt5.QtWidgets import QLabel, QPushButton

from general_functions import compute_some_statistics_for_the_dataset, get_statistics_string
from resampling_functions import do_resampling
from widgets import Widgets


class Resampling(QThread):

    def __init__(self, main_window):
        super(Resampling, self).__init__()
        self.main_window = main_window

    def run(self):
        self.main_window.widgets.get_label(Widgets.Labels.ResamplingStatusLabel.value).setText("Started resampling. Please wait...")
        resampled_dataset = do_resampling(self.main_window.state)
        compute_some_statistics_for_the_dataset(resampled_dataset)
        self.main_window.state.resampled_dataset = resampled_dataset
        self.main_window.widgets.get_label(Widgets.Labels.ResamplingStatusLabel.value).setText(
            "Done!")
        self.main_window.widgets.get_label(Widgets.Labels.ResampledDatasetStatistics.value).setText(get_statistics_string(resampled_dataset))
        self.main_window.widgets.get_button(Widgets.Buttons.ImgDiffsButton.value).setEnabled(True)