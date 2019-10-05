from PyQt5.QtCore import QThread, pyqtSignal
from functions.general_functions import classify, get_mean_precision_recall_f1_scores
from rs_types.widgets import Widgets


class Classifying(QThread):

    update_normal_classify_progress_bar = pyqtSignal(int)
    update_resample_classify_progress_bar = pyqtSignal(int)
    update_gui_after_classification = pyqtSignal()

    def __init__(self, main_window, do_resampling):
        super(Classifying, self).__init__()
        self.main_window = main_window
        self.do_resampling = do_resampling
        self.__custom_pre_process()

    def run(self):
        self.main_window.widgets.\
            get_label(Widgets.Labels.ClassifyingStatusLabel.value).setText("Started classifying. Please wait...")
        classified_data = classify(self)
        self.__store_classified_data(classified_data)
        self.__custom_post_process()

    def __store_classified_data(self, classified_data):
        if self.do_resampling:
            self.main_window.state.classified_data_resampled_case = classified_data
        else:
            self.main_window.state.classified_data_normal_case = classified_data

    def __custom_post_process(self):
        if self.do_resampling:
            self.main_window.state.resample_classify_thread_finished = True
        else:
            self.main_window.state.normal_classify_thread_finished = True
        self.update_gui_after_classification.emit()

    def __custom_pre_process(self):
        self.main_window.widgets.get_progress_bar(Widgets.ProgressBars.ResampleClassifyProgressBar.value).setValue(
            0)
        self.main_window.widgets.get_progress_bar(Widgets.ProgressBars.NormalClassifyProgressBar.value).setValue(0)
        self.main_window.widgets.get_button(Widgets.Buttons.ShowROCGraphs.value).setEnabled(False)
        self.main_window.widgets.get_button(Widgets.Buttons.ShowPRGraphs.value).setEnabled(False)
        self.main_window.widgets. \
            get_label(Widgets.Labels.AfterClassificationStatistics.value).setText(" ")