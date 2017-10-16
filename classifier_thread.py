from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import QLabel
from general_functions import classify
from widgets import Widgets


class Classifying(QThread):

    show_roc_plot = pyqtSignal(dict)
    update_normal_classify_progress_bar = pyqtSignal(int)
    update_resample_classify_progress_bar = pyqtSignal(int)

    def __init__(self, main_window, do_resampling):
        super(Classifying, self).__init__()
        self.main_window = main_window
        self.do_resampling = do_resampling
        if do_resampling:
            self.main_window.widgets.get_progress_bar(Widgets.ProgressBars.ResampleClassifyProgressBar.value).setValue(
                0)
        else:
            self.main_window.widgets.get_progress_bar(Widgets.ProgressBars.NormalClassifyProgressBar.value).setValue(0)
        self.main_window.widgets.get_button(Widgets.Buttons.ShowROCGraphs.value).setEnabled(False)
        self.main_window.widgets.get_button(Widgets.Buttons.ShowPRGraphs.value).setEnabled(False)
    def __custom_post_process(self):
        if self.do_resampling:
            self.main_window.state.resample_classify_thread_finished = True
        else:
            self.main_window.state.normal__classify_thread_finished = True
        if self.main_window.state.resample_classify_thread_finished and self.main_window.state.normal__classify_thread_finished:
            self.main_window.widgets. \
                get_label(Widgets.Labels.ClassifyingStatusLabel.value).setText("Done!")
            self.main_window.widgets.get_button(Widgets.Buttons.ShowROCGraphs.value).setEnabled(True)
            self.main_window.widgets.get_button(Widgets.Buttons.ShowPRGraphs.value).setEnabled(True)


    def run(self):
        self.main_window.widgets.\
            get_label(Widgets.Labels.ClassifyingStatusLabel.value).setText("Started classifying. Please wait...")
        classifying_data = classify(self)
        self.show_roc_plot.emit(classifying_data)
        self.__custom_post_process()


