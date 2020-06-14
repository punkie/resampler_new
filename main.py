import sys
import traceback
from PyQt5 import QtCore

from PyQt5.QtWidgets import QMainWindow, QApplication, QDialog, QFileDialog
from PyQt5.QtCore import Qt

from error_dialog import Ui_Dialog
from functions.ui_qt_slots import choose_dataset, \
    choose_outputdir, perform_resampling, \
    choose_sampling_algorithm, show_img_diffs, classify_datasets, show_roc_graphs, show_pr_graphs, \
    choose_classification_algorithm
from functions.ui_helping_functions import update_widgets_after_classification, update_widgets_after_successful_datasetload
# from generated_pyqt_ui import Ui_MainWindow
from generate_pyqt_ui_v2 import Ui_DataSamplingTools
from rs_types.classification_algorithms import ClassificationAlgorithms
from rs_types.resampling_methods import ResamplingAlgorithms
from rs_types.state import BasicState
from rs_types.widgets import Widgets

QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)


class MainWindow(QMainWindow):

    def __init__(self):
        super(MainWindow, self).__init__()
        self.ui = Ui_DataSamplingTools()
        self.ui.setupUi(self)
        self.state = BasicState()
        self.widgets = Widgets(self)
        # default algorithms
        self.state.sampling_algorithm = ResamplingAlgorithms.RO
        self.state.classification_algorithm = ClassificationAlgorithms.CART

    def update_dataset_load_progress_bar(self, value):
        self.widgets.get_progress_bar(Widgets.ProgressBars.DatasetProgressBar.value).setValue(value)

    def update_normal_classify_progress_bar(self, value):
        self.widgets.get_progress_bar(Widgets.ProgressBars.NormalClassifyProgressBar.value).setValue(value)

    def update_resample_classify_progress_bar(self, value):
        self.widgets.get_progress_bar(Widgets.ProgressBars.ResampleClassifyProgressBar.value).setValue(value)

    def update_gui_after_classification(self):
        if self.state.resample_classify_thread_finished and self.state.normal_classify_thread_finished:
            update_widgets_after_classification(self)

    def update_gui_after_dataset_load(self, value):
        update_widgets_after_successful_datasetload(self, value)

    def reraise_non_mt_exception(self, exception):
        raise exception
        # dialog = QDialog(self)
        # dialog.ui = Ui_Dialog()
        # dialog.ui.setupUi(dialog)
        # dialog.setFocus(True)
        # dialog.show()

    def do_setup(self):
        self.__register_callbacks()
        self.__fill_combo_boxes()
        self.__redefine_exceptions_hook()

    def __register_callbacks(self):
        self.widgets.get_button(Widgets.Buttons.DatasetButton.value).clicked.connect(lambda: choose_dataset(self))
        self.widgets.get_button(Widgets.Buttons.OutputDirectoryButton.value).clicked.connect(
            lambda: choose_outputdir(mw))
        self.widgets.get_button(Widgets.Buttons.StartButton.value).clicked.connect(lambda: perform_resampling(self))
        # self.widgets.get_button(Widgets.Buttons.ImgDiffsButton.value).clicked.connect(lambda: show_img_diffs(self))
        # self.widgets.get_button(Widgets.Buttons.ClassifyButton.value).clicked.connect(lambda: classify_datasets(self))
        self.widgets.get_combo_box(Widgets.ComboBoxes.ResamplingAlgorithms.value).activated.connect(
            lambda: choose_sampling_algorithm(self))
        # self.widgets.get_combo_box(Widgets.ComboBoxes.ClassificationAlgorithms.value).activated.connect(
        #     lambda: choose_classification_algorithm(self))
        # self.widgets.get_button(Widgets.Buttons.ShowROCGraphs.value).clicked.connect(lambda: show_roc_graphs(self))
        # self.widgets.get_button(Widgets.Buttons.ShowPRGraphs.value).clicked.connect(lambda: show_pr_graphs(self))

    def __fill_combo_boxes(self):
        self.widgets.get_combo_box(Widgets.ComboBoxes.ResamplingAlgorithms.value).addItems(
            [ra.value[0] for ra in ResamplingAlgorithms])
        # self.widgets.get_combo_box(Widgets.ComboBoxes.ClassificationAlgorithms.value).addItems(
        #     [ca.value[0] for ca in ClassificationAlgorithms])

    def __create_error_dialog(self, error_traceback, error_msg):
        dialog = QDialog(self)
        dialog.ui = Ui_Dialog()
        dialog.ui.setupUi(dialog)
        dialog.ui.errorTextPlaceholder.setText("Traceback (most recent call last):\n" +
            ''.join(traceback.format_tb(error_traceback)) + error_msg)

        dialog.setFocus(True)
        dialog.show()

    def __redefine_exceptions_hook(self):
        original_hook = sys.excepthook

        def call_original_exception_hook(exception_type, exception_message, error_traceback):
            self.__create_error_dialog(error_traceback, exception_type.__name__ + " " + str(exception_message))
            original_hook(exception_type, exception_message, error_traceback)
        sys.excepthook = call_original_exception_hook

if __name__ == '__main__':
    app = QApplication([])
    mw = MainWindow()
    mw.do_setup()
    mw.show()
    # ds_dialog = QFileDialog(mw)
    # ds_dialog.show()
    app.exec_()
