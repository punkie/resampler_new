import sys
from PyQt5.QtWidgets import QMainWindow, QApplication
from functions.ui_callback_functions import choose_dataset, \
    choose_outputdir, perform_resampling, \
    choose_sampling_algorithm, show_img_diffs, classify_datasets, show_roc_graphs, show_pr_graphs, \
    choose_classification_algorithm
from functions.ui_helping_functions import update_widgets_after_classification, update_widgets_after_datasetload
from generated_pyqt_ui import Ui_MainWindow
from rs_types.classification_algorithms import ClassificationAlgorithms
from rs_types.resampling_methods import ResamplingAlgorithms
from rs_types.state import BasicState
from rs_types.widgets import Widgets


class MainWindow(QMainWindow):

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
        update_widgets_after_datasetload(self, value)

    def __init__(self):
        super(MainWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.state = BasicState()
        self.widgets = Widgets(self)
        # default algorithm
        self.state.sampling_algorithm = ResamplingAlgorithms.RO
        self.state.classification_algorithm = ClassificationAlgorithms.CART

if __name__ == '__main__':
    # exceptions hook
    sys._excepthook = sys.excepthook
    def exception_hook(exctype, value, traceback):
        sys._excepthook(exctype, value, traceback)
        sys.exit(1)
    sys.excepthook = exception_hook

    app = QApplication(sys.argv)
    mw = MainWindow()
    mw.widgets.get_button(Widgets.Buttons.DatasetButton.value).clicked.connect(lambda : choose_dataset(mw))
    mw.widgets.get_button(Widgets.Buttons.OutputDirectoryButton.value).clicked.connect(lambda: choose_outputdir(mw))
    mw.widgets.get_button(Widgets.Buttons.StartButton.value).clicked.connect(lambda: perform_resampling(mw))
    mw.widgets.get_button(Widgets.Buttons.ImgDiffsButton.value).clicked.connect(lambda: show_img_diffs(mw))
    mw.widgets.get_button(Widgets.Buttons.ClassifyButton.value).clicked.connect(lambda: classify_datasets(mw))
    mw.widgets.get_combo_box(Widgets.ComboBoxes.ResamplingAlgorithms.value).activated.connect(
        lambda: choose_sampling_algorithm(mw))
    mw.widgets.get_combo_box(Widgets.ComboBoxes.ClassificationAlgorithms.value).activated.connect(
        lambda: choose_classification_algorithm(mw))
    mw.widgets.get_button(Widgets.Buttons.ShowROCGraphs.value).clicked.connect(lambda: show_roc_graphs(mw))
    mw.widgets.get_button(Widgets.Buttons.ShowPRGraphs.value).clicked.connect(lambda: show_pr_graphs(mw))
    # load default items for the different algos
    mw.widgets.get_combo_box(Widgets.ComboBoxes.ResamplingAlgorithms.value).addItems\
            ([ra.value[0] for ra in ResamplingAlgorithms])
    mw.widgets.get_combo_box(Widgets.ComboBoxes.ClassificationAlgorithms.value).addItems \
        ([ca.value[0] for ca in ClassificationAlgorithms])

    mw.show()
    sys.exit(app.exec_())