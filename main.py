import sys

from PyQt5.QtWidgets import QMainWindow, QApplication, QPushButton, QProgressBar, QComboBox
from generated_pyqt_ui import Ui_MainWindow
from resampling_methods import ResamplingAlgorithms
from ui_functions import choose_dataset, choose_outputdir, perform_resampling, choose_sampling_algorithm, \
    show_img_diffs
from state import BasicState


class MainWindow(QMainWindow):

    def update_progress_bar(self, value):
        self.findChild(QProgressBar, "datasetProgressBar").setValue(value)

    def update_dataset(self, dataset):
        self.state.dataset = dataset

    def __init__(self):
        super(MainWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.state = BasicState()
        # default algorithm
        self.state.sampling_algorithm = ResamplingAlgorithms.RO

if __name__ == '__main__':
    app = QApplication(sys.argv)
    mw = MainWindow()
    mw.findChild(QPushButton, "datasetButton").clicked.connect(lambda : choose_dataset(mw))
    mw.findChild(QPushButton, "outputDirectoryButton").clicked.connect(lambda: choose_outputdir(mw))
    mw.findChild(QPushButton, "startButton").clicked.connect(lambda: perform_resampling(mw))
    mw.findChild(QComboBox, "resamplingAlgorithms").activated.connect(lambda: choose_sampling_algorithm(mw))
    mw.findChild(QPushButton, "imgDiffsButton").clicked.connect(lambda: show_img_diffs(mw))
    # load default items
    mw.findChild(QComboBox, "resamplingAlgorithms").addItems([ra.value[0] for ra in ResamplingAlgorithms])
    mw.show()
    sys.exit(app.exec_())
