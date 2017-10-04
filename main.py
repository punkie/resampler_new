import sys

from PyQt5.QtWidgets import QMainWindow, QApplication, QPushButton, QProgressBar

from generated_pyqt_ui import Ui_MainWindow
from ui_functions import choose_dataset, choose_outputdir, perform_resampling
from state import BasicState

class ControlMainWindow(QMainWindow):

    def onCountChanged(self, value):
        self.findChild(QProgressBar, "datasetProgressBar").setValue(value)

    def __init__(self):
        super(ControlMainWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.state = BasicState()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    mySW = ControlMainWindow()
    mySW.findChild(QPushButton, "datasetButton").clicked.connect(lambda : choose_dataset(mySW))
    mySW.findChild(QPushButton, "outputDirectoryButton").clicked.connect(lambda: choose_outputdir(mySW))
    mySW.findChild(QPushButton, "startButton").clicked.connect(lambda: perform_resampling(mySW))
    mySW.show()
    sys.exit(app.exec_())
