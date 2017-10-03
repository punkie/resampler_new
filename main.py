import sys
from PySide.QtCore import *
from PySide.QtGui import *
from sample import Ui_MainWindow
from functions import choose_dataset, choose_outputdir

class ControlMainWindow(QMainWindow):

    def __init__(self, parent=None):
        super(ControlMainWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    mySW = ControlMainWindow()
    mySW.findChild(QPushButton, "datasetButton").clicked.connect(lambda : choose_dataset(mySW))
    mySW.findChild(QPushButton, "outputDirectoryButton").clicked.connect(lambda: choose_outputdir(mySW))
    mySW.show()
    sys.exit(app.exec_())
