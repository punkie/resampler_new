from PyQt4.QtGui import QFileDialog, QLabel, QComboBox, QPushButton


def choose_dataset(main_window):
    ds_dialog = QFileDialog(main_window)
    ds_dialog.show()
    if ds_dialog.exec_():
        fileNames = ds_dialog.selectedFiles()
        main_window.findChild(QLabel, "datasetPickedLabel").setText(fileNames[0])
        main_window.findChild(QComboBox, "resamplingAlgorithms").setEnabled(True)
        main_window.findChild(QPushButton, "outputDirectoryButton").setEnabled(True)
def choose_outputdir(main_window):
    od = QFileDialog(main_window).getExistingDirectory()
    #od_dialog.show()
    if od != "":
        #fileNames = od_dialog.selectedFiles()
        main_window.findChild(QLabel, "outputDirectoryPickedLabel").setText(od)
        main_window.findChild(QPushButton, "startButton").setEnabled(True)
    #outputDirectoryPickedLabel
