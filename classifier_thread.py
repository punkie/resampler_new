from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import QLabel
from general_functions import classify


class Classifying(QThread):

    show_plot = pyqtSignal(dict)
    update_classify_progress_bar = pyqtSignal(int)

    def __init__(self, main_window):
        super(Classifying, self).__init__()
        self.main_window = main_window

    def run(self):
        self.main_window.findChild(QLabel, "classifyingStatusLabel").setText("Started classifying. Please wait...")
        classifying_data = classify(self)
        self.show_plot.emit(classifying_data)
        self.main_window.findChild(QLabel, "classifyingStatusLabel").setText("Done classifying!")
