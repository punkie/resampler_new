from PyQt5.QtWidgets import QFileDialog, QLabel, QComboBox, QPushButton, QProgressBar
from resampling_functions import do_random_oversampling
from general_functions import load_dataset
from dataset_loader import DatasetLoader

def choose_dataset(main_window):
    main_window.dloader = DatasetLoader()
    main_window.dloader.countChanged.connect(main_window.onCountChanged)
    main_window.dloader.start()
    # ds_dialog = QFileDialog(main_window)
    # ds_dialog.show()
    # if ds_dialog.exec_():
    #     try:
    #         file_paths = ds_dialog.selectedFiles()
    #         main_window.state.dataset = load_dataset(file_paths[0])
    #         main_window.findChild(QLabel, "datasetPickedLabel").setText(file_paths[0])
    #         main_window.findChild(QComboBox, "resamplingAlgorithms").setEnabled(True)
    #         main_window.findChild(QPushButton, "outputDirectoryButton").setEnabled(True)
    #     except Exception as e:
    #         print (e)

def choose_outputdir(main_window):
    od = QFileDialog(main_window).getExistingDirectory()
    #od_dialog.show()
    if od != "":
        #fileNames = od_dialog.selectedFiles()
        main_window.findChild(QLabel, "outputDirectoryPickedLabel").setText(od)
        main_window.state.output_dir = od
        main_window.findChild(QPushButton, "startButton").setEnabled(True)
    #outputDirectoryPickedLabel



def perform_resampling(main_window):
    dir = main_window.state.output_dir
    do_random_oversampling(main_window.state.dataset, dir)