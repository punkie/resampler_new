from PyQt5.QtWidgets import QFileDialog, QLabel, QComboBox, QPushButton, QProgressBar
from resampling_functions import do_sampling
from dataset_loader import DatasetLoader
from resampling_methods import ResamplingAlgorithms


def choose_dataset(main_window):
    ds_dialog = QFileDialog(main_window)
    ds_dialog.show()
    if ds_dialog.exec_():
        try:
            file_paths = ds_dialog.selectedFiles()
            main_window.dloader = DatasetLoader(main_window, file_paths[0])
            main_window.dloader.update_progress_bar.connect(main_window.update_progress_bar)
            main_window.dloader.update_dataset.connect(main_window.update_dataset)
            main_window.dloader.start()
        except Exception as e:
            print (e)


def choose_outputdir(main_window):
    od = QFileDialog(main_window).getExistingDirectory()
    if od != "":
        main_window.findChild(QLabel, "outputDirectoryPickedLabel").setText(od)
        main_window.state.output_dir = od
        main_window.findChild(QPushButton, "startButton").setEnabled(True)


def choose_sampling_algorithm(main_window):
    chosen_algorithm_name = main_window.findChild(QComboBox, "resamplingAlgorithms").currentText()
    main_window.state.sampling_algorithm = ResamplingAlgorithms.get_algorithm_by_name(chosen_algorithm_name)


def perform_resampling(main_window):
    do_sampling(main_window.state)