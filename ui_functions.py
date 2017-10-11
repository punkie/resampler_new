from PyQt5.QtWidgets import QFileDialog, QLabel, QComboBox, QPushButton, QProgressBar

from classifier_thread import Classifying
from dataset_loader_thread import DatasetLoader
from general_functions import draw_comparision_picture, classify
from resampling_methods import ResamplingAlgorithms
from resampling_thread import Resampling



def choose_dataset(main_window):
    ds_dialog = QFileDialog(main_window)
    ds_dialog.show()
    if ds_dialog.exec_():
        file_paths = ds_dialog.selectedFiles()
        main_window.dloader = DatasetLoader(main_window, file_paths[0])
        main_window.dloader.update_dataset_load_progress_bar.connect(main_window.update_dataset_load_progress_bar)
        main_window.dloader.start()


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
    main_window.resampler = Resampling(main_window)
    main_window.resampler.start()


def show_img_diffs(main_window):
    draw_comparision_picture(main_window.state.dataset, main_window.state.resampled_dataset,
                             main_window.state.sampling_algorithm.value[0])


def classify_datasets(main_window):
    #classify(main_window.state)
    main_window.findChild(QProgressBar, "classifyProgressBar").setValue(0)
    main_window.classifier = Classifying(main_window)
    main_window.classifier.show_plot.connect(main_window.show_plot)
    main_window.classifier.update_classify_progress_bar.connect(main_window.update_classify_progress_bar)
    main_window.classifier.start()
    main_window.classifier_two = Classifying(main_window, True)
    # main_window.classifier_two.show_plot.connect(main_window.show_plot)
    # main_window.classifier_two.update_classify_progress_bar.connect(main_window.update_classify_progress_bar)
    main_window.classifier_two.start()