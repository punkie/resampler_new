from PyQt5.QtWidgets import QFileDialog
from functions.drawing_functions import draw_comparision_picture, draw_roc_graph, draw_pr_graph
from rs_types.resampling_methods import ResamplingAlgorithms
from rs_types.widgets import Widgets
from threads.classifier_thread import Classifying
from threads.dataset_loader_thread import DatasetLoader
from threads.resampling_thread import Resampling


def choose_dataset(main_window):
    ds_dialog = QFileDialog(main_window)
    ds_dialog.show()
    if ds_dialog.exec_():
        file_paths = ds_dialog.selectedFiles()
        main_window.dloader = DatasetLoader(main_window, file_paths[0])
        main_window.dloader.update_dataset_load_progress_bar.connect(main_window.update_dataset_load_progress_bar)
        main_window.dloader.update_gui_after_dataset_load.connect(main_window.update_gui_after_dataset_load)
        main_window.dloader.start()


def choose_outputdir(main_window):
    od = QFileDialog(main_window).getExistingDirectory()
    if od != "":
        main_window.widgets.get_label(Widgets.Labels.OutputDirectoryPickedLabel.value).setText(od)
        main_window.state.output_dir = od
        main_window.widgets.get_button(Widgets.Buttons.StartButton.value).setEnabled(True)


def choose_sampling_algorithm(main_window):
    chosen_algorithm_name = main_window.widgets.get_combo_box(Widgets.ComboBoxes.ResamplingAlgorithms.value).currentText()
    main_window.state.sampling_algorithm = ResamplingAlgorithms.get_algorithm_by_name(chosen_algorithm_name)


def perform_resampling(main_window):
    main_window.resampler = Resampling(main_window)
    main_window.resampler.start()


def show_img_diffs(main_window):
    draw_comparision_picture(main_window.state.dataset, main_window.state.resampled_dataset,
                             main_window.state.sampling_algorithm.value[0])


def classify_datasets(main_window):
    #classify(main_window.state)
    main_window.widgets.get_progress_bar(Widgets.ProgressBars.NormalClassifyProgressBar.value).setValue(0)
    main_window.classifier = Classifying(main_window, False)
    main_window.state.normal_classify_thread_finished = False
    main_window.classifier.update_gui_after_classification.connect(main_window.update_gui_after_classification)
    main_window.classifier.update_normal_classify_progress_bar.connect(main_window.update_normal_classify_progress_bar)
    main_window.classifier.start()

    main_window.widgets.get_progress_bar(Widgets.ProgressBars.ResampleClassifyProgressBar.value).setValue(0)
    main_window.classifier_rd = Classifying(main_window, True)
    main_window.state.resample_classify_thread_finished = False
    main_window.classifier_rd.update_gui_after_classification.connect(main_window.update_gui_after_classification)
    main_window.classifier_rd.update_resample_classify_progress_bar.connect(main_window.update_resample_classify_progress_bar)
    main_window.classifier_rd.start()


def show_roc_graphs(main_window):
    draw_roc_graph(main_window.state)


def show_pr_graphs(main_window):
    draw_pr_graph(main_window.state)

#
# def toggle_widgets(widgets_to_be_blocked, widgets_to_be_enabled):
#     for w in widgets_to_be_blocked:
#         w.setEnabled(False)
#     for w in widgets_to_be_enabled:
#         w.setEnabled(True)

