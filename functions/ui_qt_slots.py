import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QFileDialog, QVBoxLayout
from functions.drawing_functions import draw_comparision_picture, draw_roc_graph, draw_pr_graph, draw_pca, \
    draw_standard_graph, draw_pie_chart
from rs_types.classification_algorithms import ClassificationAlgorithms
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
        main_window.dloader.update_dataset_load_progress_bar_signal.connect(main_window.update_dataset_load_progress_bar)
        main_window.dloader.update_gui_after_dataset_load_signal.connect(main_window.update_gui_after_dataset_load)
        main_window.dloader.reraise_non_mt_exception_signal.connect(main_window.reraise_non_mt_exception)
        main_window.dloader.start()


def choose_outputdir(main_window):
    od = QFileDialog(main_window).getExistingDirectory()
    if od != "":
        main_window.widgets.get_label(Widgets.Labels.OutputDirectoryPickedLabel.value).setText(od)
        main_window.state.output_dir = od
        main_window.widgets.get_button(Widgets.Buttons.StartButton.value).setEnabled(True)


def choose_sampling_algorithm(main_window, data_tab):
    if data_tab:
        chosen_algorithm_name = main_window.widgets.\
            get_combo_box(Widgets.ComboBoxes.ResamplingAlgorithms.value).currentText()
        main_window.state.sampling_algorithm_data_tab = ResamplingAlgorithms.get_algorithm_by_name(chosen_algorithm_name)
    else:
        chosen_algorithm_name = main_window.widgets. \
            get_combo_box(Widgets.ComboBoxes.ResamplingAlgorithmsExperimentsCase.value).currentText()
        main_window.state.sampling_algorithm_experiments_tab = ResamplingAlgorithms.get_algorithm_by_name(chosen_algorithm_name)


def store_selected_k(main_window):
    number_of_folds = int (main_window.widgets.get_combo_box(Widgets.ComboBoxes.NumberOfFoldsCV.value).currentText())
    main_window.state.number_of_folds = number_of_folds
    main_window.widgets.get_progress_bar(Widgets.ProgressBars.NormalClassifyProgressBar.value).setMaximum(
        number_of_folds)
    main_window.widgets.get_progress_bar(Widgets.ProgressBars.ResampleClassifyProgressBar.value).setMaximum(
        number_of_folds)
    if main_window.widgets.get_progress_bar(Widgets.ProgressBars.NormalClassifyProgressBar.value).value() is not 0:
        main_window.widgets.get_progress_bar(Widgets.ProgressBars.NormalClassifyProgressBar.value).setValue(
            number_of_folds)
        main_window.widgets.get_progress_bar(Widgets.ProgressBars.ResampleClassifyProgressBar.value).setValue(
            number_of_folds)


def choose_classification_algorithm(main_window):
    chosen_algorithm_name = main_window.widgets. \
        get_combo_box(Widgets.ComboBoxes.ClassificationAlgorithms.value).currentText()
    main_window.state.classification_algorithm = ClassificationAlgorithms.get_algorithm_by_name(chosen_algorithm_name)


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
    main_window.classifier_rd.reraise_non_mt_exception_signal.connect(main_window.reraise_non_mt_exception)
    main_window.classifier_rd.update_resample_classify_progress_bar.connect(main_window.update_resample_classify_progress_bar)
    main_window.classifier_rd.start()
    # print ("test")


def show_roc_graphs(main_window):
    draw_roc_graph(main_window.state)


def show_pair_plot_graph(main_window, is_resampled_case):
    if is_resampled_case:
        dataset = main_window.state.resampled_dataset
    else:
        dataset = main_window.state.dataset
    df = dataset['dataset_as_dataframe']
    df.columns = main_window.state.dataset['header_row']
    negative_tc, positive_tc = main_window.state.dataset['y_values_as_set']
    df.loc[df['Y'] == negative_tc, ['Y']] = 'Negative class'
    df.loc[df['Y'] == positive_tc, ['Y']] = 'Positive class'
    pp = sns.pairplot(df, hue="Y", diag_kind="kde", palette={'Negative class': 'blue', 'Positive class': 'red'}, size=1);
    handles = pp._legend_data.values()
    labels = pp._legend_data.keys()
    del pp.fig.legends[0]
    pp.fig.legend(handles=handles, labels=labels, loc='lower center', ncol=2)
    pp.fig.subplots_adjust(top=0.92, bottom=0.085)
    pp.fig.show()

def show_normal_graph(main_window, is_resampled_case):
    draw_standard_graph(main_window, is_resampled_case)

def show_pca_graph(main_window, is_resampled_case):
    draw_pca(main_window, is_resampled_case)

def show_pie_chart(main_window, is_resampled_case):
    draw_pie_chart(main_window, is_resampled_case)

def show_pr_graphs(main_window):
    draw_pr_graph(main_window.state)

def clear_graphs(main_window):
    main_v_layout = main_window.findChild(QVBoxLayout, "verticalLayout_14")
    for i in reversed(range(main_v_layout.count())):
        clear_layout(main_v_layout.itemAt(i))

def clear_layout(layout):
    for i in reversed(range(layout.count())):
        layout.itemAt(i).widget().setParent(None)
#
# def toggle_widgets(widgets_to_be_blocked, widgets_to_be_enabled):
#     for w in widgets_to_be_blocked:
#         w.setEnabled(False)
#     for w in widgets_to_be_enabled:
#         w.setEnabled(True)

