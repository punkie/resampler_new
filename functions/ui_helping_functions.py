from functions.general_functions import get_mean_precision_recall_f1_scores
from rs_types.widgets import Widgets


def update_widgets_after_classification(main_window):
    main_window.widgets. \
        get_label(Widgets.Labels.ClassifyingStatusLabel.value).setText("Done!")
    main_window.widgets.get_button(Widgets.Buttons.ShowROCGraphs.value).setEnabled(True)
    main_window.widgets.get_button(Widgets.Buttons.ShowPRGraphs.value).setEnabled(True)
    main_window.widgets. \
        get_label(Widgets.Labels.AfterClassificationStatistics.value).setText(
            get_mean_precision_recall_f1_scores(main_window.state))


def update_widgets_after_datasetload(main_window, path):
    main_window.widgets.get_label(Widgets.Labels.DatasetPickedLabel.value).setText(path)
    main_window.widgets. \
        get_label(Widgets.Labels.DatasetStatisticsLabel.value).setText(
            main_window.state.dataset['dataset_statistics_string'])
    main_window.widgets.get_combo_box(Widgets.ComboBoxes.ResamplingAlgorithms.value).setEnabled(True)
    main_window.widgets.get_button(Widgets.Buttons.OutputDirectoryButton.value).setEnabled(True)
    main_window.widgets.get_label(Widgets.Labels.ResampledDatasetStatistics.value).setText(" ")
    main_window.widgets.get_label(Widgets.Labels.ResamplingStatusLabel.value).setText(" ")
    main_window.widgets.get_button(Widgets.Buttons.ImgDiffsButton.value).setEnabled(False)
    main_window.widgets.get_combo_box(Widgets.ComboBoxes.ClassificationAlgorithms.value).setEnabled(True)
    main_window.widgets.get_button(Widgets.Buttons.ClassifyButton.value).setEnabled(True)
    main_window.widgets.get_label(Widgets.Labels.ClassifyingStatusLabel.value).setText(" ")
    main_window.widgets.get_progress_bar(Widgets.ProgressBars.NormalClassifyProgressBar.value).setValue(0)
    main_window.widgets.get_progress_bar(Widgets.ProgressBars.ResampleClassifyProgressBar.value).setValue(0)
