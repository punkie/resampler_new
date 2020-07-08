from PyQt5.QtWidgets import QTableWidgetItem

from functions.general_functions import get_mean_precision_recall_f1_scores
from rs_types.widgets import Widgets


def update_widgets_after_classification(main_window):
    pass
    #main_window.widgets. \
    #    get_label(Widgets.Labels.ClassifyingStatusLabel.value).setText("Done!")
    #main_window.widgets.get_button(Widgets.Buttons.ShowROCGraphs.value).setEnabled(True)
    #main_window.widgets.get_button(Widgets.Buttons.ShowPRGraphs.value).setEnabled(True)
    # main_window.widgets. \
    #     get_label(Widgets.Labels.AfterClassificationStatistics.value).setText(
    #         get_mean_precision_recall_f1_scores(main_window.state))


def update_widgets_after_successful_datasetload(main_window, path):
    main_window.widgets.get_label(Widgets.Labels.DatasetPickedLabel.value).setText(path)
    # main_window.widgets. \
    #     get_label(Widgets.Labels.DatasetStatisticsLabel.value).setText(
    #         main_window.state.dataset['dataset_statistics_string'])
    main_window.setEnabled(True)
    main_window.widgets.get_button(Widgets.Buttons.OutputDirectoryButton.value).setEnabled(True)
    main_window.widgets.get_label(Widgets.Labels.FilePathLabel.value).setText("FilePath:")
    main_window.widgets.get_label(Widgets.Labels.DatasetLoadingResultLabel.value).setText("Status:  Successful load!")
    # main_window.widgets.get_table(Widgets.Tables.DataTable.value).setEnabled(True);
    dataset = main_window.state.dataset
    main_window.widgets.get_progress_bar(Widgets.ProgressBars.DatasetProgressBar.value).setValue(100)
    main_window.widgets.get_label(Widgets.Labels.TotalNumberOfExamplesLabel.value).setText(str(dataset['number_of_examples']))
    main_window.widgets.get_label(Widgets.Labels.NumberOfPositiveExamplesLabel.value).setText(str(dataset['number_of_positive_examples']))
    main_window.widgets.get_label(Widgets.Labels.TargetClassPercentageLabel.value).setText(str(dataset['positive_examples_percentage']))
    main_window.widgets.get_label(Widgets.Labels.ImbalancedRatioLabel.value).setText(str(dataset['imbalanced_ratio']))
    main_window.widgets.get_label(Widgets.Labels.SelectedDatasetExperimentsTabLabel.value).setText(str(dataset['name']))
    # main_window.widgets.get_label(Widgets.Labels.ResampledDatasetStatistics.value).setText(" ")
    # main_window.widgets.get_label(Widgets.Labels.ResamplingStatusLabel.value).setText(" ")
    # main_window.widgets.get_button(Widgets.Buttons.ImgDiffsButton.value).setEnabled(False)
    # main_window.widgets.get_combo_box(Widgets.ComboBoxes.ClassificationAlgorithms.value).setEnabled(True)
    # main_window.widgets.get_button(Widgets.Buttons.ClassifyButton.value).setEnabled(True)
    # main_window.widgets.get_label(Widgets.Labels.ClassifyingStatusLabel.value).setText(" ")
    # main_window.widgets.get_progress_bar(Widgets.ProgressBars.NormalClassifyProgressBar.value).setValue(0)
    # main_window.widgets.get_progress_bar(Widgets.ProgressBars.ResampleClassifyProgressBar.value).setValue(0)
    load_table(main_window)


def update_widgets_after_unsuccessful_datasetload(main_window):
    #main_window.widgets.get_label(Widgets.Labels.FilePathLabel.value).setText("FilePath")
    main_window.widgets.get_label(Widgets.Labels.DatasetLoadingResultLabel.value).setText("Unsuccessful load!")


def load_table(main_window):
    dataset = main_window.state.dataset
    rows_count = len(dataset['x_values'])
    rows_count = rows_count if rows_count < 10000 else 10000
    col_count = len(dataset['x_values'][0]) + 1
    data_table = main_window.widgets.get_table(Widgets.Tables.DataTable.value)
    data_table.setRowCount(rows_count)
    data_table.setColumnCount(col_count)
    for idx, el in enumerate(dataset['header_row']):
        data_table.setHorizontalHeaderItem(idx, QTableWidgetItem(el))
    for row in range(0, rows_count):
        for col in range(col_count):
            if col == col_count - 1:
                data_table.setItem(row, col, QTableWidgetItem(str(dataset['y_values'][row])))
            else:
                data_table.setItem(row, col, QTableWidgetItem(str(dataset['x_values'][row][col])))
    # data_table.setItem(0, 0, QTableWidgetItem("Cell (1,1)"))
    # data_table.setItem(0, 1, QTableWidgetItem("Cell (1,2)"))
    # data_table.setItem(1, 0, QTableWidgetItem("Cell (2,1)"))
    # data_table.setItem(1, 1, QTableWidgetItem("Cell (2,2)"))
    # data_table.setItem(2, 0, QTableWidgetItem("Cell (3,1)"))
    # data_table.setItem(2, 1, QTableWidgetItem("Cell (3,2)"))
    # data_table.setItem(3, 0, QTableWidgetItem("Cell (4,1)"))
    # data_table.setItem(3, 1, QTableWidgetItem("Cell (4,2)"))
    # data_table.setItem(4, 0, QTableWidgetItem("Cell (4,1)"))
    # data_table.setItem(4, 1, QTableWidgetItem("Cell (4,2)"))
    # data_table.setItem(5, 0, QTableWidgetItem("Cell (4,1)"))
    # data_table.setItem(5, 1, QTableWidgetItem("Cell (4,2)"))