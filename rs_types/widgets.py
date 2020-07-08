import copy
import matplotlib
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from pandas import np
from scikitplot.metrics import plot_precision_recall_curve
from sklearn.metrics import average_precision_score, precision_recall_curve

matplotlib.get_backend()
matplotlib.use("QT5Agg")
import matplotlib.pyplot as plt
from enum import Enum
from PyQt5.QtWidgets import QPushButton, QComboBox, QLabel, QProgressBar, QScrollArea, QTableWidget


# kinda custom enum...
def show_figure(event):
    fig_cpy = copy.deepcopy(event.canvas.figure)
    fig_cpy.show()

class Widgets:
    def __init__(self, main_window):
        self.buttons = {button.value: main_window.findChild(QPushButton, button.value) for button in Widgets.Buttons}
        self.combo_boxes = {combo_box.value: main_window.findChild(QComboBox, combo_box.value)
                            for combo_box in Widgets.ComboBoxes}
        self.labels = {label.value: main_window.findChild(QLabel, label.value) for label in Widgets.Labels}
        self.progress_bars = {progress_bar.value: main_window.findChild(QProgressBar, progress_bar.value)
                              for progress_bar in Widgets.ProgressBars}
        self.scroll_areas = {scroll_area.value: main_window.findChild(QScrollArea, scroll_area.value)
                             for scroll_area in Widgets.ScrollAreas}
        self.tables = {table.value: main_window.findChild(QTableWidget, table.value) for table in Widgets.Tables}

    class Buttons(Enum):
        DatasetButton = "datasetButton"
        OutputDirectoryButton = "outputDirectoryButton"
        StartButton = "startButton"
        ImgDiffsButton = "imgDiffsButton"
        ClassifyButton = "classifyButton"
        ShowROCGraphs = "showROCGraphs"
        ShowPRGraphs = "showPRGraphs"
        TestButton = "testButton"
        StandardGraphNormalDatasetButton = "standardGraphNormalDataset"
        PairPlotNormalDatasetButton = "pairPlotNormalDataset"
        PairPlotResampledDatasetButton = "pairPlotResampledDataset"
        StandardGraphResampledDatasetButton = "standardGraphResampledDataset"
        PcaGraphNormalDatasetButton = "pcaGraphNormalDataset"
        PcaGraphResampledDatasetButton = "pcaGraphResampledDataset"
        PieChartNormalDatasetButton = "pieChartNormalDataset"
        PieChartResampledDatasetButton = "pieChartResampledDataset"
        ClearButton = "clearGraphs"


    class ComboBoxes(Enum):
        ResamplingAlgorithms = "resamplingAlgorithms"
        ResamplingAlgorithmsExperimentsCase = "resamplingAlgorithmsExperimentsCase"
        ClassificationAlgorithms = "classAlgorithms"
        NumberOfFoldsCV = "numberOfFoldsCV"

    class Labels(Enum):
        DatasetPickedLabel = "datasetPickedLabel"
        DatasetStatisticsLabel = "datasetStatisticsLabel"
        ResamplingStatusLabel = "resamplingStatusLabel"
        ClassifyingStatusLabel = "classifyingStatusLabel"
        ResampledDatasetStatistics = "resampledDatasetStatistics"
        OutputDirectoryPickedLabel = "outputDirectoryPickedLabel"
        AfterClassificationStatistics = "afterClassificationStatistics"
        ClassificationAlgorithmLabel = "classAlgorithmLabel"
        FilePathLabel = "filePathLabel"
        DatasetLoadingResultLabel = "datasetLoadingResultLabel"
        DatasetLoadingTextLabel = "datasetLoadingTextLabel"
        TotalNumberOfExamplesLabel = "totalNumberOfExamplesLabel"
        NumberOfPositiveExamplesLabel = "numberOfPositivexamplesLabel"
        TargetClassPercentageLabel = "targetClassPercentageLabel"
        ImbalancedRatioLabel = "imbalancedRatioLabel"
        TotalNumberOfExamplesResampledLabel = "totalNumberOfExamplesResampledLabel"
        NumberOfPositiveExamplesResampledLabel = "numberOfPositiveExamplesResampledLabel"
        TargetClassPercentageResampledLabel = "targetClassPercentageResampledLabel"
        ImbalancedRatioResampledLabel = "imbalancedRatioResampledLabel"
        SelectedDatasetExperimentsTabLabel = "selectedDatasetExperimentsTab"

    class ProgressBars(Enum):
        DatasetProgressBar = "datasetProgressBar"
        ResampleProgressBar = "resampleProgressBar"
        NormalClassifyProgressBar = "normalClassifyProgressBar"
        ResampleClassifyProgressBar = "resampleClassifyProgressBar"

    class ScrollAreas(Enum):
        AfterClassificationStatisticsArea = "afterClassificationStatisticsArea"

    class Tables(Enum):
        DataTable = "dataTableWidget"

    def get_button(self, widget_id):
        return self.buttons[widget_id]

    def get_combo_box(self, widget_id):
        return self.combo_boxes[widget_id]

    def get_label(self, widget_id):
        return self.labels[widget_id]

    def get_progress_bar(self, widget_id):
        return self.progress_bars[widget_id]

    def get_scroll_area(self, widget_id):
        return self.scroll_areas[widget_id]

    def get_table(self, widget_id):
        return self.tables[widget_id]

    @staticmethod
    def create_confusion_matrix_graph(classified_data, sampling_algorithm):
        return QLabel('cm')

    @staticmethod
    def create_roc_graph(classified_data, sampling_algorithm):
        f, ax = plt.subplots()
        tuple_index = 1 if sampling_algorithm is not None else 0
        mean_fpr, mean_tpr, mean_auc, std_auc, tprs_lower, tprs_upper = classified_data[0]['mean_values_tuple']
        re_mean_fpr, re_mean_tpr, re_mean_auc, re_std_auc, re_tprs_lower, re_tprs_upper = classified_data[1]['mean_values_tuple']
        ax.get_figure().set_size_inches(5, 5)
        # for mt in classified_data[tuple_index]['main_tuples']:
        #     fpr, tpr, roc_auc, i, y_test_from_normal_ds, predicted_y_scores, average_precision = mt
        #     # plt.figure(normal_classifying_data['figure_number'])
        #     ax.plot(fpr, tpr, lw=1, alpha=0.3,
        #             label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
        ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
                label='Random guess', alpha=.8)
        print("Mean ROC (Standard) (AUC = %0.2f STD %0.2f)" % (mean_auc, std_auc))
        ax.plot(mean_fpr, mean_tpr, color='b',
                label=r'Mean ROC (Standard case) (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
                lw=2, alpha=.8)
        ax.plot(re_mean_fpr, re_mean_tpr, color='g',
                label=r'Mean ROC (Resampled case) (AUC = %0.2f $\pm$ %0.2f)' % (re_mean_auc, re_std_auc),
                lw=2, alpha=.8)
        # ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
        #                 label=r'$\pm$ 1 std. dev.')
        ax.set_xlim([-0.05, 1.05])
        ax.set_ylim([-0.05, 1.05])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.legend(loc="lower right", prop={'size': 7})
        ax.set_title('ROC chart')
        ax.xaxis.labelpad = -0.5
        # if sampling_algorithm is not None:
        #     ax.set_title('ROC for the resampled dataset\n (using {} algorithm)'.format(sampling_algorithm))
        # else:
        #     ax.set_title('ROC for the standard classification')

            # plt.legend(loc="lower right")

        # f.show()
        canvas = FigureCanvasQTAgg(f)
        canvas.setMinimumHeight(350)
        canvas.setMaximumHeight(350)
        # canvas.mouseDoubleClickEvent()
        # f.canvas.mpl_connect('button_press_event', show_figure)
        return canvas

    @staticmethod
    def create_pr_graph(classified_data, sampling_algorithm):
        # tuple_index = 1 if sampling_algorithm is not None else 0
        f, ax = plt.subplots()
        nml_y_true = np.concatenate(classified_data[0]['trues_list'])
        nml_probas = np.concatenate(classified_data[0]['preds_list'])
        resampled_y_true = np.concatenate(classified_data[1]['trues_list'])
        resampled_probas = np.concatenate(classified_data[1]['preds_list'])
        pr, re,  _ = precision_recall_curve(nml_y_true, nml_probas[:, 1])
        resam_pr, resam_re, _ = precision_recall_curve(resampled_y_true, resampled_probas[:, 1])
        avg_pre_normal_case = average_precision_score(nml_y_true, nml_probas[:, 1])
        # avg_pre_re_case = average_precision_score(classified_data[1]['trues_list'],
        #                                                      classified_data[1]['preds_list'])
        # ax.plot(re, pr, color='b', label='avg is', lw=2, alpha=.8)
        # ax.plot(avg_pre_re_case, color='g', label='qwe', lw=2, alpha=.8)
        #probas = np.concatenate(classified_data[tuple_index]['preds_list'], axis=0)
        #
        # re_probas = np.concatenate(classified_data[1]['preds_list'], axis=0)
        # re_y_true = np.concatenate(classified_data[1]['trues_list'])
        # plot_precision_recall_curve(nml_y_true, nml_probas,
        #                             title="Standard classification with average precision = {0:0.2f}".format(
        #                                 classified_data[0]['average_precision']),
        #                             curves=('micro'), ax=ax,
        #                             figsize=None, cmap='Reds',
        #                             title_fontsize="large",
        #                             text_fontsize="medium")
        # plot_precision_recall_curve(resampled_y_true, resampled_probas,
        #                             title="Resampled classification with average precision = {0:0.2f}".format(
        #                                 classified_data[1]['average_precision']),
        #                             curves=('micro'), ax=ax,
        #                             figsize=None, cmap='Reds',
        #                             title_fontsize="large",
        #                             text_fontsize="medium")
        # plot_precision_recall_curve(re_y_true, re_probas,
        #                             title="Normal dataset with average precision = {0:0.2f}".format(
        #                                 classified_data[tuple_index]['average_precision']) if tuple_index == 1
        #                             else "Resampled dataset with {0} and\n average precision ={1:0.2f}".format(
        #                                 sampling_algorithm, classified_data[tuple_index]['average_precision']),
        #                             curves=('micro'), ax=ax,
        #                             figsize=None, cmap='YlGnBu',
        #                             title_fontsize="large",
        #                             text_fontsize="medium")
        ax.get_figure().set_size_inches(5, 5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_title('PR chart')
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_ylim([0.0, 1.05])
        ax.set_xlim([0.0, 1.0])
        ax.plot(pr, re, color='b', label="Standard case (AUC = {:.2f})".format(classified_data[0]['average_precision']))
        ax.plot(resam_re, resam_pr, color='g', label="Re-sampled case (AUC = {:.2f})".format(classified_data[1]['average_precision']))
        ax.legend(loc="upper right", prop={'size': 7})
        ax.xaxis.labelpad = -0.5
        # plt.figure()
        # plt.step(re, pr, where='post')
        #
        # plt.xlabel('Recall')
        # plt.ylabel('Precision')
        # plt.ylim([0.0, 1.05])
        # plt.xlim([0.0, 1.0])
        # plt.title(
        #     'Average precision score, micro-averaged over all classes: AP={0:0.2f}'
        #         .format(123.3))
        # plt.show()
        canvas = FigureCanvasQTAgg(f)
        canvas.setMinimumHeight(350)
        canvas.setMaximumHeight(350)
        return canvas

    @staticmethod
    def create_decision_boundary_graph(classified_data, sampling_algorithm):
        return QLabel('db')