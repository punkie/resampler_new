import sys
import traceback
import numpy as np
import pandas as pd
from PyQt5 import QtCore
from PyQt5.QtGui import QIcon

from PyQt5.QtWidgets import QMainWindow, QApplication, QDialog, QFileDialog, QVBoxLayout, QLabel, QWidget, QScrollArea, \
    QHBoxLayout, QToolButton, QMenu, QSpacerItem, QSizePolicy
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from mlxtend.plotting import plot_decision_regions

from error_dialog import Ui_Dialog
from functions.ui_qt_slots import choose_dataset, \
    choose_outputdir, perform_resampling, \
    choose_sampling_algorithm, show_img_diffs, classify_datasets, show_roc_graphs, show_pr_graphs, \
    choose_classification_algorithm, show_pca_graph, show_normal_graph, store_selected_k, show_pair_plot_graph, \
    show_pie_chart, clear_graphs
from functions.ui_helping_functions import update_widgets_after_classification, update_widgets_after_successful_datasetload
# from generated_pyqt_ui import Ui_MainWindow
from generate_pyqt_ui_v2 import Ui_DataResamplingTools
from rs_types.classification_algorithms import ClassificationAlgorithms
from rs_types.resampling_methods import ResamplingAlgorithms
from rs_types.state import BasicState
from rs_types.widgets import Widgets
from matplotlib import pyplot as plt

QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)


class MplCanvas(FigureCanvasQTAgg):

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)

def do_sample_test(self):
    vboxLayout = self.findChild(QVBoxLayout, "verticalLayout_14")
    hLayout = QHBoxLayout()
    testLabel = QLabel("test")
    testLabel.setMinimumHeight(50)
    testLabel.setMaximumHeight(50)
    hLayout.addWidget(testLabel)
    sc = MplCanvas(self, width=5, height=4, dpi=100)
    sc.axes.plot([0, 1, 2, 3, 4], [10, 1, 20, 3, 40])
    hLayout.addWidget(sc)
    vboxLayout.addLayout(hLayout)



class MainWindow(QMainWindow):

    def __init__(self):
        super(MainWindow, self).__init__()
        self.ui = Ui_DataResamplingTools()
        self.ui.setupUi(self)
        self.state = BasicState()
        self.widgets = Widgets(self)
        # default algorithms
        self.state.sampling_algorithm_data_tab = ResamplingAlgorithms.RO
        self.state.sampling_algorithm_experiments_tab = ResamplingAlgorithms.RO
        self.state.classification_algorithm = ClassificationAlgorithms.CART
        # self.state.number_of_folds = 10

    def update_dataset_load_progress_bar(self, value):
        self.widgets.get_progress_bar(Widgets.ProgressBars.DatasetProgressBar.value).setValue(value)

    def update_normal_classify_progress_bar(self, value):
        self.widgets.get_progress_bar(Widgets.ProgressBars.NormalClassifyProgressBar.value).setValue(value)

    def update_resample_classify_progress_bar(self, value):
        self.widgets.get_progress_bar(Widgets.ProgressBars.ResampleClassifyProgressBar.value).setValue(value)

    def color_negative_red(val):
        """
        Takes a scalar and returns a string with
        the css property `'color: red'` for negative
        strings, black otherwise.
        """
        color = 'red' if val < 0 else 'black'
        return 'color: %s' % color

    def highlight_max(s):
        '''
        highlight the maximum in a Series yellow.
        '''
        is_max = s == s.max()
        return ['background-color: yellow' if v else '' for v in is_max]

    def round(digit, digit_after_fp):
        if not isinstance(digit, str):
            return ("{:." + str(digit_after_fp) + "f}").format(digit)
        else:
            return digit

    def update_gui_after_classification(self):
        # other = self.state.classified_data_resampled_case['other']
        # X = self.state.classified_data_resampled_case['new_x']
        # f, ax = plt.subplots()
        # ax.contourf(other[0], other[1], other[2], alpha=0.4)
        # ax.scatter(X[:, -1], X[:, 1], alpha=0.8)
        # clff = self.state.classified_data_resampled_case['stored_classifier']
        # X = self.state.classified_data_resampled_case['x_values']
        # y = self.state.classified_data_resampled_case['y_values']
        # plot_decision_regions(X, y, clf=clff, legend=2)
        # plt.show()
        if self.state.resample_classify_thread_finished and self.state.normal_classify_thread_finished:
            self.state.number_of_runs += 1
            # update_widgets_after_classification(self)
            vboxLayout = self.findChild(QVBoxLayout, "verticalLayout_14")
            hLayout = QHBoxLayout()
            testLabel = QLabel("Run #{}".format(self.state.number_of_runs))
            # testLabel.setFixedWidth(30)
            testLabel.setMinimumHeight(350)
            testLabel.setMaximumHeight(350)
            hLayout.addWidget(testLabel)

            a = np.array(['Classifier', 'Sampling Method', 'Balanced Accuracy', 'Precision', 'Recall', 'F1', 'G1', 'G2', 'AUC_roc', 'AUC_pr'])
            b = np.array(range(1, 10))
            c = np.array(range(1, 10))
            class_alg = self.state.classified_data_normal_case['ClassAlg']
            b = [self.state.classified_data_normal_case['ClassAlg']] + ["---"] + [round(self.state.classified_data_normal_case['bal_acc'], 3)] + list(map(lambda x: round(x, 3), self.state.classified_data_normal_case['pre_rec_f1_g_mean1_g_mean2_tuple'])) + [round(self.state.classified_data_normal_case['avg_roc'], 3)] + [round(self.state.classified_data_normal_case['average_precision'], 3)]
            c = [self.state.classified_data_resampled_case['ClassAlg']] + [self.state.classified_data_resampled_case['SamplingAlg']] + [round(self.state.classified_data_resampled_case['bal_acc'], 3)] + list(map(lambda x: round(x, 3), self.state.classified_data_resampled_case['pre_rec_f1_g_mean1_g_mean2_tuple'])) + [round(self.state.classified_data_resampled_case['avg_roc'], 3)] + [round(self.state.classified_data_resampled_case['average_precision'], 3)]
            res = np.vstack((a, b, c)).T

            # tab_2 = [['%.2f' % j for j in i] for i in res]

            fig, ax = plt.subplots(1, 1)
            fig.patch.set_visible(False)

            ax.axis('off')
            ax.axis('tight')
            df = pd.DataFrame(res, columns=[' ', 'Standard Case', 'Re-sampled Case'])
            # df.applymap('{:,.2f}'.format)
            # df.style.applymap(self.color_negative_red, subset=['NormalCase', 'ResampledCase'])
            # df.plot(table=True, ax=ax)
            #fig = ax.get_figure()
            #fig.show()
            #df.style.apply(self.highlight_max, color='darkorange', axis=None)
            # df.style.set_properties(**{'text-align': 'center'})
            ax.table(cellText=df.values, colLabels=df.columns, loc='center')
            # plt.show()

            fig.tight_layout()
            #
            canvas = FigureCanvasQTAgg(fig)
            canvas.setMinimumHeight(350)
            canvas.setMaximumHeight(350)

            # sc = MplCanvas(self, width=5, height=4, dpi=100)
            # sc.axes.plot([0, 1, 2, 3, 4], [10, 1, 20, 3, 40])
            hLayout.addWidget(canvas)

            # ax.table(cellText=df.values, colLabels=df.columns, loc='center')
            #
            # fig.tight_layout()
            #
            # plt.show()




            # sc = MplCanvas(self, width=5, height=4, dpi=100)
            # sc.axes.plot([0, 1, 2, 3, 4], [10, 1, 20, 3, 40])

            classified_data = [self.state.classified_data_normal_case, self.state.classified_data_resampled_case]
            # hLayout.addWidget(Widgets.create_roc_graph(classified_data, None))



            #hLayout.addWidget(Widgets.create_confusion_matrix_graph(classified_data, None))
            # hLayout.addWidget(Widgets.create_roc_graph(classified_data, None))

            # hLayout.addWidget(Widgets.create_pr_graph(classified_data, None))

            hLayout.addWidget(Widgets.create_roc_graph(classified_data, None))
            hLayout.addWidget(Widgets.create_pr_graph(classified_data, None))
            #hLayout.addWidget(Widgets.create_decision_boundary_graph(classified_data, None))


            # hLayout.addWidget(canvas)
            vboxLayout.addLayout(hLayout)



    def update_gui_after_dataset_load(self, value):
        update_widgets_after_successful_datasetload(self, value)


    def reraise_non_mt_exception(self, exception):
        raise exception
        # dialog = QDialog(self)
        # dialog.ui = Ui_Dialog()
        # dialog.ui.setupUi(dialog)
        # dialog.setFocus(True)
        # dialog.show()

    def do_setup(self):
        self.__register_callbacks()
        self.__fill_combo_boxes()
        self.__redefine_exceptions_hook()

    def __register_callbacks(self):
        self.widgets.get_button(Widgets.Buttons.DatasetButton.value).clicked.connect(lambda: choose_dataset(self))
        self.widgets.get_button(Widgets.Buttons.OutputDirectoryButton.value).clicked.connect(
            lambda: choose_outputdir(mw))
        self.widgets.get_button(Widgets.Buttons.StartButton.value).clicked.connect(lambda: perform_resampling(self))
        # self.widgets.get_button(Widgets.Buttons.TestButton.value).clicked.connect(lambda: do_sample_test(self))
        # self.widgets.get_button(Widgets.Buttons.ImgDiffsButton.value).clicked.connect(lambda: show_img_diffs(self))
        self.widgets.get_button(Widgets.Buttons.ClassifyButton.value).clicked.connect(lambda: classify_datasets(self))
        self.widgets.get_combo_box(Widgets.ComboBoxes.ResamplingAlgorithms.value).activated.connect(lambda: choose_sampling_algorithm(self, True))
        self.widgets.get_combo_box(Widgets.ComboBoxes.ResamplingAlgorithmsExperimentsCase.value).activated.connect(lambda: choose_sampling_algorithm(self, False))
        self.widgets.get_combo_box(Widgets.ComboBoxes.ClassificationAlgorithms.value).activated.connect(lambda: choose_classification_algorithm(self))
        self.widgets.get_combo_box(Widgets.ComboBoxes.NumberOfFoldsCV.value).activated.connect(lambda: store_selected_k(self))
        # self.widgets.get_button(Widgets.Buttons.ShowROCGraphs.value).clicked.connect(lambda: show_roc_graphs(self))
        self.widgets.get_button(Widgets.Buttons.StandardGraphNormalDatasetButton.value).clicked.connect(
            lambda: show_normal_graph(self, False))
        self.widgets.get_button(Widgets.Buttons.PairPlotNormalDatasetButton.value).clicked.connect(lambda: show_pair_plot_graph(self, False))
        self.widgets.get_button(Widgets.Buttons.PairPlotResampledDatasetButton.value).clicked.connect(
            lambda: show_pair_plot_graph(self, True))
        self.widgets.get_button(Widgets.Buttons.StandardGraphResampledDatasetButton.value).clicked.connect(
            lambda: show_normal_graph(self, True))
        self.widgets.get_button(Widgets.Buttons.PcaGraphNormalDatasetButton.value).clicked.connect(lambda: show_pca_graph(self, False))
        self.widgets.get_button(Widgets.Buttons.PcaGraphResampledDatasetButton.value).clicked.connect(lambda: show_pca_graph(self, True))
        self.widgets.get_button(Widgets.Buttons.PieChartNormalDatasetButton.value).clicked.connect(lambda: show_pie_chart(self, False))
        self.widgets.get_button(Widgets.Buttons.PieChartResampledDatasetButton.value).clicked.connect(
            lambda: show_pie_chart(self, True))
        self.widgets.get_button(Widgets.Buttons.ClearButton.value).clicked.connect(lambda: clear_graphs(self))
        # self.widgets.get_button(Widgets.Buttons.ShowPRGraphs.value).clicked.connect(lambda: show_pr_graphs(self))

        #self.findChild(QHBoxLayout, "testHorizontalLayout").addWidget(QLabel("simple test"))

        spacerItem = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)
        #self.findChild(QHBoxLayout, "testHorizontalLayout").addItem(spacerItem)
        #spacer.
        self.findChild(QHBoxLayout, "testHorizontalLayout")
        d = {'a': [1, 2, 3], 'b': [4, 5, 6], 'c': [7, 8, 9]}

        # button = self.findChild(QToolButton, "testToolButton")
        #
        # def callback_factory(k, v):
        #     return lambda: button.setText('{0}_{1}'.format(k, v))
        #
        # menu = QMenu()
        # for k, vals in d.items():
        #     sub_menu = menu.addMenu(k)
        #     for v in vals:
        #         action = sub_menu.addAction(str(v))
        #         action.triggered.connect(callback_factory(k, v))
        #
        # button.setMenu(menu)

    def __fill_combo_boxes(self):
        self.widgets.get_combo_box(Widgets.ComboBoxes.ResamplingAlgorithms.value).addItems(
            [ra.value[0] for ra in ResamplingAlgorithms])
        self.widgets.get_combo_box(Widgets.ComboBoxes.ResamplingAlgorithmsExperimentsCase.value).addItems(
            [ra.value[0] for ra in ResamplingAlgorithms])
        self.widgets.get_combo_box(Widgets.ComboBoxes.ClassificationAlgorithms.value).addItems(
            [ca.value[0] for ca in ClassificationAlgorithms])
        self.widgets.get_combo_box(Widgets.ComboBoxes.NumberOfFoldsCV.value).addItems(map(str, range(2,11)))
        self.widgets.get_combo_box(Widgets.ComboBoxes.NumberOfFoldsCV.value).setCurrentIndex(8)


    def __create_error_dialog(self, error_traceback, error_msg):
        dialog = QDialog(self)
        dialog.ui = Ui_Dialog()
        dialog.ui.setupUi(dialog)
        dialog.ui.errorTextPlaceholder.setText("Traceback (most recent call last):\n" +
            ''.join(traceback.format_tb(error_traceback)) + error_msg)

        dialog.setFocus(True)
        dialog.show()

    def __redefine_exceptions_hook(self):
        original_hook = sys.excepthook

        def call_original_exception_hook(exception_type, exception_message, error_traceback):
            self.__create_error_dialog(error_traceback, exception_type.__name__ + " " + str(exception_message))
            original_hook(exception_type, exception_message, error_traceback)
        sys.excepthook = call_original_exception_hook

if __name__ == '__main__':
    app = QApplication([])
    mw = MainWindow()
    mw.setWindowIcon(QIcon('logo.png'))
    mw.do_setup()
    mw.showMaximized()
    mw.show()
    # ds_dialog = QFileDialog(mw)
    # ds_dialog.show()
    # df = pd.DataFrame(np.random.randn(10, 2), columns=['a', 'b'])
    # df.plot(x='a', y='b')
    app.exec_()

