import sys
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QMainWindow, QApplication, QPushButton, QProgressBar, QComboBox
from generated_pyqt_ui import Ui_MainWindow
from resampling_methods import ResamplingAlgorithms
from ui_functions import choose_dataset, \
    choose_outputdir, perform_resampling, choose_sampling_algorithm, show_img_diffs, classify_datasets
from state import BasicState
from widgets import Widgets


class MainWindow(QMainWindow):

    def update_dataset_load_progress_bar(self, value):
        self.widgets.get_progress_bar(Widgets.ProgressBars.DatasetProgressBar.value).setValue(value)

    def update_normal_classify_progress_bar(self, value):
        self.widgets.get_progress_bar(Widgets.ProgressBars.NormalClassifyProgressBar.value).setValue(value)

    def update_resample_classify_progress_bar(self, value):
        self.widgets.get_progress_bar(Widgets.ProgressBars.ResampleClassifyProgressBar.value).setValue(value)

    def show_roc_plot(self, classifying_data):
        mean_fpr, mean_tpr, mean_auc, std_auc, tprs_lower, tprs_upper = classifying_data['mean_values_tuple']
        for mt in classifying_data['main_tuples']:
            fpr, tpr, roc_auc, i = mt
            plt.figure(classifying_data['figure_number'])
            plt.plot(fpr, tpr, lw=1, alpha=0.3,
                label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
                label='Luck', alpha=.8)
        plt.plot(mean_fpr, mean_tpr, color='b',
                 label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
                 lw=2, alpha=.8)
        plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                 label=r'$\pm$ 1 std. dev.')
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        plt.show()

    def __init__(self):
        super(MainWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.state = BasicState()
        self.widgets = Widgets(self)
        # default algorithm
        self.state.sampling_algorithm = ResamplingAlgorithms.RO

    def closeEvent(self, event):
        print ("Closing the app")
        self.deleteLater()

if __name__ == '__main__':
    sys._excepthook = sys.excepthook
    def exception_hook(exctype, value, traceback):
        sys._excepthook(exctype, value, traceback)
        sys.exit(1)
    sys.excepthook = exception_hook

    app = QApplication(sys.argv)
    mw = MainWindow()
    mw.widgets.get_button(Widgets.Buttons.DatasetButton.value).clicked.connect(lambda : choose_dataset(mw))
    mw.widgets.get_button(Widgets.Buttons.OutputDirectoryButton.value).clicked.connect(lambda: choose_outputdir(mw))
    mw.widgets.get_button(Widgets.Buttons.StartButton.value).clicked.connect(lambda: classify_datasets(mw))
    mw.widgets.get_button(Widgets.Buttons.ImgDiffsButton.value).clicked.connect(lambda: show_img_diffs(mw))
    mw.widgets.get_button(Widgets.Buttons.ClassifyButton.value).clicked.connect(lambda: classify_datasets(mw))
    mw.widgets.get_combo_box(Widgets.ComboBoxes.ResamplingAlgorithms.value).activated.connect(lambda: choose_sampling_algorithm(mw))
    # load default items
    mw.widgets.get_combo_box(Widgets.ComboBoxes.ResamplingAlgorithms.value).addItems([ra.value[0] for ra in ResamplingAlgorithms])
    mw.show()
    sys.exit(app.exec_())
