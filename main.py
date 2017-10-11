import sys
import matplotlib.pyplot as plt

from PyQt5.QtWidgets import QMainWindow, QApplication, QPushButton, QProgressBar, QComboBox
from generated_pyqt_ui import Ui_MainWindow
from resampling_methods import ResamplingAlgorithms
from ui_functions import choose_dataset, \
    choose_outputdir, perform_resampling, choose_sampling_algorithm, show_img_diffs, classify_datasets
from state import BasicState


class MainWindow(QMainWindow):

    def update_dataset_load_progress_bar(self, value):
        self.findChild(QProgressBar, "datasetProgressBar").setValue(value)

    def update_classify_progress_bar(self, value):
        self.findChild(QProgressBar, "classifyProgressBar").setValue(value)

    def show_plot(self, classifying_data):
        mean_fpr, mean_tpr, mean_auc, std_auc, tprs_lower, tprs_upper = classifying_data['mean_values_tuple']
        for mt in classifying_data['main_tuples']:
            fpr, tpr, roc_auc, i = mt
            plt.figure(1)
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
        # default algorithm
        self.state.sampling_algorithm = ResamplingAlgorithms.RO

if __name__ == '__main__':
    app = QApplication(sys.argv)
    mw = MainWindow()
    mw.findChild(QPushButton, "datasetButton").clicked.connect(lambda : choose_dataset(mw))
    mw.findChild(QPushButton, "outputDirectoryButton").clicked.connect(lambda: choose_outputdir(mw))
    mw.findChild(QPushButton, "startButton").clicked.connect(lambda: perform_resampling(mw))
    mw.findChild(QComboBox, "resamplingAlgorithms").activated.connect(lambda: choose_sampling_algorithm(mw))
    mw.findChild(QPushButton, "imgDiffsButton").clicked.connect(lambda: show_img_diffs(mw))
    mw.findChild(QPushButton, "classifyButton").clicked.connect(lambda: classify_datasets(mw))
    # load default items
    mw.findChild(QComboBox, "resamplingAlgorithms").addItems([ra.value[0] for ra in ResamplingAlgorithms])
    mw.show()
    sys.exit(app.exec_())
