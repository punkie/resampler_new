import pandas as pd
from PyQt5 import QtGui

import numpy as np
import matplotlib
import seaborn as sns
from PyQt5.QtWidgets import QMainWindow, QDialog, QComboBox, QPushButton, QHBoxLayout, QLabel
from pandas.plotting import scatter_matrix
from window_standard_graph import Ui_Dialog

matplotlib.get_backend()
matplotlib.use("QT5Agg")
from matplotlib import pyplot as plt
from scikitplot.metrics import plot_precision_recall_curve
from sklearn.decomposition import PCA


# class TestWindow(QMainWindow):
#
#     def __init__(self):
#         super(TestWindow, self).__init__()
#         self.ui = Ui_MainWindow()
#         self.ui.setupUi(self)
#         # self.state = BasicState()
#         # self.widgets = Widgets(self)
#         # default algorithms
#         # self.state.sampling_algorithm_data_tab = ResamplingAlgorithms.RO
#         # self.state.sampling_algorithm_experiments_tab = ResamplingAlgorithms.RO
#         # self.state.classification_algorithm = ClassificationAlgorithms.CART


def draw_simple_graph(dataset_tc_set, dataset, header_row, x_combo_box, y_combo_box):
    # QLabel("simple test")
    # w.findChild(QHBoxLayout, "testHorizontalLayout").addChild(QLabel("simple test"))
    # pass
    # pass
    # print ('test')
    negative_tc, positive_tc = dataset_tc_set
    # col_index_x = [idx for idx, el in enumerate(dataset['header_row']) if el == x_combo_box.currentText()][0]
    # col_index_y = [idx for idx, el in enumerate(dataset['header_row']) if el == y_combo_box.currentText()][0]
    # y_combo_box.currentText()
    df = dataset['dataset_as_dataframe']
    df.columns = header_row
    df.loc[df['Y'] == negative_tc, ['Y']] = 'Negative class'
    df.loc[df['Y'] == positive_tc, ['Y']] = 'Positive class'
    # df.pivot_table(columns='Y', aggfunc='size').plot(kind='pie', autopct='%.2f', subplots=True)
    # df.plot.pie(y='Y')
    # sns.set_style("whitegrid")
    # iris = sns.load_dataset("iris")
    # sns.pairplot(df, hue="Y", diag_kind="kde", palette={'Negative class' : 'blue', 'Positive class': 'red'}, size=1)
    # scatter_matrix(df, alpha=0.2, figsize=(6, 6), diagonal='kde')# df = pd.DataFrame(dataset['dataset_as_dataframe'], columns=dataset['header_row'])
    # df = pd.DataFrame(np.random.rand(50, 4), columns=['a', 'b', 'c', 'd'])
    # df.plot.scatter(x='a', y='b',)
    # col = df.Y.map({0: 'b', 1: 'r'})
    # df.plot.legend(log='upper right')
    # df.plot.scatter(x=x_combo_box.currentText(), y=y_combo_box.currentText())
    ax = sns.scatterplot(x=x_combo_box.currentText(), y=y_combo_box.currentText(), hue="Y", palette={'Negative class' : 'blue', 'Positive class': 'red'}, label="", data=df)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles[1:], labels=labels[1:])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.figure.show()


def draw_standard_graph(main_window, is_resampled_case):
    w = QDialog(main_window)
    w.ui = Ui_Dialog()
    w.ui.setupUi(w)
    if is_resampled_case:
        dataset = main_window.state.resampled_dataset
    else:
        dataset = main_window.state.dataset

    dataset_tc_set = dataset['y_values_as_set']
    pd.DataFrame(np.random.randn(10, 2), columns=['a', 'b'])
    header_row = main_window.state.dataset['header_row']

    x_combo_box = w.findChild(QComboBox, "xComboBox")
    y_combo_box = w.findChild(QComboBox, "yComboBox")
    x_combo_box.addItems(main_window.state.dataset['header_row'])
    y_combo_box.addItems(main_window.state.dataset['header_row'])
    w.findChild(QPushButton, "showGraphButton").clicked.connect(lambda: draw_simple_graph(dataset_tc_set, dataset, header_row, x_combo_box, y_combo_box))
    w.setFocus(True)
    w.show()


def draw_pie_chart(main_window, is_resampled_case):
    if is_resampled_case:
        dataset = main_window.state.resampled_dataset
    else:
        dataset = main_window.state.dataset
    # QLabel("simple test")
    # w.findChild(QHBoxLayout, "testHorizontalLayout").addChild(QLabel("simple test"))
    # pass
    # pass
    # print ('test')
    negative_tc, positive_tc = main_window.state.dataset['y_values_as_set']
    # col_index_x = [idx for idx, el in enumerate(dataset['header_row']) if el == x_combo_box.currentText()][0]
    # col_index_y = [idx for idx, el in enumerate(dataset['header_row']) if el == y_combo_box.currentText()][0]
    # y_combo_box.currentText()
    df = dataset['dataset_as_dataframe']
    df.columns = main_window.state.dataset['header_row']
    df.loc[df['Y'] == negative_tc, ['Y']] = 'Negative class'
    df.loc[df['Y'] == positive_tc, ['Y']] = 'Positive class'
    plot = df.pivot_table(columns='Y', aggfunc='size').plot(kind='pie', autopct='%.1f', colors=['blue', 'red'],
                                                            label="", legend=True, subplots=False,
                                                            textprops={'color': "w"})
    plot.legend(loc="upper left")
    # df.plot.pie(y='Y')
    plot.figure.show()

    # canvas = FigureCanvasQTAgg(plot.figure)
    # canvas.setMinimumHeight(350)
    # canvas.setMaximumHeight(350)
    # pieChart.addWidget(canvas)
    # plot.figure.show()

def draw_pca(main_window, is_resampled_dataset):
    if is_resampled_dataset:
        dataset = main_window.state.resampled_dataset
    else:
        dataset = main_window.state.dataset
    pca = PCA(n_components=2)
    # df = pd.DataFrame(np.random.rand(50, 4), columns=['a', 'b', 'c', 'd'])
    # df.plot.scatter(x='a', y='b');
    # g = sns.pairplot(dataset['dataset_as_dataframe'])
    # df = pd.DataFrame(np.random.randn(10, 2), columns=['a', 'b'])
    # df['b'] = pd.Series(list(range(len(df))))
    # df.plot(x='a', y='b')
    # scatter_matrix(df, alpha=0.2, figsize=(6, 6), diagonal='kde')
    x_visible = pca.fit_transform(dataset['x_values'])
    y_values = dataset['y_values']
    negative_tc, positive_tc = main_window.state.dataset['y_values_as_set']
    f, ax = plt.subplots()
    f.set_size_inches(6, 6)
    ax.scatter(x_visible[y_values == negative_tc, 0], x_visible[y_values == negative_tc, 1], label="Negative class",
                     alpha=0.5, color='b')
    ax.scatter(x_visible[y_values == positive_tc, 0], x_visible[y_values == positive_tc, 1], label="Positive class",
               alpha=0.5, color='r')
    if is_resampled_dataset:
        ax.set_title("PCA Re-sampled data with {}".format( main_window.state.sampling_algorithm_data_tab.value[0]))
    else:
        ax.set_title("PCA Original data")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlabel("PCA_X")
    ax.set_ylabel("PCA_Y")
    ax.legend(loc="upper left", prop={'size': 7})
    # ax.get_xaxis().tick_bottom()
    # ax.get_yaxis().tick_left()
    # ax.spines['left'].set_position(('outward', 10))
    # ax.spines['bottom'].set_position(('outward', 10))
    f.show()

def draw_comparision_picture(normal_dataset, resampled_dataset, sampling_algo_name):
    pca = PCA(n_components=2)
    x_visible = pca.fit_transform(normal_dataset['x_values'])
    x_resampled_vis = pca.transform(resampled_dataset['x_values'])
    y_values = normal_dataset['y_values']
    y_values_as_set = normal_dataset['y_values_as_set']
    y_resampled_values = resampled_dataset['y_values']
    tc_one, tc_two = y_values_as_set
    f, (ax1, ax2) = plt.subplots(1, 2)
    f.set_size_inches(12, 9)
    c0 = ax1.scatter(x_visible[y_values == tc_one, 0], x_visible[y_values == tc_one, 1], label="Class #0",
                     alpha=0.5)
    c1 = ax1.scatter(x_visible[y_values == tc_two, 0], x_visible[y_values == tc_two, 1], label="Class #1",
                     alpha=0.5)
    ax1.set_title('Original set')
    ax2.scatter(x_resampled_vis[y_resampled_values == tc_one, 0], x_resampled_vis[y_resampled_values == tc_one, 1],
                label="Class #0", alpha=.5)
    ax2.scatter(x_resampled_vis[y_resampled_values == tc_two, 0], x_resampled_vis[y_resampled_values == tc_two, 1],
                label="Class #1", alpha=.5)
    ax2.set_title(sampling_algo_name)
    for ax in (ax1, ax2):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()
        ax.spines['left'].set_position(('outward', 10))
        ax.spines['bottom'].set_position(('outward', 10))
        ax.set_xlim([-6, 8])
        ax.set_ylim([-6, 6])

    plt.figlegend((c0, c1), ('Class #0', 'Class #1'), loc='lower center',
                  ncol=2, labelspacing=0.)
    plt.tight_layout(pad=3)
    plt.show()


def draw_roc_graph(state):
    classified_data = [state.classified_data_normal_case, state.classified_data_resampled_case]
    f, (ax1, ax2) = plt.subplots(1, 2)
    for x in range(2):
        mean_fpr, mean_tpr, mean_auc, std_auc, tprs_lower, tprs_upper = classified_data[x]['mean_values_tuple']
        ax = ax1 if x == 0 else ax2
        ax.get_figure().set_size_inches(12, 9)
        for mt in classified_data[x]['main_tuples']:
            fpr, tpr, roc_auc, i, y_test_from_normal_ds, predicted_y_scores, average_precision = mt
            #plt.figure(normal_classifying_data['figure_number'])
            ax.plot(fpr, tpr, lw=1, alpha=0.3,
                label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
        ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
                label='Luck', alpha=.8)
        print ("Mean ROC (AUC = %0.2f STD %0.2f)" % (mean_auc, std_auc))
        ax.plot(mean_fpr, mean_tpr, color='b',
                 label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
                 lw=2, alpha=.8)
        ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                 label=r'$\pm$ 1 std. dev.')
        ax.set_xlim([-0.05, 1.05])
        ax.set_ylim([-0.05, 1.05])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.legend(loc="lower right")
        if x == 0:
            ax.set_title('ROC for the normal dataset')
        else:
            ax.set_title('ROC for the resampled dataset\n (using {} algorithm)'
                         .format(state.sampling_algorithm.value[0]))
    #plt.legend(loc="lower right")
    plt.show()


def draw_pr_graph(state):
    classified_data = [state.classified_data_normal_case, state.classified_data_resampled_case]
    f, (ax1, ax2) = plt.subplots(1, 2)
    for x in range(2):
        probas = np.concatenate(classified_data[x]['preds_list'], axis=0)
        y_true = np.concatenate(classified_data[x]['trues_list'])
        ax = ax1 if x == 0 else ax2
        plot_precision_recall_curve(y_true, probas,
            title="Normal dataset with average precision = {0:0.2f}".format(classified_data[x]['average_precision']) if x == 0
    else "Resampled dataset with {0} and\n average precision ={1:0.2f}".format(state.sampling_algorithm.value[0], classified_data[x]['average_precision']),
            curves=('micro', 'each_class'), ax=ax,
            figsize=None, cmap='nipy_spectral',
            title_fontsize="large",
            text_fontsize="medium")
        ax.get_figure().set_size_inches(12, 9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_ylim([0.0, 1.05])
        ax.set_xlim([0.0, 1.0])

    plt.show()
