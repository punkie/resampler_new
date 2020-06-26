import os
from cProfile import Profile
from pstats import Stats

import numpy as np
import operator
import pandas as pd

# os.listdir('../datasets')
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.metrics import specificity_score, classification_report_imbalanced
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler, ClusterCentroids, TomekLinks, NearMiss, \
    CondensedNearestNeighbour, OneSidedSelection, EditedNearestNeighbours, NeighbourhoodCleaningRule, \
    InstanceHardnessThreshold
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_curve, precision_recall_fscore_support, auc
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import tree

from algos.smote_boost import SMOTEBoost


def gather_sampling_algos():
    algos = []
    algos.append(RandomOverSampler(random_state=1))
    algos.append(SMOTE(random_state=1))
    algos.append(ADASYN(random_state=1))
    # algos.append()
    # algos.append()
    # algos.append()
    algos.append(RandomUnderSampler(random_state=1))
    # algos.append(ClusterCentroids(random_state=1))
    # algos.append(TomekLinks())
    # algos.append(NearMiss(version=1))
    # algos.append(NearMiss(version=2))
    # algos.append(NearMiss(version=3))
    # algos.append(CondensedNearestNeighbour(random_state=1))
    # algos.append(OneSidedSelection(random_state=1))
    # algos.append(EditedNearestNeighbours())
    # algos.append(NeighbourhoodCleaningRule())
    # algos.append(InstanceHardnessThreshold())
    # algos.append(SMOTEENN(random_state=1))
    # algos.append(SMOTETomek(random_state=1))
    algos.append(SMOTEBoost())

    # algos.append([])
    return algos


def gather_class_algos():
    algos = []
    algos.append(tree.DecisionTreeClassifier(criterion='entropy', random_state=np.random.RandomState(1)))
    # algos.append(RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=np.random.RandomState(1)))
    # algos.append(svm.SVC(probability=True, random_state=np.random.RandomState(1)))
    return algos


def large_func():
    dataset_files = os.listdir('../datasets/cl_datasets')
    rand_state = np.random.RandomState(1)
    cv = StratifiedKFold(n_splits=10, random_state=rand_state)
    for d_fp in dataset_files:
        with open("../result_rs_cl_tool/" + d_fp + "-rs-cl-res.txt", "w") as d_res_file:
            with open("../latex-gen/" + d_fp + "-latex-tables.txt", "w") as latex_file:

                #     first_row = pd.read_csv(d_fp, sep=',', nrows=1)
                # header_row = has_header(first_row)
                tfr = pd.read_csv('../datasets/cl_datasets/' + d_fp, sep=',', iterator=True, header=0)
                ds_as_dataframe = pd.concat(tfr)
                dataset = dict()
                columns_length = len(ds_as_dataframe.columns)
                # if header_row:
                    # dataset['header_row'] = first_row.columns.to_numpy().flatten()
                # else:
                #     dataset['header_row'] = np.array(['X_{}'.format(i) for i in range(columns_length - 1)] + ['Y'])
                # dataset['dataset_as_dataframe'] = ds_as_dataframe
                x_values = ds_as_dataframe.iloc[:, :columns_length - 1].to_numpy()
                y_values = ds_as_dataframe.iloc[:, columns_length - 1:].to_numpy().flatten()
                d_res_file.write("Basic information for the dataset:\n")
                d_res_file.write("Pos examples:\n")
                d_res_file.write("Neg examples:\n")
                d_res_file.write("Imbalanced ratio:\n")

                sampling_algos = gather_sampling_algos()
                class_algos = gather_class_algos()
                # used_algos = [tree.DecisionTreeClassifier(criterion='entropy', random_state=np.random.RandomState(1))]
                d_res_file.write("********************************************************************************\n");
                for sa in sampling_algos:
                    d_res_file.write("Sampling_algorithm:" + str(sa) + "\n")
                    for alg in class_algos:
                        d_res_file.write("Classification algorithm:" + str(alg) + "\n")
                        i = 0
                        pr_rec_f1s = []
                        g_means_1 = []
                        g_means_2 = []
                        roc_aucs = []
                        pr_aucs = []
                        for train, test in cv.split(x_values, y_values):
                            d_res_file.write("Fold number: {}\n".format(i))
                            if (str(type(sa).__name__) == 'SMOTEBoost'):
                                model = sa.fit(x_resampled_values, y_resampled_values)
                            else:
                                x_resampled_values, y_resampled_values = sa.fit_sample(x_values[train], y_values[train])
                                model = alg.fit(x_resampled_values, y_resampled_values)
                            predicted_classes = model.predict(x_values[test])
                            probas = model.predict_proba(x_values[test])
                            average_precision = average_precision_score(y_values[test], probas[:, 1])
                            pr_aucs.append(average_precision)
                            fpr, tpr, thresholds = roc_curve(y_values[test], probas[:, 1])
                            prf1 = precision_recall_fscore_support(y_values[test], predicted_classes, average='binary')
                            pr_rec_f1s.append(prf1)
                            report_from_imbalanced_learning = classification_report_imbalanced(y_values[test],
                                                                                               predicted_classes)
                            d_res_file.write("This is the report from the imbalanced_learning module:\n")
                            d_res_file.write(report_from_imbalanced_learning + "\n")
                            # d_res_file.write("Precision: {}\n".format(tpr))
                            d_res_file.write("Precision: {}\n".format(prf1[0]))
                            d_res_file.write("Recall: {}\n".format(prf1[1]))
                            d_res_file.write("F_measure: {}\n".format(prf1[2]))
                            specificity = specificity_score(y_values[test], predicted_classes)
                            # g_means_1.append(np.sqrt(prf1[1] * specificity))
                            g_mean_1 = np.sqrt(prf1[1] * specificity)
                            g_mean_2 = prf1[0] * prf1[1]
                            g_means_1.append(g_mean_1)
                            g_means_2.append(g_mean_2)
                            d_res_file.write("G_mean_1: {}\n".format(g_mean_1))
                            d_res_file.write("G_mean_2: {}\n".format(g_mean_2))
                            roc_auc = auc(fpr, tpr)
                            roc_aucs.append(roc_auc)
                            d_res_file.write("AUC_ROC: {}\n".format(roc_auc))
                            d_res_file.write("AUC_PR: {}\n".format(average_precision))
                            # print (average_precision)
                            # print(fpr, tpr, thresholds)
                            i += 1
                        d_res_file.write("\n\n")
                        # d_res_file.write("Average statistics all folds:\n")
                        # d_res_file.write("Precision: {}\n".format(tpr))
                        # d_res_file.write("Recall: {}\n")
                        # d_res_file.write("F_measure: {}\n")
                        # d_res_file.write("G_mean_1: {}\n")
                        # d_res_file.write("G_mean_2: {}\n")
                        # d_res_file.write("AUC_ROC: {}\n")
                        # d_res_file.write("AUC_PR: {}\n")

                        # classifying_data['pre_rec_f1_g_mean1_g_mean2_tuple'] = (
                        # (sum(map(np.operator.itemgetter(0), pr_rec_f1s)) / i),
                        # (sum(map(operator.itemgetter(1), pr_rec_f1s)) / i),
                        # (sum(map(operator.itemgetter(2), pr_rec_f1s)) / i),
                        # sum(g_means_1) / i,
                        # sum(g_means_2) / i)

                        avg_pre, avg_rec, avg_f1, avg_g_mean, avg_g_mean_2, avg_roc_auc, avg_pr_auc = (
                        (sum(map(operator.itemgetter(0), pr_rec_f1s)) / i),
                        (sum(map(operator.itemgetter(1), pr_rec_f1s)) / i),
                        (sum(map(operator.itemgetter(2), pr_rec_f1s)) / i),
                        sum(g_means_1) / i,
                        sum(g_means_2) / i, sum(roc_aucs) / i,  sum(pr_aucs) / i)
                        d_res_file.write("----------------------------------------------------------\n")
                        d_res_file.write("Average results (for all folds)\n")
                        d_res_file.write("Dataset: {}\n".format(d_fp))
                        d_res_file.write("Sampling algorithm: {}\n".format(str(type(sa).__name__)))
                        d_res_file.write("Classification algorithm: {}".format(type(alg).__name__))
                        d_res_file.write("Avg Precision: {}\n".format(avg_pre))
                        d_res_file.write("Avg Recall: {}\n".format(avg_rec))
                        d_res_file.write("Avg F_measure: {}\n".format(avg_f1))
                        d_res_file.write("Avg G_mean_1: {}\n".format(avg_g_mean))
                        d_res_file.write("Avg G_mean_2: {}\n".format(avg_g_mean_2))
                        d_res_file.write("Avg AUC_ROC: {}\n".format(avg_roc_auc))
                        d_res_file.write("Avg AUC_PR: {}\n".format(avg_pr_auc))
                        d_res_file.write("----------------------------------------------------------\n")

                        d_res_file.write("\n\n")

                        ds_name = d_fp.split(".")[0]
                        # latex_file.write("Results for {} dataset and {} algorithm".format(ds_name, type(alg).__name__))
                        latex_file.write("\\begin{longtable}{|p{1.5cm}|p{1cm}|p{1cm}|p{1cm}|p{1cm}|p{1cm}|p{1cm}|p{1cm}|p{1cm}|p{1cm}|}\n")

                        latex_file.write("Latex statistics for dataset {} with sm {} and algo {}\n".format(d_fp, type(sa).__name__,
                            type(alg).__name__))

                        latex_file.write("Precision & \multicolumn{1}{c|}{" + "{:.3f}".format(avg_pre) + "}\n")
                        latex_file.write("Recall & \multicolumn{1}{c|}{" + "{:.3f}".format(avg_rec) + "}\n")
                        latex_file.write("F_measure & \multicolumn{1}{c|}{" + "{:.3f}".format(avg_f1) + "}\n")
                        latex_file.write("G_mean_1 & \multicolumn{1}{c|}{" + "{:.3f}".format(avg_g_mean) + "}\n")
                        latex_file.write("G_mean_2 & \multicolumn{1}{c|}{" + "{:.3f}".format(avg_g_mean_2) + "}\n")
                        latex_file.write("AUC_ROC & \multicolumn{1}{c|}{" + "{:.3f}".format(avg_roc_auc) + "}\n")
                        latex_file.write("AUC_PR & \multicolumn{1}{c|}{" + "{:.3f}".format(avg_pr_auc) + "}\n")

                # dataset['name'] = dataset_loader.path.split("/")[-1].split(".csv")[0]
                # print ('asd')

                # with open(dataset_loader.path, newline="") as csv_input_file:
                #     # try:
                #         reader = csv.reader(csv_input_file, delimiter=",")
                #         reader_as_list = list(reader)
                #
                #         dataset['name'] = dataset_loader.path.split("/")[-1].split(".csv")[0]
                #         x_values = deque()
                #         y_values = deque()
                #         rows_count = 0
                #         dataset_loader.main_window.widgets.\
                #             get_progress_bar(Widgets.ProgressBars.DatasetProgressBar.value).setMinimum(0)
                #         dataset_loader.main_window.widgets.\
                #             get_progress_bar(Widgets.ProgressBars.DatasetProgressBar.value).setMaximum(0)
                #         for row in reader_as_list:
                #             # this here is very dangerous. should be refactored.
                #             row_floats = list(map(float, row))
                #             rows_count += 1
                #             x_values.append(row_floats[:-1])
                #             y_values.append(row_floats[-1])
                #             # dataset_loader.update_dataset_load_progress_bar_signal.emit(rows_count)
                #         dataset['x_values'] = np.array(x_values)
                #         dataset['y_values'] = np.array(y_values)
                # dataset_loader.main_window.state.dataset = dataset


def generate_latex_output(dict_with_data):
    dict_with_data = {}
    dict_with_data['d_name'] = "abalone_19.csv"
    dict_with_data['sampling_name'] = 'test_sm'
    dict_with_data['sampling_case'] = "Undersampling"
    dict_with_data['class_alg_name'] = 'test_alg'
    dict_with_data['metrics'] = {'Precision' : ['1', '2', '3', '4', '5', '6', '7', '8', '9']}
    d_fp = "abalone_19.csv"
    # convert_numbers_to_digits_and_for_the_max_make_cell_color_green
    with open("../latex-gen/" + d_fp + "-latex-tables.txt", "w") as latex_file:
        sa = RandomUnderSampler()
        alg = tree.DecisionTreeClassifier(criterion='entropy', random_state=np.random.RandomState(1))
        sa_name = type(sa).__name__
        class_alg_name = type(alg).__name__
        d_fp_for_table = d_fp.replace("_", "\\_")
        precisions = ['1', '2', '3', '4', '5', '6', '7', '8', '9']
        latex_file.write("Latex statistics for dataset {} with sm {} and algo {}\n\n".format(d_fp, sa_name, class_alg_name))
        latex_file.write("\\begin{longtable}{|p{1.5cm}|p{1cm}|p{1cm}|p{1cm}|p{1cm}|p{1cm}|p{1cm}|p{1cm}|p{1cm}|p{1cm}|}\n")
        latex_file.write("\t\hline\n")
        latex_file.write("\t\multicolumn{10}{|c|}{" + dict_with_data['sampling_case'] + " " + dict_with_data['class_alg_name'] + "} \\\\ \n")
        latex_file.write("\t\hline\n")
        if dict_with_data['sampling_case'] == "Undersampling":

            latex_file.write("\t\multicolumn{1}{|c|}{Dataset \& metric} & \multicolumn{1}{c|}{RU} & \multicolumn{1}{c|}{CC} "
             "&\multicolumn{1}{c|}{TL} & \multicolumn{1}{c|}{NM1} & \multicolumn{1}{c|}{NM2} & \multicolumn{1}{c|}{NM3} "
             "& \multicolumn{1}{c|}{CNN} & \multicolumn{1}{c|}{OSS} & \multicolumn{1}{c|}{ENN} \\\\ \n")
            latex_file.write("\t\hline\n")
            latex_file.write(
                "\t\multicolumn{1}{|l|}{\\textit{" + d_fp_for_table.split(".")[0] + "}} & & & & & & & & & \\\\ \n")
        else:
            latex_file.write("\t\multicolumn{1}{|c|}{Dataset \& metric} & \multicolumn{1}{c|}{RO} &"
             " \multicolumn{1}{c|}{SMOTE} & \multicolumn{1}{c|}{ADASYN} &"
             " \multicolumn{1}{c|}{SMOTE+TL} & \multicolumn{1}{c|}{SMOTE+ENN} \\\\ \n")
            latex_file.write("\t\hline\n")
            latex_file.write(
                "\t\multicolumn{1}{|l|}{\\textit{" + d_fp_for_table.split(".")[0] + "}} & & & & & & & & & \\\\ \n")
        latex_file.write("\t\hline\n")
        latex_file.write("\t\multicolumn{1}{|r|}{$Precision$} " + "".join([" & \multicolumn{1}{c|}{" + el + "}" for el in precisions])  + " \\\\ \n")
        latex_file.write("\t\caption{tab:testtab}\n")
        latex_file.write("\t\label{tab:testtab}\n")
        latex_file.write("\end{longtable}\\n")
if __name__ == '__main__':
    generate_latex_output()
    # profiler = Profile()
    # profiler.runcall(large_func())
    # stats = Stats(profiler)
    # stats.strip_dirs()
    # stats.sort_stats('cumulative').stats.print_stats()
    #large_func()