import os
import copy
from cProfile import Profile
from collections import Counter, defaultdict
from pstats import Stats

import numpy as np
import operator
import pandas as pd


# os.listdir('../datasets')
from os.path import dirname, abspath, join
from scipy import interp
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.metrics import specificity_score, classification_report_imbalanced
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler, ClusterCentroids, TomekLinks, NearMiss, \
    CondensedNearestNeighbour, OneSidedSelection, EditedNearestNeighbours, NeighbourhoodCleaningRule, \
    InstanceHardnessThreshold, RepeatedEditedNearestNeighbours, AllKNN
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_curve, precision_recall_fscore_support, auc
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import tree

import sys
# sys.path.append('G:/python-workspace/resampler/algos')
THIS_DIR = dirname(__file__)
CODE_DIR = abspath(join(THIS_DIR, '..', 'algos'))
sys.path.append(CODE_DIR)

import smote_boost


def over_sampling_algs():
    algs = list()
    algs.append(("No Rs Oversampling case", "No Re-sampling"))
    algs.append((RandomOverSampler(random_state=1), 'RO'))
    algs.append((SMOTE(random_state=1), 'SMOTE'))
    algs.append((ADASYN(random_state=1), 'ADASYN'))
    algs.append((SMOTETomek(random_state=1), 'SMOTE+TL'))
    algs.append((SMOTEENN(random_state=1), 'SMOTE+ENN'))
    algs.append((smote_boost.SMOTEBoost(random_state=1), "SMOTEBoost"))
    return algs


def under_sampling_algs():
    algs = list()
    algs.append(("No Rs Undersampling case", "No Re-sampling"))
    algs.append((RandomUnderSampler(random_state=1), 'RU'))
    algs.append((ClusterCentroids(random_state=1), 'CC'))
    algs.append((TomekLinks(), 'TL'))
    algs.append((NearMiss(version=1), 'NM1'))
    algs.append((NearMiss(version=2), 'NM2'))
    algs.append((NearMiss(version=3), 'NM3'))
    algs.append((CondensedNearestNeighbour(random_state=1), 'CNN'))
    algs.append((OneSidedSelection(random_state=1), 'OSS'))
    algs.append((EditedNearestNeighbours(), 'ENN'))
    algs.append((NeighbourhoodCleaningRule(), 'NCL'))
    algs.append((InstanceHardnessThreshold(random_state=1), 'IHT'))
    algs.append((RepeatedEditedNearestNeighbours(), 'RENN'))
    algs.append((AllKNN(), 'AllKNN'))
    return algs


def is_over_sampling(sampling_alg, list_ovrsampling_algs):
    return type(sampling_alg).__name__ in [type(alg).__name__ for alg in list_ovrsampling_algs]


def gather_class_algs():
    algs = []
    algs.append(("CART", tree.DecisionTreeClassifier(criterion='entropy', random_state=np.random.RandomState(1))))
    algs.append(("SVM", svm.SVC(probability=True, random_state=np.random.RandomState(1))))
    # algs.append(RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=np.random.RandomState(1)))
    # algs.append(svm.SVC(probability=True, random_state=np.random.RandomState(1)))
    return algs


def round(digit, digit_after_fp):
    if not isinstance(digit, str):
        return ("{:." + str(digit_after_fp) + "f}").format(digit)
    else:
        return digit

def has_header(dataframe):
    for el in dataframe:
        try:
            if str(el).count(".") == 2:
                continue
            float(el)
        except ValueError:
            return 1
    return None


def reorder_tuple_with_positive_class(tuples, positive_class):
    for idx, t in enumerate(tuples):
        if t[0] == positive_class:
            pos_idx = idx
    other_idx = 0 if pos_idx == 1 else 1
    return [tuples[other_idx], tuples[pos_idx]]


def large_func():
    dataset_files = os.listdir('../datasets/cl_datasets/extreme_imbalance')
    rand_state = np.random.RandomState(1)
    cv = StratifiedKFold(n_splits=10, random_state=rand_state)
    latex_dict = {}
    latex_dict['Under-sampling'] = {}
    latex_dict['Over-sampling+Hybrid'] = {}
    class_algs = gather_class_algs()
    for alg_tuple in class_algs:
        alg_alias = alg_tuple[0]
        latex_dict['Under-sampling'][alg_alias] = {}
        latex_dict['Over-sampling+Hybrid'][alg_alias] = {}
    for d_fp in dataset_files:

        # with open("../result_rs_cl_tool/" + d_fp + "-rs-cl-res.txt", "w") as d_res_file:
            with open("../latex-gen/extreme_imbalance/" + d_fp + "-latex-tables.txt", "w") as latex_file:
                first_row = pd.read_csv('../datasets/cl_datasets/extreme_imbalance/' + d_fp, delimiter=',', nrows=1)
                header_row = has_header(first_row)
                tfr = pd.read_csv('../datasets/cl_datasets/extreme_imbalance/' + d_fp, delimiter=',', iterator=True, header=header_row)
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
                t1_before, t2_before = Counter(y_values).most_common(2)
                # d_res_file.write("Basic information for the dataset:\n")
                # d_res_file.write("Pos examples:\n")
                # d_res_file.write("Neg examples:\n")
                # d_res_file.write("Imbalanced ratio:\n")
                ovr_s_algs = over_sampling_algs()
                undr_s_algs = under_sampling_algs()
                sampling_algs = undr_s_algs + ovr_s_algs

                # used_algs = [tree.DecisionTreeClassifier(criterion='entropy', random_state=np.random.RandomState(1))]
                # d_res_file.write("********************************************************************************\n");
                for alg_tuple in class_algs:
                    alg_alias = alg_tuple[0]
                    alg = copy.deepcopy(alg_tuple[1])
                    latex_dict['Under-sampling'][alg_alias][d_fp] = {}
                    latex_dict['Over-sampling+Hybrid'][alg_alias][d_fp] = {}
                    # d_res_file.write("Classification algorithm:" + alg_alias + "\n")
                    for sa in sampling_algs:
                        alg = copy.deepcopy(alg_tuple[1])
                        # d_res_file.write("Sampling_algorithm:" + sa[1] + "\n")
                        i = 0
                        pr_rec_f1s = []
                        g_means_1 = []
                        g_means_2 = []
                        roc_aucs = []
                        pr_aucs = []
                        bal_accs = []
                        sa_name = sa[1]
                        percentage_pos_samples = []

                        for train, test in cv.split(x_values, y_values):
                            # y_values[train] = y_values[train].astype(int)
                            # d_res_file.write("Fold number: {}\n".format(i))

                            if sa_name == 'SMOTEBoost':
                                model = sa[0].fit(x_values[train], y_values[train].astype(int))
                                percentage_pos_samples.append(model.percentage_pos_examples)
                            elif sa_name == 'No Re-sampling':
                                model = alg.fit(x_values[train], y_values[train].astype(int))
                                percentage_pos_samples.append((t2_before[1] / len(y_values) * 100))
                            else:
                                x_resampled_values, y_resampled_values = sa[0].fit_resample(x_values[train], y_values[train].astype(int))
                                t1_after, t2_after = reorder_tuple_with_positive_class(Counter(y_resampled_values).most_common(2), t2_before[0])
                                percentage_pos_samples.append((t2_after[1] / len(y_resampled_values) * 100))
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
                            # d_res_file.write("This is the report from the imbalanced_learning module:\n")
                            # d_res_file.write(report_from_imbalanced_learning + "\n")
                            # d_res_file.write("Precision: {}\n".format(tpr))
                            # d_res_file.write("Precision: {}\n".format(prf1[0]))
                            # d_res_file.write("Recall: {}\n".format(prf1[1]))
                            # d_res_file.write("F_measure: {}\n".format(prf1[2]))
                            specificity = specificity_score(y_values[test], predicted_classes, average='binary')
                            # g_means_1.append(np.sqrt(prf1[1] * specificity))
                            bal_accuracy = (prf1[1] + specificity) / 2
                            bal_accs.append(bal_accuracy)
                            g_mean_1 = np.sqrt(prf1[1] * specificity)
                            g_mean_2 = np.sqrt(prf1[0] * prf1[1])
                            g_means_1.append(g_mean_1)
                            g_means_2.append(g_mean_2)
                            # d_res_file.write("G_mean_1: {}\n".format(g_mean_1))
                            # d_res_file.write("G_mean_2: {}\n".format(g_mean_2))
                            roc_auc = auc(fpr, tpr)
                            roc_aucs.append(roc_auc)
                            # d_res_file.write("AUC_ROC: {}\n".format(roc_auc))
                            # d_res_file.write("AUC_PR: {}\n".format(average_precision))
                            # print (average_precision)
                            # print(fpr, tpr, thresholds)
                            i += 1
                        # d_res_file.write("\n\n")
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

                        avg_percent_pos, avg_bal_acc, avg_pre, avg_rec, avg_f1, avg_g_mean, avg_g_mean_2, avg_roc_auc, avg_pr_auc = (sum(percentage_pos_samples) / i, sum(bal_accs) / i,
                        (sum(map(operator.itemgetter(0), pr_rec_f1s)) / i),
                        (sum(map(operator.itemgetter(1), pr_rec_f1s)) / i),
                        (sum(map(operator.itemgetter(2), pr_rec_f1s)) / i),
                        sum(g_means_1) / i,
                        sum(g_means_2) / i, sum(roc_aucs) / i,  sum(pr_aucs) / i)
                        # d_res_file.write("----------------------------------------------------------\n")
                        # d_res_file.write("Average results (for all folds)\n")
                        # d_res_file.write("Dataset: {}\n".format(d_fp))
                        # d_res_file.write("Sampling algorithm: {}\n".format(str(type(sa).__name__)))
                        # d_res_file.write("Classification algorithm: {}".format(alg_alias))
                        # d_res_file.write("Balanced Accuracy: {}\n".format(avg_bal_acc))
                        # d_res_file.write("Avg Precision: {}\n".format(avg_pre))
                        # d_res_file.write("Avg Recall: {}\n".format(avg_rec))
                        # d_res_file.write("Avg F_measure: {}\n".format(avg_f1))
                        # d_res_file.write("Avg G_mean_1: {}\n".format(avg_g_mean))
                        # d_res_file.write("Avg G_mean_2: {}\n".format(avg_g_mean_2))
                        # d_res_file.write("Avg AUC_ROC: {}\n".format(avg_roc_auc))
                        # d_res_file.write("Avg AUC_PR: {}\n".format(avg_pr_auc))
                        # d_res_file.write("----------------------------------------------------------\n")

                        # d_res_file.write("\n\n")

                        if sa in ovr_s_algs:
                            latex_dict['Over-sampling+Hybrid'][alg_alias][d_fp][sa_name] = [round(avg_percent_pos, 1), round(avg_bal_acc, 3), round(avg_pre, 3), round(avg_rec, 3), round(avg_f1, 3), round(avg_g_mean, 3),
                            round(avg_g_mean_2, 3), round(avg_roc_auc, 3), round(avg_pr_auc, 3)]
                        else:
                            latex_dict['Under-sampling'][alg_alias][d_fp][sa_name] = [round(avg_percent_pos, 1), round(avg_bal_acc, 3), round(avg_pre, 3), round(avg_rec, 3), round(avg_f1, 3), round(avg_g_mean, 3),
                            round(avg_g_mean_2, 3), round(avg_roc_auc, 3), round(avg_pr_auc, 3)]
                        # print (latex_dict)
                        # ds_name = d_fp.split(".")[0]
                        # latex_file.write("Results for {} dataset and {} algorithm".format(ds_name, type(alg).__name__))

                        # latex_file.write("\\begin{longtable}{|p{1.5cm}|p{1cm}|p{1cm}|p{1cm}|p{1cm}|p{1cm}|p{1cm}|p{1cm}|p{1cm}|p{1cm}|}\n")
                        #
                        # latex_file.write("Latex statistics for dataset {} with sm {} and algo {}\n".format(d_fp, type(sa).__name__,
                        #     type(alg).__name__))
                        #
                        # latex_file.write("Precision & \multicolumn{1}{c|}{" + "{:.3f}".format(avg_pre) + "}\n")
                        # latex_file.write("Recall & \multicolumn{1}{c|}{" + "{:.3f}".format(avg_rec) + "}\n")
                        # latex_file.write("F_measure & \multicolumn{1}{c|}{" + "{:.3f}".format(avg_f1) + "}\n")
                        # latex_file.write("G_mean_1 & \multicolumn{1}{c|}{" + "{:.3f}".format(avg_g_mean) + "}\n")
                        # latex_file.write("G_mean_2 & \multicolumn{1}{c|}{" + "{:.3f}".format(avg_g_mean_2) + "}\n")
                        # latex_file.write("AUC_ROC & \multicolumn{1}{c|}{" + "{:.3f}".format(avg_roc_auc) + "}\n")
                        # latex_file.write("AUC_PR & \multicolumn{1}{c|}{" + "{:.3f}".format(avg_pr_auc) + "}\n")

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
    latex_dict
    generate_latex_output(latex_dict)


# def classify():
#     classifying_data = {}
#     classifying_data['main_tuples'] = []
#     # normal_dataset = classifier_thread.main_window.state.dataset
#     #resampled_dataset = state.resampled_dataset
#     rand_state = np.random.RandomState(1)
#     other_rand_state = np.random.RandomState(0)
#     splits = 10
#     cv = StratifiedKFold(n_splits=splits, random_state=rand_state)
#     # classifier = RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=rand_state)
#     # classifier = svm.SVC(probability=True, random_state=rand_state)
#     # classifier = copy.deepcopy(classifier_thread.main_window.state.classification_algorithm.value[1])
#
#
#     first_row = pd.read_csv('../datasets/cl_datasets/' + 'abalone_19.csv', delimiter=',', nrows=1)
#     header_row = has_header(first_row)
#     tfr = pd.read_csv('../datasets/cl_datasets/' + 'abalone_19.csv', delimiter=',', iterator=True, header=header_row)
#     ds_as_dataframe = pd.concat(tfr)
#     dataset = dict()
#     columns_length = len(ds_as_dataframe.columns)
#     # if header_row:
#     # dataset['header_row'] = first_row.columns.to_numpy().flatten()
#     # else:
#     #     dataset['header_row'] = np.array(['X_{}'.format(i) for i in range(columns_length - 1)] + ['Y'])
#     # dataset['dataset_as_dataframe'] = ds_as_dataframe
#     x_values = ds_as_dataframe.iloc[:, :columns_length - 1].to_numpy()
#     y_values = ds_as_dataframe.iloc[:, columns_length - 1:].to_numpy().flatten()
#
#
#
#     X_normal = x_values
#     y_normal = y_values
#
#     tprs = []
#     mean_fpr = np.linspace(0, 1, 100)
#     roc_aucs = []
#     pr_aucs = []
#     i = 0
#     g_means_1 = []
#     g_means_2 = []
#     pr_rec_f1s = []
#     preds_list = []
#     trues_list = []
#     bal_accs = []
#     average_precisions = []
#     RO = (RandomOverSampler(random_state=1), 'RO')
#     classifier = ("CART (Decision Tree Classifier)", tree.DecisionTreeClassifier(criterion='entropy', random_state=np.random.RandomState(1)))
#     # f, ax = plt.subplots()
#     for train, test in cv.split(X_normal, y_normal):
#         x_resampled_values, y_resampled_values = RO[0].fit_resample(X_normal[train], y_normal[train].astype(int))
#         r_dataset = dict()
#         r_dataset['x_values'] = x_resampled_values
#         r_dataset['y_values'] = y_resampled_values
#         classifier_ = classifier[1].fit(r_dataset['x_values'], r_dataset['y_values'])
#         #predicted_y_scores = classifier_.decision_function(X_normal[test])
#         predicted_classes = classifier_.predict(X_normal[test])
#         probas_ = classifier_.predict_proba(X_normal[test])
#         # if classifier_thread.do_resampling:
#             # plot_step = 0.02
#             # pca = PCA(n_components=2)
#             # X = pca.fit_transform(r_dataset['x_values'])
#             # x_min, x_max = X[:, -1].min() - 1, X[:, -1].max() + 1
#             # y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
#             # xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
#             #                      np.arange(y_min, y_max, plot_step))
#             # pca_class = classifier.fit(X, r_dataset['y_values'])
#             # Z = pca_class.predict(np.c_[xx.ravel(), yy.ravel()])
#             # Z = Z.reshape(xx.shape)
#             # ax.contourf(xx, yy, Z, alpha=0.4)
#             # ax.contourf(r_dataset['x_values'], r_dataset['y_values'], Z, alpha=0.4)
#             # classifying_data['new_x'] = X
#             # classifying_data['other'] = (xx, yy, Z)
#             # classifying_data['stored_classifier'] = classifier_
#             # return classifying_data
#         # else:
#         #     break
#         preds_list.append(probas_)
#         trues_list.append(y_normal[test])
#         average_precision = average_precision_score(y_normal[test], probas_[:, 1])
#         average_precisions.append(average_precision)
#         fpr, tpr, thresholds = roc_curve(y_normal[test], probas_[:, 1])
#         prf1 = precision_recall_fscore_support(y_normal[test], predicted_classes, average='binary')
#         pr_rec_f1s.append(prf1)
#         tprs.append(interp(mean_fpr, fpr, tpr))
#         tprs[-1][0] = 0.0
#         roc_auc = auc(fpr, tpr)
#         roc_aucs.append(roc_auc)
#         report_from_imbalanced_learning = classification_report_imbalanced(y_normal[test], predicted_classes)
#         specificity = specificity_score(y_normal[test], predicted_classes)
#         g_means_1.append(np.sqrt(prf1[1] * specificity))
#         #pr_rec_f1s.append(g_mean1)
#         g_means_2.append(np.sqrt(prf1[0] * prf1[1]))
#         bal_accuracy = (prf1[1] + specificity) / 2
#         bal_accs.append(bal_accuracy)
#
#         # pr_rec_f1s.append(g_mean2)
#         #pr_rec_f1s.append(prf1)
#
#         # print("iteration #{} with resampled dataset ({})."
#         #    " statistics:\n{}\n".format(i, classifier_thread.do_resampling,
#         #                                (classification_report_imbalanced(y_normal[test], predicted_classes))))
#         classifying_data['main_tuples'].append((fpr, tpr, roc_auc, i, y_normal[test], probas_[:, 1], average_precision))
#
#         i += 1
#         # if classifier_thread.do_resampling:
#         #     classifier_thread.update_resample_classify_progress_bar.emit(i)
#         # else:
#         #     classifier_thread.update_normal_classify_progress_bar.emit(i)
#     #grouped_pr_rec_f1s = zip(*pr_rec_f1s)
#     classifying_data['pre_rec_f1_g_mean1_g_mean2_tuple'] = ((sum(map(operator.itemgetter(0), pr_rec_f1s)) / i),
#                                             (sum(map(operator.itemgetter(1), pr_rec_f1s)) / i),
#     (sum(map(operator.itemgetter(2), pr_rec_f1s)) / i),
#     sum(g_means_1) / i,
#     sum(g_means_2) / i)
#     classifying_data['precision'] = list(map(operator.itemgetter(0), pr_rec_f1s))
#     classifying_data['recall'] = list(map(operator.itemgetter(1), pr_rec_f1s))
#     # \
#     #     ,
#     #
#     # (sum(pr_rec_f1s[1]) / i),
#     # (sum(pr_rec_f1s[2]) / i))
#     classifying_data['bal_acc'] = sum(bal_accs) / i
#     classifying_data['preds_list'] = preds_list
#     classifying_data['trues_list'] = trues_list
#     classifying_data['avg_roc'] = sum(roc_aucs) / i
#     classifying_data['average_precision'] = sum(average_precisions) / i
#     # sorted_recall = sorted((map(operator.itemgetter(1), pr_rec_f1s)))
#     # recalls = []
#     # max_precisions = []
#     # for rec in np.linspace(0, 1, num=11):
#     #     filtered_recalls = list(filter(lambda x: x[1] >= rec, pr_rec_f1s))
#     #     max_precision = max(map(operator.itemgetter(0), filtered_recalls)) if len(filtered_recalls) > 0 else 0
#     #     recalls.append(rec)
#     #     max_precisions.append(max_precision)
#     # classifying_data['recalls'] = recalls
#     # classifying_data['max_precisions'] =  max_precisions
#     # classifying_data['sorted_recall'] = sorted_recall
#     # classifying_data['precision'] = list(map(operator.itemgetter(0), pr_rec_f1s))
#     #
#     # classifying_data['decreasing_max_precision'] = list(
#     #     np.maximum.accumulate(classifying_data['precision'][::-1])[::-1])
#
#     mean_tpr = np.mean(tprs, axis=0)
#     mean_tpr[-1] = 1.0
#     mean_auc = auc(mean_fpr, mean_tpr)
#     std_auc = np.std(roc_aucs)
#
#     std_tpr = np.std(tprs, axis=0)
#     tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
#     tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
#     classifying_data['mean_values_tuple'] = (mean_fpr, mean_tpr, mean_auc, std_auc, tprs_lower, tprs_upper)
#     return classifying_data


def append_multicolumn(value):
    return " & \multicolumn{1}{c|}{" + str(value) + "}"


def pre_table_content():
    content = r"\documentclass[12pt,oneside]{report}" + "\n" \
              r"\usepackage[a4paper, left=2cm, right=2cm, top=2.5cm, bottom=2.5cm]{geometry}" + "\n" \
              r"\usepackage[table]{xcolor}\usepackage{fancyhdr}\pagestyle{fancy}" + "\n" \
              r"\usepackage[T2A]{fontenc}\usepackage[english]{babel}" + "\n" \
              r"\usepackage[utf8]{inputenc}" + "\n" \
              r"\usepackage{longtable}" + "\n" \
              r"\usepackage{amssymb}" + "\n" \
              r"\usepackage{amsmath}" + "\n" \
              r"\usepackage{color}" + "\n" \
              r"\usepackage{caption}\captionsetup[table]{name=Таблица}\definecolor{lightgray}{gray}{0.9}" + "\n" \
              r"\fancyhead{}" + "\n" \
              r"\fancyhead[RO,LE]{Методи за работа с дебалансирани множества от данни в машинно самообучение}" + "\n" \
              r"\fancyfoot{}" + "\n" \
              r"\fancyfoot[C]{\thepage}" + "\n" \
              r"\begin{document}" + "\n"
    return content


def constant_factory(value):
    return lambda: value


def get_imbalance_degree_dict():
    d = defaultdict(constant_factory("дебалансирани множества от данни"))
    d['abalone_19.csv'] = "есктремно дебалансирани множества от данни"
    d['poker-8_vs_6.csv'] = "есктремно дебалансирани множества от данни"
    d['ecoli.csv'] = "силно дебалансирани множества от данни"
    d['mammography.csv'] = "силно дебалансирани множества от данни"
    d['ozone_level.csv'] = "силно дебалансирани множества от данни"
    d['pen_digits.csv'] = "силно дебалансирани множества от данни"
    d['glass0.csv'] = "слабо дебалансирани множества от данни"
    d['vehicle2.csv'] = "слабо дебалансирани множества от данни"
    d['yeast1.csv'] = "слабо дебалансирани множества от данни"
    return d


def init_nested_dicts(parent_dict):
    for i in range(1, 9):
        if i not in parent_dict.keys():
            parent_dict[i] = defaultdict(dict)
            parent_dict[i]['NormalCase'] = defaultdict(dict)
            parent_dict[i]['ResampledCase'] = defaultdict(dict)


def transform_metric_idx(idx):
    if idx == 1: return "BA"
    elif idx == 2: return "PR"
    elif idx == 3:
        return "RE"
    elif idx == 4:
        return "F_{1}"
    elif idx == 5:
        return "G_{1}"
    elif idx == 6:
        return "G_{2}"
    elif idx == 7:
        return "AUC_{ROC}"
    elif idx == 8:
        return "AUC_{PR}"


def transform_idx(case, idx):
    # if idx >= 8:
    #     raise Exception("Not Accepted")
    # else:
    idx += 1
    if case == 'Under-sampling':
        return under_sampling_algs()[idx][1]
    else:
        return over_sampling_algs()[idx][1]

def generate_latex_output(dict_with_data):
    # dict_with_data = {}
    # dict_with_data['d_names'] = ["abalone_19.csv", "testme.csv"]
    # dict_with_data['sampling_name'] = 'testSm'
    # dict_with_data['sampling_case'] = "Undersampling"
    # dict_with_data['class_alg_name'] = 'testAlg'
    # dict_with_data['abalone_19.csv'] = {'NM' : []}
    # dict_with_data['testSm'] = {'Precision' : ['1', '2', '3', '4', '5', '6', '7', '8', '9']}
    # d_fp = "abalone_19.csv"
    # convert_numbers_to_digits_and_for_the_max_make_cell_color_green
    im_degree_dict = get_imbalance_degree_dict()

    best_results_dict = defaultdict(dict)
    #init_nested_dicts(best_results_dict)
    pre_table_inited = False
    chapter_inited = False
    for sampling_version in dict_with_data.keys():
        with open("../latex-gen/extreme_imbalance/" + sampling_version + "-latex-tables.txt", "w+", encoding="utf8") as latex_file:
            if not pre_table_inited:
                latex_file.write(pre_table_content())

                pre_table_inited = True

            for class_alg in dict_with_data[sampling_version].keys():
                for d_name in dict_with_data[sampling_version][class_alg].keys():
                    d_fp_for_table = d_name.replace("_", "\\_")
                    init_nested_dicts(best_results_dict[d_fp_for_table])
            for class_alg in dict_with_data[sampling_version].keys():
                first_d_name = list(dict_with_data[sampling_version][class_alg].keys())[0]
                if not chapter_inited:
                    latex_file.write(
                    "\\chapter*{" + sampling_version + " резултати за " + im_degree_dict[first_d_name] + "}\n")
                    chapter_inited = True
                #sa = RandomUnderSampler()
                #alg = tree.DecisionTreeClassifier(criterion='entropy', random_state=np.random.RandomState(1))
                #sa_name = type(sa).__name__
                #class_alg_name = type(alg).__name__
                #precisions = ['1', '2', '3', '4', '5', '6', '7', '8', '9']
                #latex_file.write("Latex statistics for sm {} and algo {}\n\n".format(sa_name, class_alg_name))
                # latex_file.write(pre_table_content())
                latex_file.write("\\begin{longtable}{|m{1.5cm}|m{1cm}|m{1cm}|m{1cm}|m{1cm}|m{1cm}|m{1cm}|m{1cm}|m{1cm}}\n")
                latex_file.write("\t\hline\n")
                latex_file.write("\t\multicolumn{9}{|c|}{" + sampling_version + " results " + class_alg + "} \\\\ \n")
                latex_file.write("\t\hline\n")
                latex_file.write("\t\multicolumn{1}{|c|}{DS \& SM (\\%Pos. ex.)}& \multicolumn{8}{c|}{Metrics} \\\\ \n")
                latex_file.write("\t\hline\n")
                latex_file.write("& \multicolumn{1}{c|}{$BA$} & \multicolumn{1}{c|}{$PR$} & \multicolumn{1}{c|}{$RE$} & \multicolumn{1}{c|}{$F_1$} & \multicolumn{1}{c|}{$G_1$} & \multicolumn{1}{c|}{$G_2$} & \multicolumn{1}{c|}{$AUC_{ROC}$} & \multicolumn{1}{c|}{$AUC_{PR}$} \\\\ \n")
                latex_file.write("\t\hline\n")
                for d_name in dict_with_data[sampling_version][class_alg].keys():
                    d_fp_for_table = d_name.replace("_", "\\_")
                    # best_results_dict[d_fp_for_table]['BalancedAccuracy']['NS'] = ['1', '4']
                    # best_results_dict[d_fp_for_table]['BalancedAccuracy']['SC']['1.35']['Sampling Algs'] = ['..']
                    # best_results_dict[d_fp_for_table]['BalancedAccuracy']['SC']['1.35']['Class Algs'] = ['...']
                    latex_file.write("\multicolumn{1}{|l|}{\\textit{" + d_fp_for_table +"}} & \multicolumn{8}{r|}{ } \\\\ \n")
                    latex_file.write("\t\hline\n")
                    latex_file.write("\t\hline\n")
                    latex_file.write("\t\hline\n")
                    sm_stats = [['not used'], [], [], [], [], [], [], [], []]
                    sm_stats_no_resampling = [['not used'], [], [], [], [], [], [], [], []]
                    sm_stats_sampling = [['not used'], [], [], [], [], [], [], [], []]
                    for sm, sm_values in dict_with_data[sampling_version][class_alg][d_name].items():
                        for i in range (1, 9):
                            if sm == 'No Re-sampling':
                                sm_stats_no_resampling[i].append(float(sm_values[i]))
                            else:
                                sm_stats_sampling[i].append(float(sm_values[i]))
                            sm_stats[i].append(float(sm_values[i]))
                    for i in range (1, 9):
                        if len(sm_stats_no_resampling[i]) > 0:
                            max_elem_no_resampling = max(sm_stats_no_resampling[i])
                            max_elem_indexes_no_rs = [index for index, value in enumerate(sm_stats_no_resampling[i]) if
                                                      value == max_elem_no_resampling]
                            for idx in max_elem_indexes_no_rs:
                                max_elem_rounded = round(max_elem_no_resampling, 3)
                                if max_elem_rounded not in best_results_dict[d_fp_for_table][i]['NormalCase']:
                                    best_results_dict[d_fp_for_table][i]['NormalCase'][max_elem_rounded][
                                        "ClassAlgs"] = list()
                                best_results_dict[d_fp_for_table][i]['NormalCase'][max_elem_rounded][
                                    "ClassAlgs"].append(class_alg)

                        if len(sm_stats_sampling[i]) > 0:
                            max_elem_sampling = max(sm_stats_sampling[i])
                            max_elem_indexes_sampling = [index for index, value in enumerate(sm_stats_sampling[i]) if
                                                      value == max_elem_sampling]
                            for idx in max_elem_indexes_sampling:
                                max_elem_rounded = round(max_elem_sampling, 3)
                                if max_elem_rounded not in best_results_dict[d_fp_for_table][i]['ResampledCase']:
                                    best_results_dict[d_fp_for_table][i]['ResampledCase'][max_elem_rounded][
                                        "ClassAlgs"] = list()
                                    best_results_dict[d_fp_for_table][i]['ResampledCase'][max_elem_rounded][
                                        "SamplingMethods"] = list()
                                best_results_dict[d_fp_for_table][i]['ResampledCase'][max_elem_rounded][
                                    "ClassAlgs"].append(class_alg)
                                # if idx == 0:
                                #     idx += 1
                                best_results_dict[d_fp_for_table][i]['ResampledCase'][max_elem_rounded][
                                    "SamplingMethods"].append(transform_idx(sampling_version, idx))

                        if len(sm_stats[i]) > 0:
                            max_elem = max(sm_stats[i])
                            max_elem_indexes = [index for index, value in enumerate(sm_stats[i]) if value == max_elem]
                            for idx in max_elem_indexes:
                                max_elem_rounded = round(max_elem, 3)
                                sm_stats[i][idx] = '\\textbf{' + max_elem_rounded + '}'

                    i = 0
                    for sm, sm_values in dict_with_data[sampling_version][class_alg][d_name].items():
                        print (i)
                        latex_file.write("\multicolumn{1}{|r|}{\\textit{" + sm + " (" + str(sm_values[0]) + ")}}" + append_multicolumn(round(sm_stats[1][i], 3)) + append_multicolumn(round(sm_stats[2][i], 3)) + append_multicolumn(round(sm_stats[3][i], 3)) + append_multicolumn(round(sm_stats[4][i], 3)) + append_multicolumn(round(sm_stats[5][i], 3)) + append_multicolumn(round(sm_stats[6][i], 3)) + append_multicolumn(round(sm_stats[7][i], 3)) + append_multicolumn(round(sm_stats[8][i], 3)) +"\\\\ \n")
                        # if sm == 'No Resampling':
                        #     latex_file.write("\t\hline\n")
                        #     latex_file.write("\t\hline\n")
                        i += 1
                    latex_file.write("\t\hline\n")

                # if dict_with_data['sampling_case'] == "Undersampling":
                #
                #     latex_file.write("\t\multicolumn{1}{|c|}{Dataset \& metric} & \multicolumn{1}{c|}{RU} & \multicolumn{1}{c|}{CC} "
                #      "&\multicolumn{1}{c|}{TL} & \multicolumn{1}{c|}{NM1} & \multicolumn{1}{c|}{NM2} & \multicolumn{1}{c|}{NM3} "
                #      "& \multicolumn{1}{c|}{CNN} & \multicolumn{1}{c|}{OSS} & \multicolumn{1}{c|}{ENN} \\\\ \n")
                #     latex_file.write("\t\hline\n")
                #     latex_file.write(
                #         "\t\multicolumn{1}{|l|}{\\textit{" + d_fp_for_table.split(".")[0] + "}} & & & & & & & & & \\\\ \n")
                # else:
                #     latex_file.write("\t\multicolumn{1}{|c|}{Dataset \& metric} & \multicolumn{1}{c|}{RO} &"
                #      " \multicolumn{1}{c|}{SMOTE} & \multicolumn{1}{c|}{ADASYN} &"
                #      " \multicolumn{1}{c|}{SMOTE+TL} & \multicolumn{1}{c|}{SMOTE+ENN} \\\\ \n")
                #     latex_file.write("\t\hline\n")
                #     latex_file.write(
                #         "\t\multicolumn{1}{|l|}{\\textit{" + d_fp_for_table.split(".")[0] + "}} & & & & & & & & & \\\\ \n")
                # latex_file.write("\t\hline\n")
                # latex_file.write("\t\multicolumn{1}{|r|}{$Precision$} " + "".join([" & \multicolumn{1}{c|}{" + el + "}" for el in precisions])  + " \\\\ \n")
                latex_file.write("\t\caption{}\n")
                latex_file.write("\end{longtable}\n")
            latex_file.write("\end{document}")
            chapter_inited = False
        pre_table_inited = False
    with open("../latex-gen/extreme_imbalance/" + "br-latex-tables.txt", "w+", encoding="utf8") as br_latex_file:
        br_latex_file.write(pre_table_content())
        br_latex_file.write("\\chapter*{Най-добрите резултати за " + im_degree_dict[first_d_name] + "}\n")
        br_latex_file.write("\\begin{longtable}{|m{1.5cm}|m{1cm}|m{1cm}|m{1cm}|m{0.5cm}|m{1.7cm}|}\n")
        br_latex_file.write("\t\\hline\n")
        br_latex_file.write("\t\multicolumn{6}{|c|}{Best results across all tests} \\\\")
        br_latex_file.write("\t\\hline\n")
        br_latex_file.write("\t\\multicolumn{1}{|c|}{Dataset \& metric} & \multicolumn{1}{c|}{BV w/o SM} & \multicolumn{1}{c|}{Alg}&\multicolumn{1}{c|}{BV w/ SM} & \multicolumn{1}{c|}{Alg} & \multicolumn{1}{c|}{SM} \\\\ \n")
        br_latex_file.write("\t\hline\n")
        for dataset, metric_dict in best_results_dict.items():
            br_latex_file.write("\t\multicolumn{1}{|l|}{\\textit{" + dataset + "}} & \multicolumn{5}{r|}{ } \\\\ \n")
            br_latex_file.write("\t\hline\n")
            br_latex_file.write("\t\hline\n")
            br_latex_file.write("\t\hline\n")
            for metric_index in metric_dict.keys():
                max_ele_nc = round(max(map(float, metric_dict[metric_index]['NormalCase'].keys())), 3)
                max_ele_rc = round(max(map(float, metric_dict[metric_index]['ResampledCase'].keys())), 3)
                max_ele_bold_ver_nc = "\\textbf{" + max_ele_nc + "}" if max_ele_nc >= max_ele_rc else max_ele_nc
                max_ele_bold_ver_rc = "\\textbf{" + max_ele_rc + "}" if max_ele_rc >= max_ele_nc else max_ele_rc

                nc_algs = extract_algs_sm_in_shortstacks(metric_dict[metric_index]['NormalCase'][max_ele_nc]['ClassAlgs'])
                rc_algs = extract_algs_sm_in_shortstacks(metric_dict[metric_index]['ResampledCase'][max_ele_rc]['ClassAlgs'])
                rc_sm = extract_algs_sm_in_shortstacks(metric_dict[metric_index]['ResampledCase'][max_ele_rc]['SamplingMethods'])

                br_latex_file.write("\t\multicolumn{1}{|r|}{$"+ transform_metric_idx(metric_index) +"$}  & \multicolumn{1}{c|}{" + max_ele_bold_ver_nc + "} & \multicolumn{1}{c|}{ " + nc_algs + "} & \multicolumn{1}{c|}{" + max_ele_bold_ver_rc + "} &\multicolumn{1}{c|}{" + rc_algs + "}  & \multicolumn{1}{c|}{ " + rc_sm + "} \\\\ \n")
                br_latex_file.write("\t\cline{5-6} \n")
            br_latex_file.write("\t\hline\n")
        # br_latex_file.write("\t\hline\n")
        br_latex_file.write("\t\caption{Най-добрите резултати измежду всички семплиращи методи и класификационни алгоритми} \\\\ \n")
        br_latex_file.write("\end{longtable}")
        br_latex_file.write("\end{document}")
    print ("x")

# def sample_test():
    # print ('hi there')


def extract_algs_sm_in_shortstacks(provided_list):
    provided_list = list(set(provided_list))
    result = "\shortstack[l]{"
    list_length = len(provided_list)
    for idx, el in enumerate(provided_list):
        result += el
        if idx < (list_length - 1):
            result += '/'
        if idx % 2 != 0 and idx > 0:
            if idx < (list_length - 1):
                result += '\\\\'
    result += "}"
    return result
if __name__ == '__main__':

    # print (extract_algs_sm_in_shortstacks(['SMOTE', 'TN', 'CNN', 'ASD', 'VGZZ']))


    # latex_dict = large_func()
    # generate_latex_output(latex_dict)
    profiler = Profile()
    profiler.runcall(large_func)
    stats = Stats(profiler)
    stats.strip_dirs()
    stats.sort_stats('cumulative')
    stats.print_stats()
    # classify()
    # large_func()