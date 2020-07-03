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

import sys
# sys.path.append('G:/python-workspace/resampler/algos')
THIS_DIR = dirname(__file__)
CODE_DIR = abspath(join(THIS_DIR, '..', 'algos'))
sys.path.append(CODE_DIR)

import smote_boost


def over_sampling_algs():
    algs = list()
    algs.append(("No Rs Oversampling case", "No Resampling"))
    algs.append((RandomOverSampler(random_state=1), 'RO'))
    # algs.append((SMOTE(random_state=1), 'SMOTE'))
    # algs.append((ADASYN(random_state=1), 'ADASYN'))
    # algs.append((SMOTETomek(random_state=1), 'SMOTE+TL'))
    # algs.append((SMOTEENN(random_state=1), 'SMOTE+ENN'))
    # algs.append((smote_boost.SMOTEBoost(), "SMOTEBoost"))
    return algs


def under_sampling_algs():
    algs = list()
    algs.append(("No Rs Undersampling case", "No Resampling"))
    algs.append((RandomUnderSampler(random_state=1), 'RU'))
    # algs.append((ClusterCentroids(random_state=1), 'CC'))
    # algs.append((TomekLinks(), 'TL'))
    # algs.append((NearMiss(version=1), 'NM1'))
    # algs.append((NearMiss(version=2), 'NM2'))
    # algs.append((NearMiss(version=3), 'NM3'))
    # algs.append((CondensedNearestNeighbour(random_state=1), 'CNN'))
    # algs.append((OneSidedSelection(random_state=1), 'OSS'))
    # algs.append((EditedNearestNeighbours(), 'ENN'))
    # algs.append((NeighbourhoodCleaningRule(), 'NCL'))
    # algs.append((InstanceHardnessThreshold(), 'IHT'))
    return algs


def is_over_sampling(sampling_alg, list_ovrsampling_algs):
    return type(sampling_alg).__name__ in [type(alg).__name__ for alg in list_ovrsampling_algs]


def gather_class_algs():
    algs = []
    algs.append(("CART (Decision Tree Classifier)", tree.DecisionTreeClassifier(criterion='entropy', random_state=np.random.RandomState(1))))
    # algs.append(("SVM", svm.SVC(probability=True, random_state=np.random.RandomState(1))))
    # algs.append(RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=np.random.RandomState(1)))
    # algs.append(svm.SVC(probability=True, random_state=np.random.RandomState(1)))
    return algs


def round(digit, digit_after_fp):
    if not isinstance(digit, str):
        return ("{:." + str(digit_after_fp) + "f}").format(digit)
    else:
        return digit

def large_func():
    dataset_files = os.listdir('../datasets/cl_datasets')
    rand_state = np.random.RandomState(1)
    cv = StratifiedKFold(n_splits=10, random_state=rand_state)
    latex_dict = {}
    latex_dict['Under-sampling'] = {}
    latex_dict['Over-sampling'] = {}
    class_algs = gather_class_algs()
    for alg_tuple in class_algs:
        alg_alias = alg_tuple[0]
        latex_dict['Under-sampling'][alg_alias] = {}
        latex_dict['Over-sampling'][alg_alias] = {}
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
                    ovr_s_algs = over_sampling_algs()
                    undr_s_algs = under_sampling_algs()
                    sampling_algs = undr_s_algs + ovr_s_algs

                    # used_algs = [tree.DecisionTreeClassifier(criterion='entropy', random_state=np.random.RandomState(1))]
                    d_res_file.write("********************************************************************************\n");
                    for alg_tuple in class_algs:
                        alg_alias = alg_tuple[0]
                        alg = copy.deepcopy(alg_tuple[1])
                        latex_dict['Under-sampling'][alg_alias][d_fp] = {}
                        latex_dict['Over-sampling'][alg_alias][d_fp] = {}
                        d_res_file.write("Classification algorithm:" + alg_alias + "\n")
                        for sa in sampling_algs:
                            d_res_file.write("Sampling_algorithm:" + sa[1] + "\n")
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
                                d_res_file.write("Fold number: {}\n".format(i))
                                t1_before, t2_before = Counter(y_values).most_common(2)
                                if sa_name == 'SMOTEBoost':
                                    model = sa[0].fit(x_values[train], y_values[train].astype(int))
                                    percentage_pos_samples.append(0)
                                elif sa_name == 'No Resampling':
                                    model = alg.fit(x_values[train], y_values[train].astype(int))
                                    percentage_pos_samples.append((t2_before[1] / len(y_values) * 100))
                                else:
                                    x_resampled_values, y_resampled_values = sa[0].fit_resample(x_values[train],
                                                                                                y_values[train].astype(
                                                                                                    int))
                                    t1_after, t2_after = Counter(y_resampled_values).most_common(2)
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
                                d_res_file.write("This is the report from the imbalanced_learning module:\n")
                                d_res_file.write(report_from_imbalanced_learning + "\n")
                                # d_res_file.write("Precision: {}\n".format(tpr))
                                d_res_file.write("Precision: {}\n".format(prf1[0]))
                                d_res_file.write("Recall: {}\n".format(prf1[1]))
                                d_res_file.write("F_measure: {}\n".format(prf1[2]))
                                specificity = specificity_score(y_values[test], predicted_classes)
                                # g_means_1.append(np.sqrt(prf1[1] * specificity))
                                bal_accuracy = (prf1[1] + specificity) / 2
                                bal_accs.append(bal_accuracy)
                                g_mean_1 = np.sqrt(prf1[1] * specificity)
                                g_mean_2 = np.sqrt(prf1[0] * prf1[1])
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

                            avg_percent_pos, avg_bal_acc, avg_pre, avg_rec, avg_f1, avg_g_mean, avg_g_mean_2, avg_roc_auc, avg_pr_auc = (sum(percentage_pos_samples) / i, sum(bal_accs) / i,
                            (sum(map(operator.itemgetter(0), pr_rec_f1s)) / i),
                            (sum(map(operator.itemgetter(1), pr_rec_f1s)) / i),
                            (sum(map(operator.itemgetter(2), pr_rec_f1s)) / i),
                            sum(g_means_1) / i,
                            sum(g_means_2) / i, sum(roc_aucs) / i,  sum(pr_aucs) / i)
                            d_res_file.write("----------------------------------------------------------\n")
                            d_res_file.write("Average results (for all folds)\n")
                            d_res_file.write("Dataset: {}\n".format(d_fp))
                            d_res_file.write("Sampling algorithm: {}\n".format(str(type(sa).__name__)))
                            d_res_file.write("Classification algorithm: {}".format(alg_alias))
                            d_res_file.write("Balanced Accuracy: {}\n".format(avg_bal_acc))
                            d_res_file.write("Avg Precision: {}\n".format(avg_pre))
                            d_res_file.write("Avg Recall: {}\n".format(avg_rec))
                            d_res_file.write("Avg F_measure: {}\n".format(avg_f1))
                            d_res_file.write("Avg G_mean_1: {}\n".format(avg_g_mean))
                            d_res_file.write("Avg G_mean_2: {}\n".format(avg_g_mean_2))
                            d_res_file.write("Avg AUC_ROC: {}\n".format(avg_roc_auc))
                            d_res_file.write("Avg AUC_PR: {}\n".format(avg_pr_auc))
                            d_res_file.write("----------------------------------------------------------\n")

                            d_res_file.write("\n\n")

                            if sa in ovr_s_algs:
                                latex_dict['Over-sampling'][alg_alias][d_fp][sa_name] = [round(avg_percent_pos, 1), round(avg_bal_acc, 3), round(avg_pre, 3), round(avg_rec, 3), round(avg_f1, 3), round(avg_g_mean, 3),
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


def append_multicolumn(value):
    return " & \multicolumn{1}{c|}{" + str(value) + "}"


def pre_table_content():
    content = r"\documentclass[12pt,oneside]{report}\usepackage[a4paper, left=2cm, right=2cm, top=2.5cm, bottom=2.5cm]{geometry}\usepackage[table]{xcolor}\usepackage{fancyhdr}\pagestyle{fancy}\usepackage[T2A]{fontenc}\usepackage[english]{babel}\usepackage[utf8]{inputenc}\usepackage{longtable}\usepackage{amssymb}\usepackage{amsmath}\usepackage{color}\usepackage{caption}\captionsetup[table]{name=Таблица}\definecolor{lightgray}{gray}{0.9}\fancyhead{}\fancyhead[RO,LE]{Дебалансирани множества от данни и проблемите и решенията, свързани с тях}\fancyfoot{}\fancyfoot[C]{\thepage}\begin{document}"
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


def transform_idx(case, idx):
    if case is 'Under-sampling':
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
    for sampling_version in dict_with_data.keys():
        with open("../latex-gen/" + sampling_version + "-latex-tables.txt", "w+", encoding="utf8") as latex_file:
            for class_alg in dict_with_data[sampling_version].keys():
                for d_name in dict_with_data[sampling_version][class_alg].keys():
                    d_fp_for_table = d_name.replace("_", "\\_")
                    init_nested_dicts(best_results_dict[d_fp_for_table])
            for class_alg in dict_with_data[sampling_version].keys():
                first_d_name = list(dict_with_data[sampling_version][class_alg].keys())[0]
                #sa = RandomUnderSampler()
                #alg = tree.DecisionTreeClassifier(criterion='entropy', random_state=np.random.RandomState(1))
                #sa_name = type(sa).__name__
                #class_alg_name = type(alg).__name__
                #precisions = ['1', '2', '3', '4', '5', '6', '7', '8', '9']
                #latex_file.write("Latex statistics for sm {} and algo {}\n\n".format(sa_name, class_alg_name))
                latex_file.write("\\chapter*{" + sampling_version + " резултати за " + im_degree_dict[first_d_name] + "}")
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
                    latex_file.write("\multicolumn{1}{|l|}{\\textit{" + d_fp_for_table +"}} &  & & & & & & & \\\\ \n")
                    latex_file.write("\t\hline\n")
                    sm_stats = [['not used'], [], [], [], [], [], [], [], [], []]
                    sm_stats_no_resampling = [['not used'], [], [], [], [], [], [], [], [], []]
                    sm_stats_sampling = [['not used'], [], [], [], [], [], [], [], [], []]
                    for sm, sm_values in dict_with_data[sampling_version][class_alg][d_name].items():
                        for i in range (1, 9):
                            if sm == 'No Resampling':
                                sm_stats_no_resampling[i].append(float(sm_values[i]))
                            else:
                                sm_stats_sampling[i].append(float(sm_values[i]))
                            sm_stats[i].append(float(sm_values[i]))
                    for i in range (1, 9):
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
                            if idx == 0:
                                idx += 1
                            best_results_dict[d_fp_for_table][i]['ResampledCase'][max_elem_rounded][
                                "SamplingMethods"].append(transform_idx(sampling_version, idx))


                        max_elem = max(sm_stats[i])
                        max_elem_indexes = [index for index, value in enumerate(sm_stats[i]) if value == max_elem]
                        for idx in max_elem_indexes:
                            max_elem_rounded = round(max_elem, 3)
                            sm_stats[i][idx] = '\\textbf{' + max_elem_rounded + '}'

                    i = 0
                    for sm, sm_values in dict_with_data[sampling_version][class_alg][d_name].items():
                        print (i)
                        latex_file.write("\multicolumn{1}{|r|}{\\textit{" + sm + " (" + str(sm_values[0]) + ")}}" + append_multicolumn(round(sm_stats[1][i], 3)) + append_multicolumn(round(sm_stats[2][i], 3)) + append_multicolumn(round(sm_stats[3][i], 3)) + append_multicolumn(round(sm_stats[4][i], 3)) + append_multicolumn(round(sm_stats[5][i], 3)) + append_multicolumn(round(sm_stats[6][i], 3)) + append_multicolumn(round(sm_stats[7][i], 3)) + append_multicolumn(round(sm_stats[8][i], 3)) +"\\\\ \n")
                        latex_file.write("\t\hline\n")
                        if sm == 'No Resampling':
                            latex_file.write("\t\hline\n")
                            latex_file.write("\t\hline\n")
                        i += 1

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
            latex_file.write(r"\begin{longtable}{|m{1.5cm}|m{1cm}|m{1cm}|m{1cm}|m{1.7cm}|} ")
            latex_file.write(r"\hline")
            latex_file.write(r"\multicolumn{5}{|c|}{Best results across all tests} \\")
            latex_file.write(r"\hline")
            latex_file.write(r"\multicolumn{1}{|c|}{Dataset \& metric} & \multicolumn{1}{c|}{BV No SM} &\multicolumn{1}{c|}{BV SM} & \multicolumn{1}{c|}{Sampling Method} & \multicolumn{1}{c|}{Algorithm} \\")
            latex_file.write(r"\hline")
            # latex_file.write(r"\multicolumn{1}{|l|}{\textit{" + )
    print ("x")

# def sample_test():
    # print ('hi there')

if __name__ == '__main__':
    #latex_dict = large_func()
    #generate_latex_output(latex_dict)
    profiler = Profile()
    profiler.runcall(large_func)
    stats = Stats(profiler)
    stats.strip_dirs()
    stats.sort_stats('cumulative')
    stats.print_stats()
    # large_func()