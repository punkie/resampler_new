import copy
import numpy as np
import operator
import math
from matplotlib import pyplot as plt
from imblearn.metrics import classification_report_imbalanced, specificity_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_curve, auc
from scipy import interp
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_recall_fscore_support
from functions import resampling_functions
from collections import Counter
from sklearn.decomposition import PCA


def compute_some_statistics_for_the_dataset(dataset):
    dataset['number_of_examples'] = len(dataset['y_values'])
    unique_target_values = Counter(dataset['y_values'])
    first_mc_tuple, second_mc_tuple = unique_target_values.most_common(2)
    dataset['y_values_as_set'] = (first_mc_tuple[0], second_mc_tuple[0])
    dataset['number_of_positive_examples'] = second_mc_tuple[1]
    dataset['positive_examples_percentage'] = "{:.1f}".format((second_mc_tuple[1] / len(dataset['y_values'])) * 100)
    dataset['imbalanced_ratio'] =  round_half_up(first_mc_tuple[1] / second_mc_tuple[1], 1)
    __create_statistics_string(dataset)


def round_half_up(n, decimals=0):
    multiplier = 10 ** decimals
    return math.floor(n * multiplier + 0.5) / multiplier

def classify(classifier_thread):
    classifying_data = {}
    classifying_data['main_tuples'] = []
    normal_dataset = classifier_thread.main_window.state.dataset
    #resampled_dataset = state.resampled_dataset
    rand_state = np.random.RandomState(1)
    other_rand_state = np.random.RandomState(0)
    splits = classifier_thread.main_window.state.number_of_folds
    cv = StratifiedKFold(n_splits=splits, random_state=rand_state)
    # classifier = RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=rand_state)
    # classifier = svm.SVC(probability=True, random_state=rand_state)
    classifier = copy.deepcopy(classifier_thread.main_window.state.classification_algorithm.value[1])
    X_normal = normal_dataset['x_values']
    y_normal = normal_dataset['y_values']

    tprs = []
    mean_fpr = np.linspace(0, 1, 100)
    roc_aucs = []
    pr_aucs = []
    i = 0
    g_means_1 = []
    g_means_2 = []
    pr_rec_f1s = []
    preds_list = []
    trues_list = []
    bal_accs = []
    average_precisions = []
    f, ax = plt.subplots()
    for train, test in cv.split(X_normal, y_normal):
        if classifier_thread.do_resampling:
            r_dataset = resampling_functions.\
            do_resampling_without_writing_to_file(classifier_thread.main_window.state.sampling_algorithm_experiments_tab,
                                                  X_normal[train], y_normal[train])
            classifier_ = classifier.fit(r_dataset['x_values'], r_dataset['y_values'].astype(int))
        else:
            classifier_ = classifier.fit(X_normal[train], y_normal[train].astype(int))
        #predicted_y_scores = classifier_.decision_function(X_normal[test])
        predicted_classes = classifier_.predict(X_normal[test])
        probas_ = classifier_.predict_proba(X_normal[test])
        # if classifier_thread.do_resampling:
            # plot_step = 0.02
            # pca = PCA(n_components=2)
            # X = pca.fit_transform(r_dataset['x_values'])
            # x_min, x_max = X[:, -1].min() - 1, X[:, -1].max() + 1
            # y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
            # xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
            #                      np.arange(y_min, y_max, plot_step))
            # pca_class = classifier.fit(X, r_dataset['y_values'])
            # Z = pca_class.predict(np.c_[xx.ravel(), yy.ravel()])
            # Z = Z.reshape(xx.shape)
            # ax.contourf(xx, yy, Z, alpha=0.4)
            # ax.contourf(r_dataset['x_values'], r_dataset['y_values'], Z, alpha=0.4)
            # classifying_data['new_x'] = X
            # classifying_data['other'] = (xx, yy, Z)
            # classifying_data['stored_classifier'] = classifier_
            # return classifying_data
        # else:
        #     break
        preds_list.append(probas_)
        trues_list.append(y_normal[test])
        average_precision = average_precision_score(y_normal[test], probas_[:, 1])
        average_precisions.append(average_precision)
        fpr, tpr, thresholds = roc_curve(y_normal[test], probas_[:, 1])
        prf1 = precision_recall_fscore_support(y_normal[test], predicted_classes, average='binary')
        pr_rec_f1s.append(prf1)
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        roc_aucs.append(roc_auc)
        report_from_imbalanced_learning = classification_report_imbalanced(y_normal[test], predicted_classes)
        specificity = specificity_score(y_normal[test], predicted_classes)
        g_means_1.append(np.sqrt(prf1[1] * specificity))
        #pr_rec_f1s.append(g_mean1)
        g_means_2.append(np.sqrt(prf1[0] * prf1[1]))
        bal_accuracy = (prf1[1] + specificity) / 2
        bal_accs.append(bal_accuracy)

        # pr_rec_f1s.append(g_mean2)
        #pr_rec_f1s.append(prf1)

        # print("iteration #{} with resampled dataset ({})."
        #    " statistics:\n{}\n".format(i, classifier_thread.do_resampling,
        #                                (classification_report_imbalanced(y_normal[test], predicted_classes))))
        classifying_data['main_tuples'].append((fpr, tpr, roc_auc, i, y_normal[test], probas_[:, 1], average_precision))

        i += 1
        if classifier_thread.do_resampling:
            classifier_thread.update_resample_classify_progress_bar.emit(i)
        else:
            classifier_thread.update_normal_classify_progress_bar.emit(i)
    #grouped_pr_rec_f1s = zip(*pr_rec_f1s)
    classifying_data['pre_rec_f1_g_mean1_g_mean2_tuple'] = ((sum(map(operator.itemgetter(0), pr_rec_f1s)) / i),
                                            (sum(map(operator.itemgetter(1), pr_rec_f1s)) / i),
    (sum(map(operator.itemgetter(2), pr_rec_f1s)) / i),
    sum(g_means_1) / i,
    sum(g_means_2) / i)
    classifying_data['precision'] = list(map(operator.itemgetter(0), pr_rec_f1s))
    classifying_data['recall'] = list(map(operator.itemgetter(1), pr_rec_f1s))
    # \
    #     ,
    #
    # (sum(pr_rec_f1s[1]) / i),
    # (sum(pr_rec_f1s[2]) / i))
    classifying_data['bal_acc'] = sum(bal_accs) / i
    classifying_data['preds_list'] = preds_list
    classifying_data['trues_list'] = trues_list
    classifying_data['avg_roc'] = sum(roc_aucs) / i
    classifying_data['average_precision'] = sum(average_precisions) / i
    # sorted_recall = sorted((map(operator.itemgetter(1), pr_rec_f1s)))
    # recalls = []
    # max_precisions = []
    # for rec in np.linspace(0, 1, num=11):
    #     filtered_recalls = list(filter(lambda x: x[1] >= rec, pr_rec_f1s))
    #     max_precision = max(map(operator.itemgetter(0), filtered_recalls)) if len(filtered_recalls) > 0 else 0
    #     recalls.append(rec)
    #     max_precisions.append(max_precision)
    # classifying_data['recalls'] = recalls
    # classifying_data['max_precisions'] =  max_precisions
    # classifying_data['sorted_recall'] = sorted_recall
    # classifying_data['precision'] = list(map(operator.itemgetter(0), pr_rec_f1s))
    #
    # classifying_data['decreasing_max_precision'] = list(
    #     np.maximum.accumulate(classifying_data['precision'][::-1])[::-1])

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(roc_aucs)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    classifying_data['mean_values_tuple'] = (mean_fpr, mean_tpr, mean_auc, std_auc, tprs_lower, tprs_upper)
    return classifying_data


# def compute_data_for_pr_graph(state):
#     classifying_data = {}
#     normal_dataset = state.dataset
#     rand_state = np.random.RandomState(1)
#     cv = StratifiedKFold(n_splits=10)
#     classifier = copy.deepcopy(state.classification_algorithm.value[1])
#     # classifier.random_state = None
#     # classifier reset random state
#     X_normal = normal_dataset['x_values']
#     y_normal = normal_dataset['y_values']
#
#     list_with_max_precisions = []
#     for cv_iterations in range(20):
#         i = 0
#         pr_rec_f1s = []
#         for train, test in cv.split(X_normal, y_normal):
#             if False:
#                 r_dataset = resampling_functions. \
#                     do_resampling_without_writing_to_file(state.sampling_algorithm,
#                                                           X_normal[train], y_normal[train])
#                 classifier_ = classifier.fit(r_dataset['x_values'], r_dataset['y_values'])
#             else:
#                 classifier_ = classifier.fit(X_normal[train], y_normal[train])
#             predicted_classes = classifier_.predict(X_normal[test])
#             prf1 = precision_recall_fscore_support(y_normal[test], predicted_classes, average='binary')
#             pr_rec_f1s.append(prf1)
#             i += 1
#         # grouped_pr_rec_f1s = zip(*pr_rec_f1s)
#         classifying_data['pre_rec_f1_tuple'] = ((sum(map(operator.itemgetter(0), pr_rec_f1s)) / i),
#                                                 (sum(map(operator.itemgetter(1), pr_rec_f1s)) / i),
#                                                 (sum(map(operator.itemgetter(2), pr_rec_f1s)) / i))
#         sorted_recall = sorted((map(operator.itemgetter(1), pr_rec_f1s)))
#         recalls = []
#         max_precisions = []
#         for rec in np.linspace(0, 1, num=11):
#             filtered_recalls = list(filter(lambda x: x[1] >= rec, pr_rec_f1s))
#             max_precision = max(map(operator.itemgetter(0), filtered_recalls)) if len(filtered_recalls) > 0 else 0
#             recalls.append(rec)
#             max_precisions.append(max_precision)
#         list_with_max_precisions.append(max_precisions)
#     classifying_data['recalls'] = recalls
#     classifying_data['max_precisions'] = list(map(lambda x: sum(x) / 20, zip(*list_with_max_precisions)))
#     return classifying_data


def get_mean_precision_recall_f1_scores(state):
    # precision, recall, f1, gm1, gm2 = state.classified_data_normal_case['pre_rec_f1_g_mean1_g_mean2_tuple']
    precision, recall, f1, gm1, gm2 = (0, 0, 0, 0, 0)
    precision_re, recall_re, f1_re, gm1_re, gm2_re = state.classified_data_resampled_case['pre_rec_f1_g_mean1_g_mean2_tuple']

    heading = "{:>40}".format("Normal DS  /  Resampled DS")
    precision = "{:>15}".format("Precision: ") + "{:.3f}".format(precision) + \
                "{:>15}".format("Precision: ") + "{:.3f}".format(precision_re)
    recall = "{:>17}".format("Recall: ") + "{:.3f}".format(recall) + \
             "{:>17}".format("Recall: ") + "{:.3f}".format(recall_re)
    f_score = "{:>14}".format("F_score: ") +  "{:.3f}".format(f1) + \
              "{:>15}".format("F_score: ") +  "{:.3f}".format(f1_re)
    g_mean1 = "{:>14}".format("G_mean_1: ") +  "{:.3f}".format(gm1) + \
              "{:>15}".format("G_mean_1: ") +  "{:.3f}".format(gm1_re)
    g_mean2 = "{:>14}".format("G_mean_2: ") +  "{:.3f}".format(gm2) + \
              "{:>15}".format("G_mean_2: ") +  "{:.3f}".format(gm2_re)
    result_scores = heading + "\n\n" + precision + "\n" + recall + "\n" + f_score + "\n" + g_mean1 + "\n" + g_mean2
    print (result_scores)
    return result_scores


def __create_statistics_string(dataset):
    dataset_statistics_string = 'Number of examples: {}\nPercentage of minor class examples: {}\n' \
                                'Imbalanced ratio: {}'.format(
        dataset['number_of_examples'], dataset['positive_examples_percentage'], dataset['imbalanced_ratio'])
    dataset['dataset_statistics_string'] = dataset_statistics_string


