import numpy as np
import operator
from imblearn.metrics import classification_report_imbalanced
from scipy import interp
from sklearn import tree, svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_recall_fscore_support
from functions import resampling_functions
from collections import Counter


def compute_some_statistics_for_the_dataset(dataset):
    dataset['number_of_examples'] = len(dataset['y_values'])
    unique_target_values = Counter(dataset['y_values'])
    first_mc_tuple, second_mc_tuple = unique_target_values.most_common(2)
    dataset['y_values_as_set'] = (first_mc_tuple[0], second_mc_tuple[0])
    dataset['target_class_percentage'] = second_mc_tuple[1] / len(dataset['y_values'])
    dataset['imbalanced_ratio'] = first_mc_tuple[1] / second_mc_tuple[1]
    __create_statistics_string(dataset)


def classify(classifier_thread):
    classifying_data = {}
    classifying_data['main_tuples'] = []
    normal_dataset = classifier_thread.main_window.state.dataset
    #resampled_dataset = state.resampled_dataset
    rand_state = np.random.RandomState(1)
    other_rand_state = np.random.RandomState(0)
    cv = StratifiedKFold(n_splits=10, random_state=rand_state)
    # classifier = RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=rand_state)
    # classifier = svm.SVC(probability=True, random_state=rand_state)
    classifier = tree.DecisionTreeClassifier(random_state=rand_state, criterion='entropy')
    X_normal = normal_dataset['x_values']
    y_normal = normal_dataset['y_values']

    #X_resampled = resampled_dataset['x_values']
    #y_resampled = resampled_dataset['y_values']

    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    i = 0
    pr_rec_f1s = []
    for train, test in cv.split(X_normal, y_normal):
        if classifier_thread.do_resampling:
            r_dataset = resampling_functions.\
            do_resampling_without_writing_to_file(classifier_thread.main_window.state.sampling_algorithm,
                                                  X_normal[train], y_normal[train])
            classifier_ = classifier.fit(r_dataset['x_values'], r_dataset['y_values'])
        else:
            classifier_ = classifier.fit(X_normal[train], y_normal[train])
        #predicted_y_scores = classifier_.decision_function(X_normal[test])
        predicted_classes = classifier_.predict(X_normal[test])
        probas_ = classifier_.predict_proba(X_normal[test])
        average_precision = average_precision_score(y_normal[test], probas_[:, 1])
        fpr, tpr, thresholds = roc_curve(y_normal[test], probas_[:, 1])
        prf1 = precision_recall_fscore_support(y_normal[test], predicted_classes, average='binary')
        pr_rec_f1s.append(prf1)
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        if not classifier_thread.do_resampling:
            print("iteration #{} with resampled dataset ({})."
                   " statistics:\n{}\n".format(i, classifier_thread.do_resampling,
                                               (classification_report_imbalanced(y_normal[test], predicted_classes))))
        classifying_data['main_tuples'].append((fpr, tpr, roc_auc, i, y_normal[test], probas_[:, 1], average_precision))

        i += 1
        if classifier_thread.do_resampling:
            classifier_thread.update_resample_classify_progress_bar.emit(i)
        else:
            classifier_thread.update_normal_classify_progress_bar.emit(i)
    #grouped_pr_rec_f1s = zip(*pr_rec_f1s)
    classifying_data['pre_rec_f1_tuple'] = ((sum(map(operator.itemgetter(0), pr_rec_f1s)) / i),
                                            (sum(map(operator.itemgetter(1), pr_rec_f1s)) / i),
    (sum(map(operator.itemgetter(2), pr_rec_f1s)) / i))

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    classifying_data['mean_values_tuple'] = (mean_fpr, mean_tpr, mean_auc, std_auc, tprs_lower, tprs_upper)
    return classifying_data


def __create_statistics_string(dataset):
    dataset_statistics_string = 'Number of examples: {}\nPercentage of minor class examples: {}\n' \
                                'Imbalanced ratio: {}'.format(
        dataset['number_of_examples'], dataset['target_class_percentage'], dataset['imbalanced_ratio'])
    dataset['dataset_statistics_string'] = dataset_statistics_string


def get_mean_precision_recall_f1_scores(state):
    precision, recall, f1 = state.classified_data_normal_case['pre_rec_f1_tuple']
    precision_re, recall_re, f1_re = state.classified_data_resampled_case['pre_rec_f1_tuple']

    heading = "{:>40}".format("Normal DS  /  Resampled DS")
    precision = "{:>15}".format("Precision: ") + "{:.3f}".format(precision) + \
                "{:>15}".format("Precision: ") + "{:.3f}".format(precision_re)
    recall = "{:>17}".format("Recall: ") + "{:.3f}".format(recall) + \
             "{:>17}".format("Recall: ") + "{:.3f}".format(recall_re)
    f_score = "{:>14}".format("F_score: ") +  "{:.3f}".format(f1) + \
              "{:>15}".format("F_score: ") +  "{:.3f}".format(f1_re)
    result_scores = heading + "\n\n" + precision + "\n" + recall + "\n" + f_score
    return result_scores