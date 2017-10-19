from collections import Counter

import numpy as np
from scipy import interp
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold

from functions import resampling_functions


def compute_some_statistics_for_the_dataset(dataset):
    dataset['number_of_examples'] = len(dataset['y_values'])
    unique_target_values = Counter(dataset['y_values'])
    first_mc_tuple, second_mc_tuple = unique_target_values.most_common(2)
    dataset['y_values_as_set'] = (first_mc_tuple[0], second_mc_tuple[0])
    dataset['target_class_percentage'] = second_mc_tuple[1] / first_mc_tuple[1]
    __create_statistics_string(dataset)


def classify(classifier_thread):
    classifying_data = {}
    if classifier_thread.do_resampling:
        classifying_data['figure_number'] = 2
    else:
        classifying_data['figure_number'] = 1
    classifying_data['main_tuples'] = []
    normal_dataset = classifier_thread.main_window.state.dataset
    #resampled_dataset = state.resampled_dataset
    random_state = np.random.RandomState(0)
    cv = StratifiedKFold(n_splits=10)
    classifier = RandomForestClassifier()
    # classifier = svm.LinearSVC(random_state=random_state)
    # classifier = tree.DecisionTreeClassifier(criterion="entropy")
    X_normal = normal_dataset['x_values']
    y_normal = normal_dataset['y_values']

    #X_resampled = resampled_dataset['x_values']
    #y_resampled = resampled_dataset['y_values']

    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    i = 0
    for train, test in cv.split(X_normal, y_normal):
        if classifier_thread.do_resampling:
            r_dataset = resampling_functions.\
            do_resampling_without_writing_to_file(classifier_thread.main_window.state.sampling_algorithm, X_normal[train], y_normal[train])
            classifier_ = classifier.fit(r_dataset['x_values'], r_dataset['y_values'])
        else:
            classifier_ = classifier.fit(X_normal[train], y_normal[train])
        #predicted_y_scores = classifier_.decision_function(X_normal[test])
        probas_ = classifier_.predict_proba(X_normal[test])
        average_precision = average_precision_score(y_normal[test], probas_[:, 1])
        fpr, tpr, thresholds = roc_curve(y_normal[test], probas_[:, 1])
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        classifying_data['main_tuples'].append((fpr, tpr, roc_auc, i, y_normal[test], probas_[:, 1], average_precision))

        i += 1
        if classifier_thread.do_resampling:
            classifier_thread.update_resample_classify_progress_bar.emit(i)
        else:
            classifier_thread.update_normal_classify_progress_bar.emit(i)

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
    dataset_statistics_string = 'Number of examples: {}\nPercentage of minor class examples: {}'.format(
        dataset['number_of_examples'], dataset['target_class_percentage'])
    dataset['dataset_statistics_string'] = dataset_statistics_string