import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

import resampling_functions
from collections import Counter, deque
from PyQt5.QtWidgets import QProgressBar
from imblearn.datasets import fetch_datasets
from sklearn.decomposition import PCA
from sklearn import svm
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
from scipy import interp


def write_dataset_to_csv(path, dataset=None, x_values_param=None, y_values_param=None):
    if dataset is not None:
        x_values_param = dataset['data']
        y_values_param = dataset['target']
    with open(path, "w", newline="\n") as csv_output_file:
        dataset_writer = csv.writer(csv_output_file, delimiter=",")
        for row_idx, dv in enumerate(x_values_param):
            dataset_writer.writerow(np.append(dv, y_values_param[row_idx]))


def load_dataset(dataset_loader):
    with open(dataset_loader.path, newline="") as csv_input_file:
        try:
            reader = csv.reader(csv_input_file, delimiter=",")
            reader_as_list = list(reader)
            dataset = {}
            dataset['name'] = dataset_loader.path.split("/")[-1].split(".csv")[0]
            x_values = deque()
            y_values = deque()
            count = 0
            dataset_loader.main_window.findChild(QProgressBar, "datasetProgressBar").setMaximum(len(reader_as_list))
            for row in reader_as_list:
                row_floats = list(map(float, row))
                count += 1
                x_values.append(row_floats[:-1])
                y_values.append(row_floats[-1])
                dataset_loader.update_dataset_load_progress_bar.emit(count)
            dataset['x_values'] = np.array(x_values)
            dataset['y_values'] = np.array(y_values)
            dataset_loader.main_window.state.dataset = dataset
        except Exception as e:
            print(e)


def compute_some_statistics_for_the_dataset(dataset):
    dataset['number_of_examples'] = len(dataset['y_values'])
    unique_target_values = Counter(dataset['y_values'])
    first_mc_tuple, second_mc_tuple = unique_target_values.most_common(2)
    dataset['y_values_as_set'] = (first_mc_tuple[0], second_mc_tuple[0])
    dataset['target_class_percentage'] = second_mc_tuple[1] / first_mc_tuple[1]


def get_statistics_string(dataset):
    dataset_statistics = 'Number of examples: {}\nPercentage of minor class examples: {}'.format(
        dataset['number_of_examples'], dataset['target_class_percentage'])
    return dataset_statistics


def classify(classifier_thread, with_resampling=False):
    try:
        classifying_data = {}
        if with_resampling:
            classifying_data['figure_number'] = 2
        else:
            classifying_data['figure_number'] = 1
        classifying_data['main_tuples'] = []
        normal_dataset = classifier_thread.main_window.state.dataset
        #resampled_dataset = state.resampled_dataset
        random_state = np.random.RandomState(0)
        cv = StratifiedKFold(n_splits=10)
        classifier = RandomForestClassifier(n_estimators=100)
        X_normal = normal_dataset['x_values']
        y_normal = normal_dataset['y_values']

        #X_resampled = resampled_dataset['x_values']
        #y_resampled = resampled_dataset['y_values']

        tprs = []
        aucs = []
        mean_fpr = np.linspace(0, 1, 100)
        i = 0
        for train, test in cv.split(X_normal, y_normal):
            if with_resampling:
                r_dataset = resampling_functions.\
                do_resampling_without_writing_to_file(classifier_thread.main_window.state.sampling_algorithm, X_normal[train], y_normal[train])
                classifier_ = classifier.fit(r_dataset['x_values'], r_dataset['y_values'])
            else:
                classifier_ = classifier.fit(X_normal[train], y_normal[train])
            probas_ = classifier_.predict_proba(X_normal[test])
            fpr, tpr, thresholds = roc_curve(y_normal[test], probas_[:, 1])
            tprs.append(interp(mean_fpr, fpr, tpr))
            tprs[-1][0] = 0.0
            roc_auc = auc(fpr, tpr)
            aucs.append(roc_auc)
            classifying_data['main_tuples'].append((fpr, tpr, roc_auc, i))
            # to move
            # plt.plot(fpr, tpr, lw=1, alpha=0.3,
            #          label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

            i += 1
            classifier_thread.update_classify_progress_bar.emit(i)
        # to move
        # plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
        #          label='Luck', alpha=.8)

        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
        # to move
        # plt.plot(mean_fpr, mean_tpr, color='b',
        #          label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
        #          lw=2, alpha=.8)

        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        # to move
        # plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
        #                  label=r'$\pm$ 1 std. dev.')
        classifying_data['mean_values_tuple'] = (mean_fpr, mean_tpr, mean_auc, std_auc, tprs_lower, tprs_upper)
        # to move
        # plt.xlim([-0.05, 1.05])
        # plt.ylim([-0.05, 1.05])
        # plt.xlabel('False Positive Rate')
        # plt.ylabel('True Positive Rate')
        # plt.title('Receiver operating characteristic example')
        # plt.legend(loc="lower right")
        # plt.show()
        return classifying_data
    except Exception as e:
        print (e)


def draw_comparision_picture(normal_dataset, resampled_dataset, sampling_algo_name):
    pca = PCA(n_components=2)
    x_visible = pca.fit_transform(normal_dataset['x_values'])
    x_resampled_vis = pca.transform(resampled_dataset['x_values'])

    y_values = normal_dataset['y_values']
    y_values_as_set = normal_dataset['y_values_as_set']

    y_resampled_values = resampled_dataset['y_values']

    tc_one, tc_two = y_values_as_set

    f, (ax1, ax2) = plt.subplots(1, 2)
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

    # make nice plotting
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


def extract_binarized_imbalanced_datasets():
    for dataset_name, dataset_values in fetch_datasets().items():
        write_dataset_to_csv("./binarized-datasets/" + dataset_name + ".csv", dataset_values)

