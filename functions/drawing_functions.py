from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import precision_recall_curve


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
    classified_datas = [state.classified_data_normal_case, state.classified_data_resampled_case]
    f, (ax1, ax2) = plt.subplots(1, 2)
    for x in range(2):
        mean_fpr, mean_tpr, mean_auc, std_auc, tprs_lower, tprs_upper = classified_datas[x]['mean_values_tuple']
        ax = ax1 if x == 0 else ax2
        ax.get_figure().set_size_inches(12, 9)
        for mt in classified_datas[x]['main_tuples']:
            fpr, tpr, roc_auc, i, y_test_from_normal_ds, predicted_y_scores, average_precision = mt
            #plt.figure(normal_classifying_data['figure_number'])
            ax.plot(fpr, tpr, lw=1, alpha=0.3,
                label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
        ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
                label='Luck', alpha=.8)
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
            ax.set_title('ROC for the resampled dataset\n (using {} algorithm)'.format(state.sampling_algorithm.value[0]))
    #plt.legend(loc="lower right")
    plt.show()


def draw_pr_graph(state):
    classified_datas = [state.classified_data_normal_case, state.classified_data_resampled_case]
    f, (ax1, ax2) = plt.subplots(1, 2)
    for x in range(2):
        # currently drawing only for the first fold
        ax = ax1 if x == 0 else ax2
        ax.get_figure().set_size_inches(12, 9)
        fpr, tpr, roc_auc, i, y_test_from_normal_ds, predicted_y_scores, average_precision = classified_datas[x]['main_tuples'][x]
        precision, recall, _ = precision_recall_curve(y_test_from_normal_ds, predicted_y_scores)
        ax.step(recall, precision, color='b', alpha=0.2, where='post')
        ax.set_title('2-class Precision-Recall curve \nfor the 1st Fold of the Cross-validation \non {0}: AUC={1:0.2f}'.format(
            "normal dataset" if x == 0 else "resampled dataset",
            average_precision))
        ax.fill_between(recall, precision, step='post', alpha=0.2,
                         color='b')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_ylim([0.0, 1.05])
        ax.set_xlim([0.0, 1.0])
    plt.show()