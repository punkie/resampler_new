from random import sample, shuffle

import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from imblearn.datasets import fetch_datasets
from sklearn.datasets import make_classification
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, LinearRegression, Perceptron
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.multiclass import unique_labels


def print_examples():
    ts = fetch_datasets()['thyroid_sick']
    print(ts.data.shape)
    target_classes = sorted(Counter(ts.target).items())
    print(sorted(Counter(ts.target).items()))
    ds = load_ds()
    labels = ['Целеви класове']
    healty, sick = ([len(list(filter(lambda x: x[-1] == 0, ds)))], [len(list(filter(lambda x: x[-1] == 1, ds)))])
    # healty = [target_classes[0][1]]
    # sick = [target_classes[1][1]]

    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width / 2, healty, width, label='Здрави')
    rects2 = ax.bar(x + width / 2, sick, width, label='Болни')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Брой примери')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)

    fig.tight_layout()

    plt.show()

def load_ds(path):
    df = pd.read_csv(path, sep=',', header=None)
    values = df.values
    return values


def load_transformed_ds(path, subset, negative_samples, positive_samples):
    df = load_ds(path)
    if subset:
        Xy_sub_0 = sample(list(filter(lambda x: x[-1] == 0, df)), negative_samples)
        Xy_sub_1 = sample(list(filter(lambda x: x[-1] == 1, df)), positive_samples)
        df = Xy_sub_0 + Xy_sub_1
        shuffle(df)
    X, y = (list(map(lambda x: x[:-1], df)), list(map(lambda x: x[-1], df)))
    #X_filtered = [[x[0], x[-5]] for x in X]
    return (np.array(X), np.array(y));


def logistic_regression():
    # X_test, y_test = load_transformed_ds('./binarized-datasets/thyroid_sic_2_test.data', True, 300, 73)
    X, y = make_classification(1000, 2, 2, 0, weights=[.99, .01], random_state=15)
    X_test, y_test = make_classification(1000, 2, 2, 0, weights=[.99, .01], random_state=15)
    # X_tr, y_tr = load_transformed_ds('./binarized-datasets/thyroid_sic_2.data', True, 300, 73)
    pca = PCA(n_components=2)
    # X = pca.fit_transform(X)
    clf = LogisticRegression().fit(X, y)

    xx, yy = np.mgrid[-5:5:.01, -5:5:.01]
    grid = np.c_[xx.ravel(), yy.ravel()]
    predicted_y_s = clf.predict(X_test)
    print (accuracy_score(y_test, predicted_y_s))
    cm = confusion_matrix(y_test, predicted_y_s)

    probs = clf.predict_proba(grid)[:, 1].reshape(xx.shape)

    f, ax = plt.subplots(figsize=(8, 6))
    cont = ax.contour(xx, yy, probs, levels=[.5], cmap="Greys", vmin=0, vmax=.6)
    scatt1 = ax.scatter(X_test[y_test == 0, 0], X_test[y_test == 0, 1], c=np.array(list(filter(lambda x : x == 0, y_test))), s=100,
               cmap="RdBu", vmin=-.2, vmax=1.2, marker="x", label="0",
               edgecolor="white", linewidth=1)
    scatt2 = ax.scatter(X_test[y_test == 1, 0], X_test[y_test == 1, 1], c=np.array(list(filter(lambda x: x == 1, y_test))),
               s=100,
               cmap="RdBu", vmin=-.2, vmax=1.2, marker="o", label="1",
               edgecolor="white", linewidth=1)

    h1,l1 = cont.legend_elements()
    h2,l2 = scatt1.legend_elements()
    h3,l3 = scatt2.legend_elements()

#    legend = ax.legend(*scatter.legend_elements(), loc="upper right", title="primeri");
    ax.legend(h1 + h2 + h3, ["P(y=1)"] + ["y=0"]  + ["y=1"])
    # ax.add_artist(legend)

    #ax.grid(True)
    ax.set(aspect="equal",
           xlim=(-5, 5), ylim=(-5, 5),
           xlabel="$X_1$", ylabel="$X_2$")
    plt.show()
    return cm
    # ts = fetch_datasets()['thyroid_sick']
    # pca = PCA(n_components=2)
    #pca.fit_transform(ts['x_values'])
    #load_different_classes()


def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=False):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(10, 2))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontdict={'size': 22})
    # plt.colorbar()

    if target_names is not None:
        # tick_marks = np.array(len(target_names))
        tick_marks = np.arange(len(target_names))
        plt.yticks(tick_marks, target_names, fontsize=22)
        plt.xticks(tick_marks, target_names, fontsize=22)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center", fontdict={'size': 22},
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center", fontdict={'size': 22},
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('Истинска стойност', fontdict={'size': 22})
    plt.xlabel('Предсказана стойност\n\nточност={:0.4f}'.format(accuracy), fontdict={'size': 22})
    plt.show()

if __name__ == "__main__":
    #np.set_printoptions(precision=2)
    #load_different_classes()
    #load_and_print_ds()
    #load_transformed_ds()
    cm = logistic_regression()
    plot_confusion_matrix(cm,
                          normalize=False,
                          target_names=['0', '1'],
                          title="Confusion Matrix")
    #print_examples()