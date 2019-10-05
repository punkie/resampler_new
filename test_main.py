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
    print (confusion_matrix(y_test, predicted_y_s))
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
    # ts = fetch_datasets()['thyroid_sick']
    # pca = PCA(n_components=2)
    #pca.fit_transform(ts['x_values'])
    #load_different_classes()
if __name__ == "__main__":
    #load_different_classes()
    #load_and_print_ds()
    #load_transformed_ds()
    logistic_regression()
    #print_examples()