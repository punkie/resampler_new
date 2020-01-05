import matplotlib.pyplot as plt
from scikitplot.classifiers import plot_precision_recall_curve_with_cv
from sklearn.datasets import make_classification
from sklearn.datasets import load_digits as load_data
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB


def test():
    X, y = load_data(return_X_y=True)

    nb = GaussianNB()
    nb.fit(X, y)
    probas = nb.predict_proba(X)
    plot_precision_recall_curve_with_cv(nb, X, y)
    plt.show()

if __name__ == "__main__":
    test()