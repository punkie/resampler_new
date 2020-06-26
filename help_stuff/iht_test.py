import numpy as np
from collections import Counter
from sklearn.datasets import make_classification
from imblearn.under_sampling import InstanceHardnessThreshold

def convert_neg_class(target_class):
    if target_class == 0:
        return -1
    else:
        return target_class

def sample_func():
    X, y = make_classification(n_classes=2, class_sep=2,
    weights=[0.1, 0.9], n_informative=3, n_redundant=1, flip_y=0,
    n_features=20, n_clusters_per_class=1, n_samples=1000, random_state=10)
    print('Original dataset shape %s' % Counter(y))
    cnv_vec = np.vectorize(convert_neg_class)
    y = cnv_vec(y)

    iht = InstanceHardnessThreshold(random_state=42)
    X_res, y_res = iht.fit_resample(X, y)
    print('Resampled dataset shape %s' % Counter(y_res))  # doctest: +ELLIPSIS

if __name__ == '__main__':
    sample_func()