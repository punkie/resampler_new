import numpy as np
from enum import Enum
from sklearn import svm, tree
from sklearn.ensemble import RandomForestClassifier


class ClassificationAlgorithms(Enum):

    CART = ("CART (DecisionTree)", tree.DecisionTreeClassifier(criterion='entropy',
                                                               random_state=np.random.RandomState(1)))
    SVM = ("SVM", svm.SVC(probability=True, random_state=np.random.RandomState(1)))
    RANDOMFOREST = ("RandomForest", RandomForestClassifier(n_estimators=100, criterion='entropy',
                                                           random_state=np.random.RandomState(1)))

    @classmethod
    def get_algorithm_by_name(cls, name):
        filtered_algos = filter(lambda ca: ca.value[0] == name, ClassificationAlgorithms)
        return list(filtered_algos).pop()
