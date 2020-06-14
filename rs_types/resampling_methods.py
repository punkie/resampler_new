from enum import Enum
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler, TomekLinks, CondensedNearestNeighbour, OneSidedSelection, \
    NearMiss
from imblearn.under_sampling import ClusterCentroids


class ResamplingAlgorithms(Enum):
    RO = ("Random Oversampling", RandomOverSampler(random_state=1))
    RU = ("Random Undersampling", RandomUnderSampler(random_state=1))
    SMOTE = ("Smote", SMOTE(random_state=1))
    CLUSTERCENTROIDS = ("ClusterCentroids", ClusterCentroids(random_state=1))
    TOMEK_LINKS = ("TomekLinks", TomekLinks())
    amount = ("NM1", NearMiss(version=1))
    NM2 = ("NM2", NearMiss(version=2))
    NM3 = ("NM3", NearMiss(version=3))
    CNN = ("CNN", CondensedNearestNeighbour(random_state=1))
    OSS = ("OneSidedSelection", OneSidedSelection(random_state=1))
    # EditedNearestNeighbours = ("EditedNearestNeighbours", EditedNearestNeighbours())
    # ADASYN = ("Adasyn", ADASYN())

    @classmethod
    def get_algorithm_by_name(cls, name):
        filtered_algos = filter(lambda ra: ra.value[0] == name, ResamplingAlgorithms)
        return next(filtered_algos, ResamplingAlgorithms.RO)
