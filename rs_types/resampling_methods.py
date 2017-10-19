from enum import Enum

from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler


class ResamplingAlgorithms(Enum):
    RO = ("Random Oversampling", RandomOverSampler())
    RU = ("Random Undersampling", RandomUnderSampler())
    SMOTE = ("Smote", SMOTE(kind="regular"))

    @classmethod
    def get_algorithm_by_name(cls, name):
        filtered_algos = filter(lambda ra: ra.value[0] == name, ResamplingAlgorithms)
        return list(filtered_algos).pop()
