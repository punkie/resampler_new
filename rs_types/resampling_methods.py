from enum import Enum
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler


class ResamplingAlgorithms(Enum):
    RO = ("Random Oversampling", RandomOverSampler(random_state=1))
    RU = ("Random Undersampling", RandomUnderSampler(random_state=1))
    SMOTE = ("Smote", SMOTE(kind="regular", random_state=1))
    # ADASYN = ("Adasyn", ADASYN())

    @classmethod
    def get_algorithm_by_name(cls, name):
        filtered_algos = filter(lambda ra: ra.value[0] == name, ResamplingAlgorithms)
        return list(filtered_algos).pop()
