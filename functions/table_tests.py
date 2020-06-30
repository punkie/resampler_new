import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import DataFrame
from scipy import rand

fig, ax = plt.subplots(1, 1)
fig.patch.set_visible(False)
ax.axis('off')
ax.axis('tight')


# s = pd.Series(['Class. Alg.', 'Balanced Accuracy', 'Precision', 'Recall', 'F1', 'G1', 'G2', 'AUC', 'AUC_roc'], index=index)

a = np.array(['Class. Alg.', 'Balanced Accuracy', 'Precision', 'Recall', 'F1', 'G1', 'G2', 'AUC_roc', 'AUC_pr'])
b = np.array(range(1, 10))
c = np.array(range(1, 10))

res = np.vstack((a, b, c)).T

df = pd.DataFrame(res, columns=[' ', 'NormalCase', 'ResampledCase'])

ax.table(cellText=df.values, colLabels=df.columns, loc='center')

fig.tight_layout()

plt.show()