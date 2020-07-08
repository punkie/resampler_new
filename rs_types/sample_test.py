import matplotlib.pyplot as plt
import pandas as pd
import time
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import precision_recall_curveir, roc_curve
from sklearn.preprocessing import label_binarize
from sklearn.datasets import load_digits

# df = pd.DataFrame({
#     'name':['john','mary','peter','jeff','bill','lisa','jose'],
#     'age':[23,78,22,19,45,33,20],
#     'gender':['M','F','M','M','M','F','M'],
#     'state':['california','dc','california','dc','california','texas','texas'],
#     'num_children':[2,0,0,3,2,1,4],
#     'num_pets':[5,1,0,5,2,2,3]
# })
# df.plot(kind='scatter',x='num_children',y='num_pets',color='red')
# plt.show()
# ts = ts.cumsum()
#ts.plot()
print ("x")
df = pd.DataFrame(np.random.rand(50, 4), columns=['a', 'b', 'c', 'd'])
df.plot(x='a', y='b')
plt.show()
# time.sleep(1000)