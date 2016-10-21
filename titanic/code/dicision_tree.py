from sklearn import tree
import sys
import pandas as pd
from pandas import DataFrame, Series
import numpy as np
import collections as cl

pd.set_option('display.expand_frame_repr',False)
pd.set_option('display.max_colwidth',10)

split_num = 5
features_columns = ['Sex','Age','Fare']
target_columns = ['Survived']

or_data = np.array_split(pd.read_csv("/Users/x/code/kaggle/titanic/data/train.csv").fillna(-1)[['Survived','Sex','Age','Fare']],split_num)
train_data = or_data[0]
for i in range(1,split_num-1):
    train_data.append(or_data[i], ignore_index=True)
test_data = or_data[split_num-1]

train_features = train_data[features_columns]
train_features.Sex=train_features.Sex.map({'male':0, 'female':1 ,np.nan:-1})
train_features.Age=train_features.Age.map(lambda l: -1 if l == np.nan else int(l))
train_features.Fare=train_features.Fare.map(lambda l:-1 if l == np.nan else float(l))
train_target = train_data[target_columns]

clf = tree.DecisionTreeClassifier()
clf = clf.fit(train_features, train_target)


# test_data = pd.read_csv("/Users/x/code/kaggle/titanic/data/test.csv")[['Survived','Sex','Age','Fare']]
test_features  =test_data[features_columns]
test_target= test_data[target_columns]
test_features.Sex=test_features.Sex.map({'male':0, 'female':1 ,np.nan:-1})
test_features.Age=test_features.Age.map(lambda l: -1 if l == np.nan else int(l))
test_features.Fare=test_features.Fare.map(lambda l:-1 if l == np.nan else float(l))
rst = clf.predict(test_features)

import pdb; pdb.set_trace()
diff = rst - test_target[target_columns].tolist()

cl.Counter(diff)
