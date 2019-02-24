import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split

mr=pd.read_csv('./mushroom/mushroom.csv',header=None)
label=[]
data=[]
attr_list=[]
for row_index, row in mr.iterrows():
    label.append(row.ix[0])
    row_data=[]
    for v in row.ix[1:]:
        row_data.append(ord(v))
    data.append(row_data)

data_train, data_test, label_train, label_test=train_test_split(data, label)
clf=RandomForestClassifier()
clf.fit(data_train, label_train)
pre=clf.predict(data_test)
score=metrics.accuracy_score(label_test,pre)
report=metrics.classification_report(label_test,pre)
print(score)
print(report)