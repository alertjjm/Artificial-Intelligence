from sklearn import svm,metrics
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd

tbl=pd.read_csv("./bmi/bmi.csv")
label=tbl["label"]
height=tbl["height"]/200
weight=tbl["weight"]/100
wh=pd.concat([weight,height],axis=1)
data_train,data_test,label_train,label_test=train_test_split(wh,label)
clf=svm.SVC()
clf.fit(data_train,label_train)
predict=clf.predict(data_test)
score=metrics.accuracy_score(label_test,predict)
report=metrics.classification_report(label_test,predict)
print(score)
print(report)
tbl=pd.read_csv("./bmi/bmi.csv",index_col=2)# index역할을 하는 column이 label, tbl[2]라는 뜻이구나
fig=plt.figure()
ax=fig.add_subplot(1,1,1)
def scatter(lbl,color):
    b=tbl.loc[lbl]
    print(b)
    ax.scatter(b["weight"],b["height"],c=color,label=lbl)
scatter("fat","red")
scatter("normal","yellow")
scatter("thin","purple")

ax.legend()
plt.savefig("./bmi/bmi-test.png")