
# imported library
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('train.csv')
dataset1 = pd.read_csv('test.csv')
dataset3 = pd.read_csv('gender_submission.csv')
#checking nan in columbs of dataset
dataset.isnull().sum()
dataset1.isnull().sum()
#checking the datatype of series in dataframes
dataset.dtypes
#checking shape
dataset.shape
dataset.head(30)
dataset.describe()
#visualing the data
dataset.plot(kind='box', subplots=True, layout=(2,7), sharex=False, sharey=False)
plt.show()

dataset.hist()
plt.show()
#making traing nad test datset
X_train= dataset.iloc[:,[2,4,5,6,7,9]].values
y_train= dataset.iloc[:,1:2].values

X_test = dataset1.iloc[:,[1,3,4,5,6,8]].values
y_test = dataset3.iloc[:,1:2].values
# filling the NAN values withh mean 
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values="NaN",strategy="mean",axis=0)
imputer = imputer.fit(X_train[:, 2:3])
X_train[:, 2:3] = imputer.transform(X_train[:,2:3])

imputer = Imputer(missing_values="NaN",strategy="mean",axis=0)
imputer = imputer.fit(X_test[:, [2,5]])
X_test[:, [2,5]] = imputer.transform(X_test[:,[2,5]])

# making the dummy varaible of catagorical data


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X_train[:,1] = labelencoder_X.fit_transform(X_train[:, 1])
onehotencoder = OneHotEncoder(categorical_features=[0])
X_train = onehotencoder.fit_transform(X_train).toarray()

labelencoder_X1 = LabelEncoder()
X_test[:,1] = labelencoder_X1.fit_transform(X_test[:, 1])
onehotencoder1 = OneHotEncoder(categorical_features=[0])
X_test = onehotencoder1.fit_transform(X_test).toarray()

# performing standard scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)

# performng PCA 
from sklearn.decomposition import PCA
pca = PCA(n_components = None)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
explained_variance = pca.explained_variance_ratio_

#importing model
from sklearn.linear_model import LinearRegression
#maiing object
regressor = LinearRegression()
#trainnig the model
regressor.fit(X_train, y_train)
#predicting the model on test data ste
y_pred = regressor.predict(X_test)
y_pred = y_pred > 0.5
# confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
l = dataset3.iloc[:,0].values

y_pred[1]

sub =[]
for i in range(len(y_pred)):
    if(y_pred[i]==False):
        sub.append(0)
    else:
        sub.append(1)
        
j = np.asarray(sub)
submission=pd.DataFrame({
    "PassengerId":dataset1['PassengerId']
})
submission["Survived"]=sub
submission.info()
submission.to_csv('submission.csv',index=False)

# accuracy of 97.84%


