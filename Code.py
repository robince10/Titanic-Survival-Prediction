import pandas as pd
import matplotlib.pyplot as plt

#Importing Data
df = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')

#Data Preprocessing  
df.isna().sum()
df_test.isna().sum()
df['Embarked'].value_counts()
df['Embarked'].fillna('S',inplace = True)
df_test['Embarked'].value_counts()
df_test['Embarked'].fillna('S',inplace = True)
df['Age'].fillna(df['Age'].mean(),inplace = True)
df_test['Age'].fillna(df_test['Age'].mean(),inplace = True)
df['Fare'].max()
df['Fare'].min()
plt.boxplot(df['Fare'])
bins = [0,20,100,700]
groups = ['LF','MF','HF']
df['Fare_Class'] = pd.cut(df['Fare'],bins,labels = groups)
df_test['Fare_Class'] = pd.cut(df_test['Fare'],bins,labels = groups)
plt.boxplot(df['Age'])
bins_2 = [0,15,50,100]
groups_2 = ['Children', 'Adults', 'Old']
df['Age_dist'] = pd.cut(df['Age'],bins_2,labels = groups_2)
df_test['Age_dist'] = pd.cut(df_test['Age'],bins_2,labels = groups_2)
df.isna().sum()
df['Fare_Class'].value_counts()
df['Fare_Class'].fillna(method = 'ffill', inplace = True)
df_test['Fare_Class'].fillna(method = 'ffill', inplace = True)
X = df.drop(['PassengerId','Name','Age','Ticket','Fare','Cabin','Survived','Embarked'],axis = 1)
X_test = df_test.drop(['PassengerId','Name','Age','Ticket','Fare','Cabin','Embarked'],axis = 1)
df['SibSp'].value_counts()
Y = df['Survived']
X_new = X.append(X_test)
X_new = pd.get_dummies(X_new)
X_new = X_new.drop(['Sex_female','Fare_Class_LF','Age_dist_Children'],axis = 1)
X = X_new[:891]
X_test = X_new[891:]
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
X = ss.fit_transform(X)
X_test = ss.transform(X_test)

#Training the model
from sklearn.svm import SVC
classifier = SVC(kernel = 'gaussian', random_state = 0)
classifier.fit(X, Y)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y,classifier.predict(X))
type(df['Fare_Class'])
