#!/usr/bin/env python
# coding: utf-8

# In[33]:


import numpy as np
import pandas as pd
import tensorflow as tf
import keras
import csv
import os.path

import matplotlib.pyplot as plt 
#from IPython import get_ipython
#get_ipython().run_line_magic('matplotlib', 'inline')

import seaborn as sns
#sns.set(palette="Set2")

from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, f1_score,average_precision_score, confusion_matrix,
                             average_precision_score, precision_score, recall_score, roc_auc_score, )
from mlxtend.plotting import plot_confusion_matrix

from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler


from xgboost import XGBClassifier, plot_importance
from imblearn.over_sampling import SMOTE

userhome = os.path.expanduser('~')
csvfile= os.path.join(userhome, 'Downloads', 'Churn_Modelling.csv')
#filedata = open(csvfile, "r")

dataset = pd.read_csv(csvfile)
print(dataset.head())
#print(dataset.describe())
#print(dataset.info())

####Removing unwanted columns####

dataset.drop(["RowNumber","CustomerId","Surname"], axis=1, inplace=True)
#dataset.drop([10000],axis = 0, inplace=True)
#print(len(dataset.dtypes))

####Data Visualization starts####
#iris = sns.load_dataset("dataset")
#dataViz = sns.load_dataset("dataset")
#y = dataset.Exited
#X = dataset.drop('Exited',axis=1)
#sns.pairplot(dataset, hue="Exited",palette="bright")
print("Countplot")
_, ax = plt.subplots(1, 3, figsize=(18, 6))
plt.subplots_adjust(wspace=0.3)
sns.countplot(x = "NumOfProducts", hue="Exited", data = dataset, ax= ax[0])
sns.countplot(x = "HasCrCard", hue="Exited", data = dataset, ax = ax[1])
sns.countplot(x = "IsActiveMember", hue="Exited", data = dataset, ax = ax[2])

#sns.countplot(dataset, hue="Exited",palette="bright")
print("swarmplot")
_, ax = plt.subplots(1, 3, figsize=(18, 6))
plt.subplots_adjust(wspace=0.3)
sns.swarmplot(x = "NumOfProducts", y = "Age", hue="Exited", data = dataset, ax= ax[0])
sns.swarmplot(x = "HasCrCard", y = "Age", data = dataset, hue="Exited", ax = ax[1])
###Data visualization Ends####


#dataset["Geography"] = dataset["Geography"].astype(str)
#dataset["Gender"] = dataset["Gender"].astype(str)
#print(dataset.head())
#print(dataset.info())
encoder = LabelEncoder()
#encoder.fit(dataset["Country"].astype(str))
#encoder.fit(dataset["Gender"].astype(str))
dataset["Geography"] = encoder.fit_transform(dataset["Geography"])
dataset["Gender"] = encoder.fit_transform(dataset["Gender"])

#labelencoder = LabelEncoder()
#dataset[:, 0] = labelencoder.fit_transform(dataset[:, 0])
#print(dataset)



#####Predicting with ML models####
X = dataset.drop("Exited", axis=1)
y = dataset["Exited"]
#print("This is X:")
#print(X)
#print()
#print("This is y:")
#print(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=0)
##print(X_train, X_test, y_train, y_test)

clf = GaussianNB()
#print(clf)
clf.fit(X_train, y_train)
#print(X_train, y_train)
pred = clf.predict(X_test)
accuracy_score(pred, y_test)
print("Naive Bayes: ")
print(accuracy_score(pred, y_test))

clf = LogisticRegression()
clf.fit(X_train, y_train)
pred = clf.predict(X_test)
accuracy_score(pred, y_test)
print("LogisticRegression: ")
print(accuracy_score(pred, y_test))

clf = tree.DecisionTreeClassifier()
clf.fit(X_train, y_train)
pred = clf.predict(X_test)
accuracy_score(pred, y_test)
print("Decision Tree: ")
print(accuracy_score(pred, y_test))

clf = RandomForestClassifier(n_estimators = 200, random_state=200)
clf.fit(X_train, y_train)
pred = clf.predict(X_test)
accuracy_score(pred, y_test)
print("RandForest: ")
print(accuracy_score(pred, y_test))

clf  = XGBClassifier(max_depth = 10,random_state = 10, n_estimators=220, eval_metric = 'auc', min_child_weight = 3, colsample_bytree = 0.75, subsample= 0.9)
clf.fit(X_train, y_train)
pred = clf.predict(X_test)
accuracy_score(pred, y_test)
print("XGB: ")
print(accuracy_score(pred, y_test))

scaler = MinMaxScaler()
bumpy_features = ["CreditScore", "Age", "Balance",'EstimatedSalary']
df_scaled = pd.DataFrame(data = X)
df_scaled[bumpy_features] = scaler.fit_transform(X[bumpy_features])
df_scaled.head()


#svclassifier = SVC(kernel='linear')
#svclassifier = SVC(kernel='poly', degree=10)
svclassifier = SVC(kernel='rbf')
svclassifier.fit(X_train, y_train)
print("yes here")
pred = svclassifier.predict(X_test)
print("SVM Accuracy")
print(accuracy_score(pred, y_test))
print("Confusion Matrix for SVM")
#print(confusion_matrix(y_test,pred))
#print(classification_report(y_test,pred))



X = df_scaled
sm  = SMOTE(random_state=42)
X_res, y_res = sm.fit_sample(X, y)
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size= 0.2, random_state=7)

clf = XGBClassifier(max_depth = 12,random_state=7, n_estimators=100, eval_metric = 'auc', min_child_weight = 3,
                    colsample_bytree = 0.75, subsample= 0.8)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1:", f1_score(y_test, y_pred))
print("Area under precision (AUC) Recall:", average_precision_score(y_test, y_pred))


#Confusion Matrix
confusion_matrix(y_test, y_pred)
print("Confusion Matrix: ")
print(confusion_matrix(y_test, y_pred))


# In[34]:


from yellowbrick.classifier import ClassPredictionError


# In[35]:


classes = ['Exited', 'Not Exited']
clf = RandomForestClassifier(n_estimators = 200, random_state=200)
visualizer = ClassPredictionError(clf)
visualizer.fit(X_train, y_train)
visualizer.score(X_test,y_test)
visualizer.show()


# In[36]:


svclassifier = SVC(kernel='rbf')
visualizer = ClassPredictionError(svclassifier)
visualizer.fit(X_train, y_train)
visualizer.score(X_test,y_test)
visualizer.show()


# In[10]:


clf = tree.DecisionTreeClassifier()
visualizer = ClassPredictionError(clf)
visualizer.fit(X_train, y_train)
visualizer.score(X_test,y_test)
visualizer.show()


# In[21]:


clf = LogisticRegression()
visualizer = ClassPredictionError(clf)
visualizer.fit(X_train, y_train)
visualizer.score(X_test,y_test)
visualizer.show()


# In[17]:


clf = GaussianNB()
visualizer = ClassPredictionError(clf)
visualizer.fit(X_train, y_train)
visualizer.score(X_test,y_test)
visualizer.show()


# In[19]:


from yellowbrick.classifier import ClassificationReport
model = GaussianNB()
visualizer = ClassificationReport(model, support=True)

visualizer.fit(X_train, y_train)        # Fit the visualizer and the model
visualizer.score(X_test, y_test)        # Evaluate the model on the test data
visualizer.show()  


# In[20]:


from yellowbrick.classifier import PrecisionRecallCurve
visualizer = PrecisionRecallCurve(GaussianNB())
visualizer.fit(X_train, y_train)        # Fit the visualizer and the model
visualizer.score(X_test, y_test)        # Evaluate the model on the test data
visualizer.show() 


# In[22]:


clf = LogisticRegression()
visualizer = PrecisionRecallCurve(clf)
visualizer.fit(X_train, y_train)
visualizer.score(X_test,y_test)
visualizer.show()


# In[24]:





# In[25]:


svclassifier = SVC(kernel='rbf')
visualizer = PrecisionRecallCurve(svclassifier)
visualizer.fit(X_train, y_train)
visualizer.score(X_test,y_test)
visualizer.show()


# In[43]:


clf = tree.DecisionTreeClassifier()
visualizer = PrecisionRecallCurve(clf)
visualizer.fit(X_train, y_train)
visualizer.score(X_test,y_test)
visualizer.show()


# In[ ]:




