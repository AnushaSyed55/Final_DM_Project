#!/usr/bin/env python
# coding: utf-8

# # Anusha's Final Term Project
# ->KNN,SVM,Random Forest and LSTM 
# 

# In[7]:


# Importing the important libraries

import numpy as np
import random
import pandas as pd

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler , MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report , confusion_matrix, accuracy_score, f1_score

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier # Random Forest Classifier
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV

from keras.models import Sequential



import warnings
warnings.filterwarnings('ignore')


# In[9]:


# Loading the dataset

df = pd.read_csv('heart.csv')
df.head()


# In[10]:


from sklearn import preprocessing
 
# label_encoder object knows how to understand word labels.
label_encoder = preprocessing.LabelEncoder()
 
# Encode labels in column 'species'.
df['Sex']= label_encoder.fit_transform(df['Sex'])
df['ChestPainType']= label_encoder.fit_transform(df['ChestPainType'])
df['RestingECG']= label_encoder.fit_transform(df['RestingECG'])
df['ExerciseAngina']= label_encoder.fit_transform(df['ExerciseAngina'])
df['ST_Slope']= label_encoder.fit_transform(df['ST_Slope'])

 
df['ST_Slope'].unique()


# In[11]:


print(df.shape)
print(df["HeartDisease"].value_counts())


# In[12]:


df.isnull().sum()


# In[13]:


df.describe()


# In[14]:


df.info()


# In[15]:


# Seperating the dependent and the independent columns

X = df.drop("HeartDisease" , axis = 1)
y = df['HeartDisease']


# In[16]:


# Seperating the train and test dataset
x_train,x_test,y_train,y_test = train_test_split(X,y,stratify = y , random_state=42,test_size=0.2)


# In[17]:


print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)


# In[18]:


x_train[:2]


# # KNN Algorithm

# In[19]:


knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(x_train,y_train)
y_pred_knn = knn.predict(x_test)

print(knn.score(x_test,y_test))
print(classification_report(y_test,y_pred_knn))
print(confusion_matrix(y_test,y_pred_knn))


# In[20]:


type(y_test)


# # SVM Algorithm

# In[21]:


from sklearn.svm import SVC
classifier = SVC(kernel='rbf', random_state=27)
classifier.fit(x_train, y_train)


# In[22]:


y_pred_svm = classifier.predict(x_test)


# In[23]:


print(classifier.score(x_test,y_test))
print(classification_report(y_test, y_pred_svm))
print(confusion_matrix(y_test,y_pred_svm))


# # Random Forest Algorithm

# In[24]:


randomforest = RandomForestClassifier()
randomforest.fit(x_train,y_train)
y_pred_rf = randomforest.predict(x_test)


# print('Train Accuracy %s' % round(accuracy_score(y_test, y_pred_rf),2))
# print('Train F1-score %s' % f1_score(y_test, y_pred_rf, average=None))
# print(classification_report(y_test, y_pred_rf))
# print("Confusion Matrix:\n",confusion_matrix(y_test, y_pred_rf))

# # LSTM - Long Short Term Memory
# 

# In[32]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers.core import Dense,Activation,Dropout
from keras.layers import LSTM
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

from keras.callbacks import EarlyStopping
import math


# In[33]:


X_train,Y_train,X_test,Y_test=train_test_split(X,y,test_size=0.5,random_state=0)


# In[34]:


print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)


# In[35]:


print(X_train[:1])


# In[36]:


X_train=np.expand_dims(X_train, axis=2) 
Y_train=np.expand_dims(Y_train, axis=2)
es=EarlyStopping(patience=7)
model=Sequential()
model.add(LSTM(1,input_shape=(11,1)))
model.add(Dense(1,activation='softmax'))
model.add(Activation('sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='Adam',metrics=['accuracy'])
model.fit(X_train,X_test,epochs=10,batch_size=1,verbose=1,callbacks=[es])
predict = model.predict(Y_train)
c = model.evaluate(Y_train,Y_test)


# In[37]:


len(predict)


# In[38]:


knn_acc = []
knn_tss = []
knn_prec = []
knn_tn = []
knn_tp = []
knn_fn = []
knn_fp = []

svm_acc = []
svm_tss = []
svm_prec = []
svm_tn = []
svm_tp = []
svm_fn = []
svm_fp = []

randmf_acc = []
randmf_tss = []
randmf_prec = []
randmf_tn = []
randmf_tp = []
randmf_fn = []
randmf_fp = []

for i in range(0,11):

  x_train,x_test,y_train,y_test = train_test_split(X,y,stratify = y , random_state=42,test_size=0.3)
  
  ## Running KNN Algorithm
  knn.fit(x_train,y_train)
  y_pred_knn = knn.predict(x_test)

  tn, fp, fn, tp = confusion_matrix(y_test, y_pred_knn).ravel()
  knn_tn.append(tn)
  knn_tp.append(fp)
  knn_fn.append(fn)
  knn_fp.append(tp)

  acck = (tp + tn) / (tn + fp + fn + tp)
  tss = (tp / (tp - fn)) - (fp / (fp + tn))
  precision = tp / (tp + fp)

  knn_acc.append(acck)
  knn_tss.append(tss)
  knn_prec.append(precision)

  ## Running Random Forest Algorithm
  randomforest.fit(x_train,y_train)
  y_pred_rf = randomforest.predict(x_test)

  tn, fp, fn, tp = confusion_matrix(y_test, y_pred_rf).ravel()
  randmf_tn.append(tn)
  randmf_tp.append(fp)
  randmf_fn.append(fn)
  randmf_fp.append(tp)

  accrf = (tp + tn) / (tn + fp + fn + tp)
  tss = (tp / (tp - fn)) - (fp / (fp + tn))
  precision = tp / (tp + fp)

  randmf_acc.append(accrf)
  randmf_tss.append(tss)
  randmf_prec.append(precision)

  ## Running SVM Algorithm
  classifier.fit(x_train, y_train)
  y_pred_svm = classifier.predict(x_test)

  tn, fp, fn, tp = confusion_matrix(y_test, y_pred_svm).ravel()
  svm_tn.append(tn)
  svm_tp.append(fp)
  svm_fn.append(fn)
  svm_fp.append(tp)

  accs = (tp + tn) / (tn + fp + fn + tp)
  tss = (tp / (tp - fn)) - (fp / (fp + tn))
  precision = tp / (tp + fp)

  svm_acc.append(accs)
  svm_tss.append(tss)
  svm_prec.append(precision)


# In[39]:


## Average: 
avg_knn_acc = sum(knn_acc) / len(knn_acc)
avg_knn_tss = sum(knn_tss) / len(knn_tss)
avg_knn_prec = sum(knn_prec) / len(knn_prec)
avg_knn_tn = sum(knn_tn) / len(knn_tn)
avg_knn_tp = sum(knn_tp) / len(knn_tp)
avg_knn_fn = sum(knn_fn) / len(knn_fn)
avg_knn_fp = sum(knn_fp) / len(knn_fp)


# In[40]:


## Average: 
avg_svm_acc = sum(svm_acc) / len(svm_acc)
avg_svm_tss = sum(svm_tss) / len(svm_tss)
avg_svm_prec = sum(svm_prec) / len(svm_prec)
avg_svm_tn = sum(svm_tn) / len(svm_tn)
avg_svm_tp = sum(svm_tp) / len(svm_tp)
avg_svm_fn = sum(svm_fn) / len(svm_fn)
avg_svm_fp = sum(svm_fp) / len(svm_fp)


# In[41]:


## Average: 
avg_randmf_acc = sum(randmf_acc) / len(randmf_acc)
avg_randmf_tss = sum(randmf_tss) / len(randmf_tss)
avg_randmf_prec = sum(randmf_prec) / len(randmf_prec)
avg_randmf_tn = sum(randmf_tn) / len(randmf_tn)
avg_randmf_tp = sum(randmf_tp) / len(randmf_tp)
avg_randmf_fn = sum(randmf_fn) / len(randmf_fn)
avg_randmf_fp = sum(randmf_fp) / len(randmf_fp)


# In[42]:


table_cols = {'Algorithm' : ['KNN' , 'SVM' , 'Random Forest'] , 'TP' : [avg_knn_tp , avg_svm_tp , avg_randmf_tp] , 'FP' : [avg_knn_fp, avg_svm_fp, avg_randmf_fp] , 'FN' : [avg_knn_fn, avg_svm_fn, avg_randmf_fn] , 'TN' : [avg_knn_tn, avg_svm_tn, avg_randmf_tn] , 'ACC' : [avg_knn_acc, avg_svm_acc, avg_randmf_acc], 'TSS' : [avg_knn_tss , avg_svm_tss, avg_randmf_tss] , 'Precision' : [avg_knn_prec, avg_svm_prec, avg_randmf_prec]}


# In[43]:


table_cols1 = pd.DataFrame.from_dict(table_cols)


# In[44]:


table_cols1


# In[79]:




