# Importing the important libraries

import numpy as np
import random
import pandas as pd
import math

from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler , MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report , confusion_matrix, accuracy_score, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier # Random Forest Classifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV

from keras.models import Sequential
from keras.layers.core import Dense,Activation,Dropout
from keras.layers import LSTM
from keras.callbacks import EarlyStopping

import warnings
warnings.filterwarnings('ignore')


## Listing all the algorithms

def knn_C(x_train,x_test,y_train,y_test):
    
    knn_model = KNeighborsClassifier(n_neighbors=7)
    knn_model.fit(x_train,y_train)
    y_pred_knn = knn_model.predict(x_test)

    print(knn_model.score(x_test,y_test))
    print(classification_report(y_test,y_pred_knn))
    print(confusion_matrix(y_test,y_pred_knn))
    
    
    return y_pred_knn
    
    
def SVM_C(x_train,x_test,y_train,y_test):
    
    from sklearn.svm import SVC
    
    svm_model = SVC(kernel='rbf', random_state=27)
    svm_model.fit(x_train, y_train)
    y_pred_svm = svm_model.predict(x_test)
    
    print(svm_model.score(x_test,y_test))
    print(classification_report(y_test, y_pred_svm))
    print(confusion_matrix(y_test,y_pred_svm))
    
    
    return y_pred_svm
    
    
def RandomForest_C(x_train,x_test,y_train,y_test):
    
    randomforest_model = RandomForestClassifier()
    randomforest_model.fit(x_train,y_train)
    y_pred_rf = randomforest_model.predict(x_test)
    
    print(randomforest_model.score(x_test,y_test))
    print(classification_report(y_test, y_pred_rf))
    print(confusion_matrix(y_test,y_pred_rf))
    
    
    return y_pred_rf
    
    
def LSTM_algo(X_train,Y_train,X_test,Y_test):
    
    X_train=np.expand_dims(X_train, axis=2) 
    Y_train=np.expand_dims(Y_train, axis=2)
#     es=EarlyStopping(patience=7)
    
    lstm_model=Sequential()
    lstm_model.add(LSTM(1,input_shape=(11,1)))
    lstm_model.add(Dense(1,activation='softmax'))
    lstm_model.add(Activation('sigmoid'))
    
    lstm_model.compile(loss='binary_crossentropy',optimizer='Adam',metrics=['accuracy'])
    
    lstm_model.fit(X_train,X_test,epochs=10,batch_size=1,verbose=1) #,callbacks=[es])
    y_pred_lstm = lstm_model.predict(Y_train)
    
    
    return y_pred_lstm


def array_avg(x):
    
    avg_x = sum(x) / len(x)
    
    return avg_x

def TenFoldCV_knn(X,y):
    
    knn_acc = []
    knn_tss = []
    knn_prec = []
    knn_tn = []
    knn_tp = []
    knn_fn = []
    knn_fp = []

    for i in range(0,11):

        x_train,x_test,y_train,y_test = train_test_split(X,y,stratify = y , random_state=42,test_size=0.3)

        ## Running KNN Algorithm
        y_pred_knn=knn_C(x_train,x_test,y_train,y_test)

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
    
    
    ## Average: 
    avg_knn_acc = array_avg(knn_acc)
    avg_knn_tss = array_avg(knn_tss)
    avg_knn_prec = array_avg(knn_prec)
    avg_knn_tn = array_avg(knn_tn)
    avg_knn_tp = array_avg(knn_tp)
    avg_knn_fn = array_avg(knn_fn)
    avg_knn_fp = array_avg(knn_fp)

    
def TenFoldCV_SVM(X,y):
    
    svm_acc = []
    svm_tss = []
    svm_prec = []
    svm_tn = []
    svm_tp = []
    svm_fn = []
    svm_fp = []

    for i in range(0,11):

        x_train,x_test,y_train,y_test = train_test_split(X,y,stratify = y , random_state=42,test_size=0.3)
        
        
        ## Running SVM Algorithm
        y_pred_svm=SVM_C(x_train,x_test,y_train,y_test)

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
        
        
    ## Average: 
    avg_svm_acc = array_avg(svm_acc)
    avg_svm_tss = array_avg(svm_tss)
    avg_svm_prec = array_avg(svm_prec)
    avg_svm_tn = array_avg(svm_tn)
    avg_svm_tp = array_avg(svm_tp)
    avg_svm_fn = array_avg(svm_fn)
    avg_svm_fp = array_avg(svm_fp)
    
    
def TenFoldCV_RandomForest(X,y):
    
    randmf_acc = []
    randmf_tss = []
    randmf_prec = []
    randmf_tn = []
    randmf_tp = []
    randmf_fn = []
    randmf_fp = []

    for i in range(0,11):

        x_train,x_test,y_train,y_test = train_test_split(X,y,stratify = y , random_state=42,test_size=0.3)
        
        
        ## Running Random Forest Algorithm
        y_pred_rf=RandomForest_C(x_train,x_test,y_train,y_test)

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
        
        
    ## Average: 
    avg_randmf_acc = array_avg(randmf_acc)
    avg_randmf_tss = array_avg(randmf_acc)
    avg_randmf_prec = array_avg(randmf_acc)
    avg_randmf_tn = array_avg(randmf_acc)
    avg_randmf_tp = array_avg(randmf_acc)
    avg_randmf_fn = array_avg(randmf_acc)
    avg_randmf_fp = array_avg(randmf_acc)
    





