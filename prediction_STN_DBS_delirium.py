#Machine Learning-Driven Radiomic Profiling of Thalamus-Amygdala Nuclei for Prediction of Postoperative Delirium after STN-DBS in Parkinson's Disease Patients
#Radziunas A. et al. 2024

#Statistical and Machine learning algorithms for STN-DBS delirium prediction:
#0 - Binary Logistic Regression
#1 - Decision Tree Classifier
#2 - Linear Discriminant Analysis
#3 - Naive Bayes Classifier
#4 - Support Vector Machine
#5 - Artificial Neural Network
#6 - One Class Support Vector Machine
#7 - Autoencoder       

from __future__ import division
import pandas as pd
import numpy as np

import tensorflow.compat.v2 as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout

import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelBinarizer
from sklearn.metrics import (accuracy_score, confusion_matrix, auc, roc_curve)
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, OneClassSVM

from statistics import stdev
from random import choice
from numpy import interp
import os

tf.get_logger().setLevel('ERROR')


#--------------------------------------
#Data
#--------------------------------------

#Select 5, 10 or 20
number_of_features = 20
datadir = "data"
datafile = os.path.join(datadir, "data_radiomics_features_selected_mRMR.csv")
df = pd.read_csv(datafile, index_col=0)

if number_of_features == 20:
    X = df.iloc[:, 0:-1]
elif number_of_features == 10:
    X = df.iloc[:, 0:10]
elif number_of_features == 5:
    X = df.iloc[:, 0:5]
else:
    print("Error")

y = df.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.33)
base_fpr = np.linspace(0, 1, 101)

#--------------------------------------
#Prediction Accuracy
#--------------------------------------

def predictionR(classifier, X_train, X_test, y_train, y_test):
    pipe = make_pipeline(StandardScaler(), classifier)
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    return y_test, y_pred, pipe

def evaluationR(y, y_hat, title = 'Confusion Matrix'):
    cm = confusion_matrix(y, y_hat, labels=[1.0, 2.0])
    sensitivity = cm[0,0]/(cm[0,0] + cm[0,1])
    specificity = cm[1,1]/(cm[1,1] + cm[1,0])
    accuracy = accuracy_score(y, y_hat)
    fpr, tpr, thresholds = roc_curve(y, y_hat, pos_label=2)
    AUC = auc(fpr, tpr)
    tpr = interp(base_fpr, fpr, tpr)
    tpr[0] = 0.0
    return accuracy, sensitivity, specificity, AUC, tpr

def print_accuracy(res_acc, res_sens, res_spec, res_AUC, tprs):
    print("%4.2f  ±%4.2f    %4.2f ±%4.2f   %4.2f ±%4.2f   %4.2f ±%4.2f" % (100*sum(res_acc)/len(res_acc), 100*stdev(res_acc), 100*sum(res_sens)/len(res_sens), 100*stdev(res_sens),
          100*sum(res_spec)/len(res_spec), 100*stdev(res_spec), sum(res_AUC)/len(res_AUC), stdev(res_AUC)))
    tprs = np.array(tprs)
    mean_tprs = tprs.mean(axis=0)
    std = tprs.std(axis=0)
    tprs_upper = np.minimum(mean_tprs + std, 1)
    tprs_lower = mean_tprs - std
    plt.figure()
    plt.plot(base_fpr, mean_tprs, 'b')
    plt.fill_between(base_fpr, tprs_lower, tprs_upper, color='grey', alpha=0.3)

    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.text(x = 0.5, y = 0.2, s="AUC = %4.4f" % (sum(res_AUC)/len(res_AUC)))
    #plt.show()

scaler = StandardScaler()

#--------------------------------------
#Artificial Neural Network
#--------------------------------------

def ANN(X_train, X_test, y_train, y_test, scaler, class_ratio):
    scaler = scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    lb = LabelBinarizer().fit(y_train)
    y_train = lb.transform(y_train)
    y_test = lb.transform(y_test)
    
    #balancing the classes
    #Bad_DBS = STN-DBS delirium, GoodDBS = no STN-DBS delirium
    #Target column: STN-DBS delirium (1-yes, 2-no)
    class_Bad_DBS=float(np.sum(y_train==0))
    class_Good_DBS=float(np.sum(y_train==1))
    class_ratio=class_Good_DBS/class_Bad_DBS

    class_weights = {0:class_ratio, 1:1}

    model = Sequential()
    model.add(Dense(5, input_dim=X_train.shape[1], activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=50, class_weight=class_weights, batch_size=5, verbose=0)
    predictions = model.predict(X_test)
    predictions = lb.inverse_transform(predictions)
    y_test = lb.inverse_transform(y_test)
    return y_test, predictions


#--------------------------------------
#Support Vector Machine
#--------------------------------------

def predSVM(classifier, X_train, X_test, y_train, y_test):
    pipe = make_pipeline(StandardScaler(), classifier)
    pipe.fit(X_train)
    y_pred = pipe.predict(X_test)
    y_pred[y_pred==1] = 2.0
    y_pred[y_pred==-1] = 1.0
    return y_test, y_pred, pipe


#--------------------------------------
#Autoencoder
#--------------------------------------

def Autoencoder(scaler, df):
    
    df1 = df[df['STN-DBS delirium (1-yes, 2-no)'] == 1]
    df2 = df[df['STN-DBS delirium (1-yes, 2-no)'] == 2]

    #STN-DBS delirium
    X1 = df1.iloc[:, 0:-1]
    y1 = df1.iloc[:, -1]
    #no STN-DBS delirium
    X2 = df2.iloc[:, 0:-1]
    y2 = df2.iloc[:, -1]

    #no STN-DBS delirium outcome splitting in training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X2, y2, test_size=0.33)
    #scaling
    scaler = scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
   
    #STN-DBS delirium outcome
    X1 = scaler.transform(X1)

    #Autoencoder
    encoding_dim = 3
    model = Sequential()
    if X_train.shape[1] == 20:
            model.add(Dense(10, input_dim=X_train.shape[1], activation='relu'))
            model.add(Dropout(0.2))
            model.add(Dense(encoding_dim, activation='relu'))
            model.add(Dropout(0.2))
            model.add((Dense(10, activation='relu')))
            model.add(Dropout(0.2))
            model.add((Dense(X_train.shape[1], activation='linear')))
    elif X_train.shape[1] == 10:
            model.add(Dense(5, input_dim=X_train.shape[1], activation='relu'))
            model.add(Dropout(0.2))
            model.add(Dense(encoding_dim, activation='relu'))
            model.add(Dropout(0.2))
            model.add((Dense(5, activation='relu')))
            model.add(Dropout(0.2))
            model.add((Dense(X_train.shape[1], activation='linear')))
    elif X_train.shape[1] == 5:
            model.add(Input(shape=X_train.shape[1],))
            model.add(Dense(5,  activation='relu'))
            model.add(Dropout(0.2))
            model.add(Dense(encoding_dim, activation='relu'))
            model.add(Dropout(0.2))
            model.add((Dense(X_train.shape[1], activation='linear')))
    else:
            print("Features error")
    model.compile(loss='msle', optimizer='adam', metrics=['mse'])
    
    model.fit(X_train, X_train, epochs=100, batch_size=5, validation_data=(X_train, X_train), verbose=0, shuffle=True)
    
    #no STN-DBS delirium outcome: prediction and errors
    #training
    X_train_pred = model.predict(X_train)
    errors = np.sum(np.square(X_train_pred - X_train)/X_train.shape[1], axis=1)
    threshold = np.mean(errors) + stdev(errors)
    #testing
    X_test_pred = model.predict(X_test)
    errors = np.sum(np.square(X_test_pred - X_test)/X_train.shape[1], axis=1)
    y_pred = [1.0 if err > threshold else 2.0 for err in errors]
    
    #STN-DBS delirium outcome: prediction of outliers
    X1_pred = model.predict(X1)
    errors = np.sum(np.square(X1_pred - X1)/X_train.shape[1], axis=1)
    y1_pred = [1.0 if err > threshold else 2.0 for err in errors] #bloga klase = 1, gera = 2
    
    #all predictions
    t_vals = np.concatenate([y_test.values, y1.values])
    p_vals = np.concatenate([y_pred, y1_pred])

    return t_vals, p_vals

#----------------------------------------------------------------------------------------
#Methods for DBS outcome prediction
#----------------------------------------------------------------------------------------


def prediction_DBS(X, y, N = 10, scaler = scaler, df=df):
  
    res_acc =  [[], [], [], [], [], [], [], []]
    res_sens = [[], [], [], [], [], [], [], []]
    res_spec = [[], [], [], [], [], [], [], []]
    res_AUC =  [[], [], [], [], [], [], [], []]
    tprs =     [[], [], [], [], [], [], [], []]
    
    
    for i in range(N):
        print("Run %d" %(i))

        #split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, stratify=y)

        #scaling  
        scaler = StandardScaler()
        scaler = scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
    
        #balancing the classes   
        class_Bad_DBS=float(np.sum(y_train==1))
        class_Good_DBS=float(np.sum(y_train==2))
        class_ratio=class_Good_DBS/class_Bad_DBS

        #Classes: 1 - STN-DBS delirium outcome (bad DBS), 2 - no STN-DBS delirium outcome (good DBS) 
        class_weights = {1:class_ratio, 2:1}


        #----------------------------------------------
        #Classification Methods
        #----------------------------------------------
        #Binary Logistic Regression
        logreg = LogisticRegression(class_weight=class_weights)
        logreg.fit(X_train, y_train)
        y_pred = logreg.predict(X_test)
        #print(y_test)
        #print(y_pred)
        
        acc, sens, spec, AUC, tpr = evaluationR(y_test, y_pred)
        res_acc[0].append(acc)
        res_sens[0].append(sens)
        res_spec[0].append(spec)
        res_AUC[0].append(AUC)
        tprs[0].append(tpr)

        #Decision Tree Classifier
        y_test, y_pred, model = predictionR(DecisionTreeClassifier(class_weight=class_weights), X_train, X_test, y_train, y_test)
        acc, sens, spec, AUC, tpr = evaluationR(y_test, y_pred)
        res_acc[1].append(acc)
        res_sens[1].append(sens)
        res_spec[1].append(spec)
        res_AUC[1].append(AUC)
        tprs[1].append(tpr)
        
        #Linear Discriminant Analysis
        y_test, y_pred, model = predictionR(LinearDiscriminantAnalysis(), X_train, X_test, y_train, y_test)
        acc, sens, spec, AUC, tpr = evaluationR(y_test, y_pred)
        res_acc[2].append(acc)
        res_sens[2].append(sens)
        res_spec[2].append(spec)
        res_AUC[2].append(AUC)
        tprs[2].append(tpr)

        #Naive Bayes Classifier
        y_test, y_pred, model = predictionR(GaussianNB(), X_train, X_test, y_train, y_test)
        acc, sens, spec, AUC, tpr = evaluationR(y_test, y_pred)
        res_acc[3].append(acc)
        res_sens[3].append(sens)
        res_spec[3].append(spec)
        res_AUC[3].append(AUC)
        tprs[3].append(tpr)
          
        #Support Vector Machine
        y_test, y_pred, model = predictionR(SVC(class_weight=class_weights), X_train, X_test, y_train, y_test)
        acc, sens, spec, AUC, tpr = evaluationR(y_test, y_pred)
        res_acc[4].append(acc)
        res_sens[4].append(sens)
        res_spec[4].append(spec)
        res_AUC[4].append(AUC)
        tprs[4].append(tpr)

        #Artificial Neural Network
        y_test, y_pred = ANN(X_train, X_test, y_train, y_test, scaler, class_ratio)
        acc, sens, spec, AUC, tpr = evaluationR(y_test, y_pred)
        res_acc[5].append(acc)
        res_sens[5].append(sens)
        res_spec[5].append(spec)
        res_AUC[5].append(AUC)
        tprs[5].append(tpr)
        
        ##Anomaly detection##

        #One class Support Vector Machine
        y_test, y_pred, model = predSVM(OneClassSVM(nu=7/34, gamma=0.005), #split 0.1, 7 with (yes) + 27 without (no) STN-DBS delirium
                                        X_train, X_test, y_train, y_test)
        acc, sens, spec, AUC, tpr = evaluationR(y_test, y_pred)
        res_acc[6].append(acc)
        res_sens[6].append(sens)
        res_spec[6].append(spec)
        res_AUC[6].append(AUC)
        tprs[6].append(tpr)
     
        #Autoencoder
        y_test, y_pred = Autoencoder(scaler, df)
        acc, sens, spec, AUC, tpr = evaluationR(y_test, y_pred)
        res_acc[7].append(acc)
        res_sens[7].append(sens)
        res_spec[7].append(spec)
        res_AUC[7].append(AUC)
        tprs[7].append(tpr)
        
        
    print("Accuracy %  Sensitivity %  Specificity %  AUC")

    #LogReg
    print("\nBinary Logistic Regression")
    print_accuracy(res_acc[0], res_sens[0], res_spec[0], res_AUC[0], tprs[0])
    #DecTree
    print("\nDecision Tree Classifier")
    print_accuracy(res_acc[1], res_sens[1], res_spec[1], res_AUC[1], tprs[1])
    #Linear Discriminat 
    print("\nLinear Discriminant Analysis")
    print_accuracy(res_acc[2], res_sens[2], res_spec[2], res_AUC[2], tprs[2])
    #Naive Bayes
    print("\nNaive Bayes Classifier")
    print_accuracy(res_acc[3], res_sens[3], res_spec[3], res_AUC[3], tprs[3])
    #Support Vector Machine
    print("\nSupport Vector Machine")
    print_accuracy(res_acc[4], res_sens[4], res_spec[4], res_AUC[4], tprs[4])
    #Artificial Neural Network
    print("\nArtificial Neural Network")
    print_accuracy(res_acc[5], res_sens[5], res_spec[5], res_AUC[5], tprs[5])
    ##Anomaly Detection##
    #One Class Support Vector Machine
    print("\nOne class SVM")
    print_accuracy(res_acc[6], res_sens[6], res_spec[6], res_AUC[6], tprs[6])
    #Autoencoder
    print("\nAutoencoder")
    print_accuracy(res_acc[7], res_sens[7], res_spec[7], res_AUC[7], tprs[7])
    print("\n")
    
        
    with open(os.path.join(datadir, str(number_of_features), "ACC.npy"), 'wb') as f:
        np.save(f, res_acc)
    with open(os.path.join(datadir, str(number_of_features), "SENS.npy"), 'wb') as f:
        np.save(f, res_sens)
    with open(os.path.join(datadir, str(number_of_features), "SPEC.npy"), 'wb') as f:
        np.save(f, res_spec)
    with open(os.path.join(datadir, str(number_of_features), "AUC.npy"), 'wb') as f:
        np.save(f, res_AUC)
    with open(os.path.join(datadir, str(number_of_features), "TPRS.npy"), 'wb') as f:
        np.save(f, tprs)


N_runs=1000


prediction_DBS(X, y, N_runs)


