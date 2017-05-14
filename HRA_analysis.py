#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 12 10:34:06 2017

@author: wellingtonjohnson
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import norm


#A few functions I found online in order to create some shortcuts to exploring variables,
#imputing missing data, showing if data is missing in a certain column, and just exploring the 
#categorical variables in a dataset
def cat_imputation(data, column, value):
    data.loc[data[column].isnull(),column] = value
             
def cat_exploration(data,column):
    return data[column].value_counts()

def show_missing(data):
    missing = data.columns[data.isnull().any()].tolist()
    return missing

def print_unique_cols(data):
    for col in data:
        if data[col].dtype == 'object':
            print(data[col].name)
            print (data[col].unique())

df = pd.read_csv('/Users/wellingtonjohnson/Desktop/HRA.csv')


#Simple Data Exploration

print(df.describe())
print(df.info())
print(df.columns)




#Data Visualization

#Correlation heat map of data
corrmat = df.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True);
plt.show()
           
           
_ = sns.barplot(x='time_spend_company', y='salary', data=df, hue='left')
_ = plt.title('Salaries and Tenure by Retention/Loss')
_ = plt.xlabel('Tenure (Years)')
_ = plt.ylabel('Average Salary')
plt.show()

tleft = sns.barplot(x='time_spend_company',y='left',data=df)
plt.show()

tsalary = sns.barplot(x='salary',y='left',data=df)
plt.show()

tsales = sns.barplot(x='sales',y='left',data=df)
plt.show()





#Binarizing statisfaction_level, last_evaluation, and  average_monthly_hours


df['avg_satisfaction_level'] = pd.cut(df['satisfaction_level'], 3)
avg_satrange = df[['avg_satisfaction_level', 'left']].groupby(['avg_satisfaction_level']).mean()

print(avg_satrange)

df.loc[df['satisfaction_level'] <=  0.393, 'satisfaction_level'] = 0
df.loc[(df['satisfaction_level'] > 0.393) & (df['satisfaction_level'] <= 0.697), 'satisfaction_level'] = 1
df.loc[(df['satisfaction_level'] > 0.697) & (df['satisfaction_level'] <= 1), 'satisfaction_level'] = 2
df.drop(['avg_satisfaction_level'], axis = 1, inplace = True)



df['avg_mon_hours_range'] = pd.cut(df['average_montly_hours'], 3)
avg_prange = df[['avg_mon_hours_range', 'left']].groupby(['avg_mon_hours_range']).mean()

print(avg_prange)

df.loc[df['average_montly_hours'] <= 167.333, 'average_montly_hours'] = 0
df.loc[(df['average_montly_hours'] > 167.333) & (df['average_montly_hours'] <= 238.667), 'average_montly_hours'] = 1
df.loc[(df['average_montly_hours'] > 238.667) & (df['average_montly_hours'] <= 310.000), 'average_montly_hours'] = 2
df.drop(['avg_mon_hours_range'], axis = 1, inplace = True)



df['evaluation_range'] = pd.cut(df['last_evaluation'], 3)
last_evaluation_range = df[['evaluation_range','left']].groupby(['evaluation_range']).mean()

print(last_evaluation_range)

df.loc[df['last_evaluation'] <=  0.573, 'last_evaluation'] = 0
df.loc[(df['last_evaluation'] > 0.573) & (df['last_evaluation'] <= 0.787), 'last_evaluation'] = 1
df.loc[(df['last_evaluation'] > 0.787) & (df['last_evaluation'] <= 1), 'last_evaluation'] = 2
df.drop(['evaluation_range'], axis = 1, inplace = True)


#Exploring categorical variables  

print(cat_exploration(df,'sales'))


#Binarizing categorical variables

df['sales'].replace(['sales', 'accounting', 'hr', 'technical', 'support', 'management',
        'IT', 'product_mng', 'marketing', 'RandD'], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], inplace = True)

print(cat_exploration(df,'salary'))

df = df.replace({'salary':{
                    'low': 1,
                    'medium': 2,
                    'high': 3}})
    
    
    
    

#Split our data
from sklearn.model_selection import train_test_split

X = df.drop('left', axis = 1).values
y = df['left'].values
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state = 44 )





#Running our models
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix

###################
#AdaBoost Classifier
# begins by fitting a classifier on the original  dataset and then fits additional copies 
# of the classifier on the same dataset  but where the weights of the incorrectly classified
# instances  are adjusted such that the subsequent classifiers focus on more difficult cases
from sklearn.ensemble import AdaBoostClassifier

#ada_parameters = {'n_estimators' : np.arange(50,200,10)}

#ada = AdaBoostClassifier()

#ada_cv = GridSearchCV(ada, param_grid = ada_parameters, cv = 5)

#ada_cv.fit(X_train,y_train)

#print(ada_cv.best_params_) # received the best param at: n_estimators = 100
#print(ada_cv.best_score_) # received a score of 0.894157846487

     
ada_best = AdaBoostClassifier(n_estimators = 100)

ada_best.fit(X_train,y_train)

ada_score = ada_best.score(X_test,y_test)

print(ada_score) #0.889333333333, not bad at all

ada_y_pred = ada_best.predict(X_test)

print(confusion_matrix(y_test,ada_y_pred))
print(classification_report(y_test,ada_y_pred))


#################
#SGD Classifier
# This estimator implements regularized linear models with stochastic gradient descent learning:
# The gradient of the loss is estimated  each sample at a time and the model is updated alognt the way
# with a decreasing strength schedule (aka learning rate)

from sklearn.linear_model import SGDClassifier


#sgd_parameters = [{'loss': ['hinge','log', 'squared_hinge'],
               #'penalty': ['l2','l1','elasticnet']}]

#sgd = SGDClassifier()

#sgd_cv = GridSearchCV(sgd,param_grid=sgd_parameters,cv = 5)

#sgd_cv.fit(X_train,y_train)

#print(sgd_cv.best_params_) #The parameters for this model are implementing the log loss function with an elasticnet penalty
#print(sgd_cv.best_score_) #The best score produced was 0.775564630386

     
sgd_best = SGDClassifier(loss = 'log', penalty = 'l1')

sgd_best.fit(X_train,y_train)

sgd_score = sgd_best.score(X_test,y_test)

print(sgd_score) #0.785 not terrible, did not perform as well as our adaboost classfier

sgd_y_pred = sgd_best.predict(X_test)

print(confusion_matrix(y_test,sgd_y_pred))
print(classification_report(y_test,sgd_y_pred)) #Precision: 0.83, Recall: 0.61, f1-score:0.63



############
#LogisticRegression

from sklearn.linear_model import LogisticRegression

#lrg_parameters = {'penalty':['l1','l2'], 'C': np.logspace(-5,8,15)}

#lrg = LogisticRegression()

#lrg_cv = GridSearchCV(lrg, param_grid = lrg_parameters)

#lrg_cv.fit(X_train,y_train)

#print(lrg_cv.best_params_) #Best parameters seem to be: {'C': 0.0061054022965853268, 'penalty': 'l2'}
#print(lrg_cv.best_score_) #Best score seems to be 0.785232102675


lrg_best = LogisticRegression(C = 0.0061054022965853268, penalty = 'l2')

lrg_best.fit(X_train,y_train)

lrg_score = lrg_best.score(X_test,y_test)

print(lrg_score) #0.785666666667 performes about as well as the sgd classifier

lrg_y_pred = lrg_best.predict(X_test)

print(confusion_matrix(y_test,lrg_y_pred))
print(classification_report(y_test,lrg_y_pred)) #Precision: 0.76, Recall: 0.79, f1-score:0.76




##############
#KNeighborsClassifier

#Neighbors-based classification is a type of instance-based learning or non-generalizing learning: 
#it does not attempt to construct a general internal model, but simply stores instances of the training data.
#Classification is computed from a simple majority vote of the nearest neighbors of each point: 
#a query point is assigned the data class which has the most representatives within the nearest neighbors of the point.

from sklearn.neighbors import KNeighborsClassifier

#knn_parameters = {'n_neighbors': np.arange(1,10,1),
              #'weights': ['uniform','distance'],
            #'algorithm': ['auto']}

#knn_for_cv = KNeighborsClassifier

#knn_cv = GridSearchCV(knn_for_cv,param_grid = knn_parameters)

#knn_cv.fit(X_train,y_train)

#print(knn_cv.best_params_) 
#print(knn_cv.best_score_) 
  
#The best parameters for this model given the data and the feature engineering seems to be:
    #Algorithm: auto, n_neighbors: 4, and weights: distance
# The best score this model gave was 0.957746478873, very good


knn_best = KNeighborsClassifier(algorithm = 'auto', n_neighbors = 4, weights = 'distance')

knn_best.fit(X_train,y_train)

knn_score = knn_best.score(X_test,y_test)

print(knn_score) #0.959333333333, much better! We might be onto something

knn_y_pred = knn_best.predict(X_test)

print(confusion_matrix(y_test,knn_y_pred))
print(classification_report(y_test,knn_y_pred)) #0.96 Precision, Recall, and f1-score


###############
#Support Vector Classifier

from sklearn.svm import SVC

#svc_parameters = {'C' : [1,10,100], 'gamma': [0.1,0.01]}
#Its recommended to spread out different C and gamma parameters exponentially while hyperparameter tuning


#svc = SVC()

#svc_cv = GridSearchCV(svc,param_grid = svc_parameters)

#svc_cv.fit(X_train,y_train)

#print(svc_cv.best_params_)
#print(svc_cv.best_score_)

#The best parameters for this model given the data and the feature engineering seems to be:
    #C : 100, gamma: 0.1
# The best score this model gave given the best parameters was 0.961746812234, better than our previous KNN model!


#Now I'll fit the best SVC model I got on the data and see what the classificatipn report looks like
svc_best = SVC(C = 100, gamma = 0.1)

svc_best.fit(X_train,y_train)

svc_score = svc_best.score(X_test,y_test)

print(svc_score) #0.967, a little bit better than our KNeighbors Classifier, but we'll take it!

svc_y_pred = svc_best.predict(X_test)

print(confusion_matrix(y_test,svc_y_pred))
print(classification_report(y_test,svc_y_pred)) #0.97 Precision, Recall, and f1-score

#This classification report on the SVC model ran does very well on the data, high amount of 
#true positive and negatives and low amounts of false positives and negatuves, which is what we want




