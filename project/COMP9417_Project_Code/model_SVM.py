# Date:11/13/2019
# Author:Group work
#
# This file is to 1. use GridSearch to find the best hyper-parameters for SVM of each target
# 2. Then it will use these best hyper-parameters to build the SVM model and do a 5-fold cross validation
# to compute the average ROC-AUC score and accuracy score for each target

import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_validate
import matplotlib.pyplot as plt

b_pred = True  # set to True to use the base model to predict the 
b_gridsearch = False  # set to True to use the GridSearch function


def main(target, b_all_feature, metric):
    # read dataset
    # this is the dataset we have extracted from the original one
    data = pd.read_csv("./Pre/features_labels_processed.csv", header = 0, index_col = 0) 
    data = data.dropna()
    X = None
    
    # b_all_feature is the switch to decide whether use all features or only use the selected features
    # here we want to get the feature table and the target label
    if target == 'panas_negative_sum':
        # 1. for panas_negative score
        if b_all_feature:
            X = data.iloc[:, :14]
        else:
            X = data[['gps_mean_distance', 'other', 'study', 'dark_mean_duration',
        'phonelock_mean_duration']]
        y = data['panas_negative_sum']   #target column i.e price range

    if target == 'panas_postive_sum':
        # 2. for panas_postive score
        if b_all_feature:
            X = data.iloc[:, :14]
        else:
            X = data[['devices per day', 'gps_mean_distance', 'dorm', 'other',
        'dark_mean_duration', 'phonecharge_mean_duration',
         'average_stationary', 'average_walking']]
        y = data['panas_postive_sum']   #target column i.e price range

    if target == 'flourishing_sum':
        # 3. for flourishing score
        if b_all_feature:
            X = data.iloc[:, :14]
        else:
            X = data[['devices per day', 'mean_duration', 'gps_mean_distance', 'dorm',
        'other', 'study', 'phonecharge_mean_duration',
        'phonelock_mean_duration', 'average_running', 'average_walking']]
        y = data['flourishing_sum']   #target column i.e price range

    # to standardize the data
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # if b_gridsearch is True, we do GridSearch
    if b_gridsearch:
        # define SVC
        parameters = {'kernel':('linear', 'rbf', 'sigmoid', 'poly'), 'C':[1, 5, 10], 'degree': [2,3,4,5]}
        svc_CV = SVC(gamma='auto', probability = True)
        
        # We do grid search
        clf = GridSearchCV(svc_CV, parameters, cv=10, n_jobs = -1, scoring = 'roc_auc', iid = False, return_train_score = True)
        clf.fit(X, y)
        best_params = clf.best_params_
        print(best_params)


    # these are the optimal model hyper-parameters we used GridSearch to find out
    # there are 2 optimal models, one used for predicting the target based on all features, one based on selected features
    if target == 'panas_negative_sum':
        if b_all_feature:
            svc_best = SVC(gamma='auto', kernel= 'rbf', C= 5, probability = True) # all features
        else:
            svc_best = SVC(gamma='auto', kernel= 'linear', C= 5, probability = True) # feature slected
    if target == 'panas_postive_sum':
        if b_all_feature:
            svc_best = SVC(gamma='auto', kernel= 'sigmoid', C= 1, probability = True)
        else:
            svc_best = SVC(gamma='auto', kernel= 'rbf', C= 1, probability = True)

    if target == 'flourishing_sum':
        if b_all_feature:
            svc_best = SVC(gamma='auto', kernel= 'rbf', C= 1, probability = True)
        else:
            svc_best = SVC(gamma='auto', kernel= 'sigmoid', C= 5, probability = True)

    # is b_pred is True, we do a 5-fold cross validation
    if b_pred:
        cv_score = cross_validate(svc_best, X, y, scoring=['roc_auc', 'accuracy', 'r2', 'neg_log_loss'], cv=5, return_train_score=True)
        from sklearn import metrics
        return cv_score
        


if __name__ == '__main__':
    # this is the target output
    target_list = ['panas_negative_sum', 'panas_postive_sum', 'flourishing_sum']
    # this is the metrics we want to see
    metrics_list = ['train_roc_auc', 'train_accuracy', 'test_roc_auc','test_accuracy', ]
    # this list used to let the progream decide whether we use all the features or selected features
    all_list = [True, False]
    for b in all_list:
        for target in target_list:
            # drawing the Graph
            x_labels = []
            y_list = []
            for metric in metrics_list:
                cv_score = main(target = target, b_all_feature = b, metric = metric)
                mean_socre = np.mean(cv_score[metric])
                x_labels.append(metric)
                y_list.append(mean_socre)
            y = y_list
            plt.xticks(np.arange(len(x_labels)), x_labels)
            plt.bar(np.arange(len(x_labels)), y, label=metric, color = ['cornflowerblue', 'mediumaquamarine', 'cornflowerblue', 'mediumaquamarine'])
            plt.title(f'target: {target}')
            plt.ylim(top = 1.0)
            plt.savefig(f'./Figs/result_mean/{target}_{b}.png')
            # plt.show()
            plt.clf()
