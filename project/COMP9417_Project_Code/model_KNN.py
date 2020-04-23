'''
Date:11/13/2019
Author:Group work

This code will extract data from the csv file and divide it into three dataframes.
Then each dataframe will show the correlation between the test_roc_auc and the number
of neighbors.At last,the KNN modules with optimal K are built and they are tested by
cross validation to obtain the metrics.
'''

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import statistics 
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.naive_bayes import GaussianNB as GNB
from sklearn.naive_bayes import MultinomialNB as MNB
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_validate
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import accuracy_score, precision_score, recall_score

#this function will obtain whole data from the processed csv file and
#split the data into three dataframes based on the various labels.
#Meanwhile it will display the heatmap(the code has been commented) that
#shows the corrlation between every single feature.
def get_df():
    df = pd.read_csv("features_labels_processed.csv",index_col='uid')
    
    #plot heat map
    # ~ corrmat = df.corr()
    # ~ top_corr_features = corrmat.index
    # ~ plt.figure(figsize=(20,20))
    # ~ g=sns.heatmap(df[top_corr_features].corr(),annot=True,cmap="RdYlGn")
    # ~ plt.show()
    
    flour_df = df.iloc[:,:-2].copy()
    common_df = df.iloc[:,:-3].copy()
    pos_df = pd.concat([common_df,df["panas_postive_sum"]],axis=1, ignore_index=False)
    neg_df = pd.concat([common_df,df["panas_negative_sum"]],axis=1, ignore_index=False)
    #drop the record with empty label
    flour_df.dropna(how='any',inplace=True)
    pos_df.dropna(how='any',inplace=True)
    neg_df.dropna(how='any',inplace=True)
    return flour_df,pos_df,neg_df
    
def split_df(df):
    feature = df.iloc[:,:-1]
    
    #normalize feature
    scaler = MinMaxScaler(feature_range=(0,1))
    feature = scaler.fit_transform(feature)
    return feature,df.iloc[:,-1]  

#this function will return the typical metric:test_roc_auc
def cv_performance_roc(algo,algo_name,name,feature,label):
    scoring = {'acc': 'accuracy',
           'prec_macro': 'precision_macro',
           'rec_micro': 'recall_macro',
            'roc_auc':'roc_auc'
            }
    scores = cross_validate(algo,feature,label, scoring=scoring,
                         cv=5, return_train_score=True)
    # ~ print(f"{name}'s performance for {algo_name} is shown below")
    # ~ print(scores.keys())
    return statistics.mean(scores['test_roc_auc'])

#this function will return a series of metrics of train set and test set    
def cv_performance(algo,algo_name,name,feature,label):
    scoring = {'acc': 'accuracy',
           'prec_macro': 'precision_macro',
           'rec_micro': 'recall_macro',
            'roc_auc':'roc_auc'
            }
    #conduct cross_validation with 5 CV
    scores = cross_validate(algo,feature,label, scoring=scoring,
                         cv=5, return_train_score=True)
    # ~ print(f"{name}'s performance for {algo_name} is shown below")
    # ~ print(scores.keys())
    return ['train_roc_auc',statistics.mean(scores['train_roc_auc'])],['train_acc',statistics.mean(scores['train_acc'])],\
        ['test_roc_auc',statistics.mean(scores['test_roc_auc'])],['test_acc',statistics.mean(scores['test_acc'])]

#this function will return the various metrics with increasing neighbors    
def knn_k(df):
    feature,label = split_df(df)
    results = []
    for i in range(1,8):
        knn = KNN(n_neighbors = i)
        #get the tyical metric:roc_auc score
        results.append(cv_performance_roc(knn,"knn",df.columns[-1],feature,label))
    return results

#this function will find best number of neighbors , input this parameter
#to KNN module and get a series of metrics.
def knn(df):
    feature,label = split_df(df)
    #gridsearch
    parameters = {'n_neighbors':np.arange(1,5)}
    grid_knn = GridSearchCV(KNN(),parameters,cv=6,return_train_score=True,iid=True)
    grid_knn.fit(feature, label)
    estimator = grid_knn.best_estimator_
    k_neigh = estimator.get_params()['n_neighbors']
    knn = KNN(n_neighbors = k_neigh)
    return cv_performance(knn,"knn",df.columns[-1],feature,label)
    
#plot the graph showing a list of metrics with optimal K
def plot_graph(k1,name):    
    labels, ys = zip(*k1)
    xs = np.arange(len(labels)) 
    metrics_list = ['train_roc_auc', 'train_accuracy', 'test_roc_auc','test_accuracy', ]
    plt.bar(xs, ys, color = ['cornflowerblue', 'mediumaquamarine', 'cornflowerblue', 'mediumaquamarine'])
    plt.xticks(xs, labels)
    plt.ylim(top = 1.0)
    plt.title(f'target:{name}')
    plt.savefig(f'./image/target_{name}.png')
    plt.show()
    plt.clf()
    
#plot the graph showing the correlation bewteen K and test_roc_auc
def plot_k(k1,name):
    x = list(range(1,8))
    plt.plot(x,k1,)
    plt.title(f'target:{name}')
    plt.xlabel("K")
    plt.ylabel("the mean of roc_auc")
    plt.ylim(top = 1.0,bottom = 0)
    plt.savefig(f'./image/k_{name}.png')
    plt.show()
    
if __name__ == "__main__":
    #get three dataframes
    flour_df,pos_df,neg_df = get_df()
    
    #get the list of metrics of Flourishing score with various neighbors
    k1 = knn_k(flour_df)
    #show the metrics change with increasing neighbors
    plot_k(k1,"flourishing_sum")
    #get the list of metrics of Flourishing score with optimal k
    k1 = knn(flour_df)
    plot_graph(k1,"flourishing_sum")
    
    #get the list of metrics of Positive affect with various neighbors
    k2 = knn_k(pos_df)
    #show the metrics change with increasing neighbors
    plot_k(k2,"panas_postive_sum")
    #get the list of metrics of Positive affect with optimal k
    k2 = knn(flour_df)
    plot_graph(k2,"panas_postive_sum")
    
    #get the list of metrics of Negative affect with various neighbors
    k3 = knn_k(neg_df)
    #show the metrics change with increasing neighbors
    plot_k(k3,"panas_negative_sum")
    #get the list of metrics of Negative affect with optimal k
    k3 = knn(flour_df)
    plot_graph(k3,"panas_negative_sum")


