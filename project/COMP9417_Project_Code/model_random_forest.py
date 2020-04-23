'''
Date: 11/13/2019
Author: Group work

This program uses random forest classifier from sklearn library to make prediction about
flourishing, Panas positive and Panas negative.

Totally we have five parameters to adjust for the optimal values, we start with the param
n_estimators say number of trees we will use at the RF classifier. Then find the optimal
value of max_depth which is the maximun depth of tree in the RF. After that we need to find
the optimal values of min_samples_split and min_samples_leaf which can help us control the
size of our tree. Finally, we find the optimal max_features parameters using above optimal
parameters.

After we get the five optimas, we can build our RF classifier model. We use the original
data set to do the cross validation with tarin set rate test set 4:1, then get the AUC-ROC
scores and accuracy.

'''

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split, cross_validate
from sklearn.metrics import mean_squared_error, roc_auc_score, roc_curve, auc, accuracy_score
import matplotlib.pyplot as plt
import joblib
import os

def normalizationFeature(dfDataset):
    """this function is used to do normalization for the dataset"""

    for col in dfDataset.columns:
        if col != 'uid':
            MIN = min(dfDataset[col])
            MAX = max(dfDataset[col])
            newCol = dfDataset[col].apply(lambda x: (x - MIN) / (MAX - MIN))
            dfDataset = dfDataset.drop(col, axis=1)
            dfDataset[col] = newCol

    return dfDataset

def normalizationLabel(dfDataset):
    """this function is used to do normalization for the dataset"""

    for col in dfDataset.columns:
        if col != 'uid':
            newCol = dfDataset[col].apply(lambda x: 1 if x=='high' else 0)
            dfDataset = dfDataset.drop(col, axis=1)
            dfDataset[col] = newCol

    return dfDataset

def processData():
    """this function is used to read all files from dir"""

    fileName = 'features_labels.csv'
    dataset = pd.read_csv(fileName)
    dataset.dropna(inplace=True)
    allLabels = dataset.iloc[:,[-1, -2, -3]]
    dataset.drop(['flourishing_sum.', 'panas_negative_sum.', 'panas_postive_sum.'], axis=1, inplace=True)
    allFeaturesNorm = normalizationFeature(dataset)
    allLabelsNorm = normalizationLabel(allLabels)
    allLabelsNorm.columns = ['panas_postive_sum', 'panas_negative_sum', 'flourishing_sum']

    dataset = allFeaturesNorm.merge(allLabelsNorm, left_index=True, right_index=True)
    return allFeaturesNorm, allLabelsNorm

def randomFrost(trainFeatures, trainLabel):
    """this function is used to build randomFroest"""

    trainFeatures.drop(['uid',], axis=1, inplace=True)
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(trainFeatures, trainLabel, test_size=0.20)
    # find the optimal number of trees
    param = {
        'n_estimators': range(10, 91, 10),
    }

    rfModelOne = GridSearchCV(estimator=RandomForestClassifier(min_samples_split=3, min_samples_leaf=3, \
                                max_depth=8, max_features='auto', random_state=10), param_grid=param, cv=5)
    rfModelOne.fit(Xtrain, Ytrain)
    prediction = rfModelOne.predict(Xtest)
    Rsqure = rfModelOne.score(Xtest, Ytest)
    optimalTreeNumber = rfModelOne.best_params_['n_estimators']

    # find the optimal max depth of tree
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(trainFeatures, trainLabel, test_size=0.20)
    paramMaxDepth = {
        'max_depth' : range(3, 7, 1),
        'min_samples_split': range(2, 7, 1),
    }
    rfModelTwo = GridSearchCV(estimator=RandomForestClassifier(n_estimators=optimalTreeNumber, min_samples_leaf=20,\
                                max_features='auto', oob_score=True, random_state=10), param_grid=paramMaxDepth, cv=5)
    rfModelTwo.fit(Xtrain, Ytrain)
    predictionTwo = rfModelTwo.predict(Xtest)
    Rsqure = rfModelTwo.score(Xtest, Ytest)
    optimalMaxDepth = rfModelTwo.best_params_['max_depth']

    # find the optimal min_samples_split and min_samples_leaf
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(trainFeatures, trainLabel, test_size=0.20)
    paramMin = {
        'min_samples_split': range(2, 10, 1),
        'min_samples_leaf': range(2, 10, 1),
    }
    rfModelThird = GridSearchCV(estimator=RandomForestClassifier(n_estimators=optimalTreeNumber, max_depth=optimalMaxDepth,\
                                max_features='auto', oob_score=True, random_state=10), param_grid=paramMin, cv=5)
    rfModelThird.fit(Xtrain, Ytrain)
    predictionThird = rfModelThird.predict(Xtest)
    Rsqure = rfModelThird.score(Xtest, Ytest)
    optimalMinSampleSplit = rfModelThird.best_params_['min_samples_split']
    optimalMinSampleLeaf = rfModelThird.best_params_['min_samples_leaf']

    # find the optimal max_features
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(trainFeatures, trainLabel, test_size=0.36)
    paramMaxFeature = {
        'max_features': range(2, 15, 1),
    }
    rfModelFourth = GridSearchCV(estimator=RandomForestClassifier(n_estimators=optimalTreeNumber, max_depth=optimalMaxDepth,\
                                min_samples_split=optimalMinSampleSplit, min_samples_leaf=optimalMinSampleLeaf, \
                                oob_score=True, random_state=10), param_grid=paramMaxFeature, cv=5)
    rfModelFourth.fit(Xtrain, Ytrain)
    predictionFourth = rfModelFourth.predict(Xtest)
    Rsqure = rfModelFourth.score(Xtest,Ytest)
    optimalMaxFeatures = rfModelFourth.best_params_['max_features']

    # finally, we find all the optiaml params
    allParams = {
        'n_estimators': optimalTreeNumber,
        'max_depth': optimalMaxDepth,
        'min_samples_split': optimalMinSampleSplit,
        'min_samples_leaf': optimalMinSampleLeaf,
        'max_features': optimalMaxFeatures
    }
    print(allParams)
    # use the optiaml params to train our model
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(trainFeatures, trainLabel, test_size=0.20)
    rfModel = RandomForestClassifier(**allParams)
    scoring = {
        'acc': 'accuracy',
        'prec_macro': 'precision_macro',
        'rec_micro': 'recall_macro',
        'roc_auc': 'roc_auc',
    }

    # use train data to fit the random forest model
    # then we do k-fold cross validation and get the AUC-ROC score and accuracy
    rfModel.fit(Xtrain, Ytrain)
    importance = rfModel.feature_importances_
    features = trainFeatures.columns
    plt.title('feature importance for lable Panas negative')
    plt.bar(features, importance, color=['cornflowerblue'])
    plt.xticks(rotation=90)

    YpredictionTrain = rfModel.predict(Xtrain)
    YpredictionProbTrain = rfModel.predict_proba((Xtrain))[:, 1]

    # calculate the AUC-ROC score and accuracy
    rocValueTrain = roc_auc_score(Ytrain, YpredictionProbTrain)
    trainAccuracy = accuracy_score(Ytrain, YpredictionTrain)
    YpredictionTest = rfModel.predict(Xtest)
    YpredictionProb = rfModel.predict_proba(Xtest)[:, 1]
    rocValueTest = roc_auc_score(Ytest, YpredictionProb)
    testAccuracy = accuracy_score(Ytest, YpredictionTest)
    y = [rocValueTrain, trainAccuracy, rocValueTest, testAccuracy]
    x = ['train_roc_auc', 'train_accuracy', 'test_roc_auc', 'test_accuracy']
    plt.title('target: panas_postive')
    plt.bar(x, y, color=['cornflowerblue', 'mediumaquamarine'])
    plt.ylim(top=1.0)
    plt.show()

if __name__ == "__main__":
    # process data like normalization
    allFeatures, allLabels = processData()
    # label ['panas_postive_sum', 'panas_negative_sum', 'flourishing_sum']
    # we use random forest for our three targets
    randomFrost(allFeatures, allLabels['flourishing_sum'])
    randomFrost(allFeatures, allLabels['panas_negative_sum'])
    randomFrost(allFeatures, allLabels['panas_postive_sum'])
