#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import numpy as np
import pandas as pd

from model import model
from sklearn.cross_validation import cross_val_score, train_test_split
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.preprocessing import Imputer

from sklearn.pipeline import Pipeline


def read_train_data():
    df = pd.read_csv('train.csv')
    y = df['TARGET'].values
    X = df.drop('TARGET', axis=1).values
    y[np.isnan(y)] = -1
    X[np.isnan(X)] = -1
    return X, y

def split_data(t_size):
    X, y = read_train_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=t_size, random_state=0)
    return X_train, X_test, y_train, y_test

def test_grid(ensembl, param_grid):
    X_train, y_train = read_train_data()
    grid_search = GridSearchCV(ensembl, param_grid=param_grid, cv=3, verbose=3, n_jobs=4)
    grid_search.fit(X_train, y_train)
    return grid_search.best_params_, grid_search.best_score_

def get_score(params_rf, params_ab):
    X, y = read_train_data()
    clf = Pipeline([('imputer', Imputer(strategy='most_frequent')),
                    ('rf', AdaBoostClassifier(
                        # base_estimator=RandomForestClassifier(n_jobs=4, **params_rf),
                        **params_ab))])
    score = cross_val_score(clf, X, y, cv=5, scoring='roc_auc')
    return score

def test_rf():
    print "TEST RANDOM FOREST"
    rf = RandomForestClassifier()
    param_grid = {'n_estimators': [100],
                  'max_depth': [3, 5, 7, 9],
                  'bootstrap': [True, False]}
    best_parms, best_score = test_grid(rf, param_grid)
    print best_parms, best_score
    return best_parms

def test_ab():
    print "TEST ADABOOST RANDOM FOREST"
    ab = AdaBoostClassifier()
    # param_grid = {'base_estimator': [RandomForestClassifier(n_jobs=4, n_estimators=100, bootstrap=False, max_depth=7)],
    param_grid = {'base_estimator': [RandomForestClassifier(n_jobs=4, n_estimators=50, bootstrap=False, max_depth=6)],
                  'n_estimators': [50],
                  'learning_rate': [1., 1.5, 2., 3.]}
    best_parms, best_score = test_grid(ab, param_grid)
    print best_parms, best_score
    return best_parms




if __name__ == '__main__':
    params_rf = test_rf()
    params_ab = test_ab()
    # parms = test_rf()
    # for n_ada in [20, 40, 60, 80, 100]:
    #     x = get_score(parms, n_ada)
    #     print '%d\t%.4f' % (n_ada, np.mean(x))
    print get_score(params_rf, params_ab)
