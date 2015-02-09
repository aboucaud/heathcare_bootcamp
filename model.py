import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer, StandardScaler
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

class CustomClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, x_rf=0.3, x_gb=0.3, max_depth_rf=6, max_depth_gb=2, n_estim_rf=20, n_estim_gb=100, n_jobs=-1, C=1.):
        self.x_rf = x_rf
        self.x_gb = x_gb
        self.max_depth_rf = max_depth_rf
        self.max_depth_gb = max_depth_gb
        self.n_estim_rf = n_estim_rf
        self.n_estim_gb = n_estim_gb
        self.n_jobs = n_jobs
        self.C = C

    def fit(self, X_train, y_train):
        self.rf = RandomForestClassifier(max_depth=self.max_depth_rf,
                                         n_estimators=self.n_estim_rf,
                                         n_jobs=self.n_jobs)
        self.gb = GradientBoostingClassifier(max_depth=self.max_depth_gb,
                                             n_estimators=self.n_estim_gb)
        self.svm = SVC(C=self.C)

        self.rf.fit(X_train, y_train)
        self.svm.fit(X_train, y_train)
        self.gb.fit(X_train, y_train)

    def predict(self, X_test):
        mix_array = self.proba(X_test)
        return mix_array.round()

    def predict_proba(self, X_test):
        class1 = self.proba(X_test)
        class0 = 1 - class1
        return np.column_stack([class0, class1])

    def proba(self, X_test):
        rf_array = self.rf.predict(X_test)
        svm_array = self.svm.predict(X_test)
        gb_array = self.gb.predict(X_test)

        mix_array = (self.x_rf * rf_array +
                     self.x_gb * gb_array +
                     (1 - self.x_rf - self.x_gb) * svm_array)
        return mix_array

myc_params = {
    "x_rf": 0.3,
    "x_gb": 0.3,
    "max_depth_rf": 6,
    "max_depth_gb": 2,
    "n_estim_rf": 20,
    "n_estim_gb": 100,
    "n_jobs": -1,
    "C": 1.}

def custom_pipeline(params):
    return Pipeline([
        ('imp', Imputer(strategy='most_frequent')),
        ('scl', StandardScaler()),
        ('myc', CustomClassifier(params))
        ])

def model(X_train, y_train, X_test):
    clf = custom_pipeline(myc_params)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_score = clf.predict_proba(X_test)
    return y_pred, y_score
