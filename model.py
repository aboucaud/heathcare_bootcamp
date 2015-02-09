from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.preprocessing import Imputer
from sklearn.pipeline import Pipeline

params_rf = {
    'n_estimators': 50,
    'max_depth': 6,
    'bootstrap': False
    'criterion': 'entropy'
}

params_ab = {
    'n_estimators': 50,
    'learning_rate': 1.5
}

def model(X_train, y_train, X_test):
    clf = pipelin(params_rf, params_ab)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_score = clf.predict_proba(X_test)
    return y_pred, y_score

def pipelin(params_rf, params_ab):
    return Pipeline([('imputer', Imputer(strategy='most_frequent')),
                     ('rf', AdaBoostClassifier(
                            base_estimator=RandomForestClassifier(**params_rf),
                            **params_ab))])
