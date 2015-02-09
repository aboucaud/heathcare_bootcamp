from sklearn.ensemble import AdaRandomForestClassifier
from sklearn.preprocessing import Imputer
from sklearn.pipeline import Pipeline

def model(X_train, y_train, X_test):
    clf = Pipeline([('imputer', Imputer(strategy='most_frequent')),
        ('rf', AdaBoostClassifier(
            base_estimator=RandomForestClassifier(max_depth=3, n_estimators=100),
            n_estimators=20))])
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_score = clf.predict_proba(X_test)
    return y_pred, y_score
