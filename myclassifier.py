import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

class MyClassifier(object):
    def __init__(self, a=0.3, b=0.3, md1=6, md2=2, nest1=20, nest2=100, n_jobs=-1, C=1.):
        self.a = a
        self.b = b
        self.rf = RandomForestClassifier(max_depth=md1, n_estimators=nest1, n_jobs=n_jobs)
        self.svm = SVC(C=C)
        self.gb = GradientBoostingClassifier(max_depth=md2, n_estimators=nest2) #, n_jobs=n_jobs)

    def fit(self, X_train, y_train):
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
        print rf_array[:10]
        svm_array = self.rf.predict(X_test)
        print svm_array[:10]
        gb_array = self.rf.predict(X_test)
        print gb_array[:10]

        mix_array = (self.a * rf_array +
                     self.b * gb_array +
                     (1 - self.a - self.b) * svm_array)
        return mix_array


def main():
    import sys
    a = float(sys.argv[1])
    b = float(sys.argv[2])
    from gridsearch import split_data
    X_train, X_test, y_train, y_test = split_data(0.6)
    mc = MyClassifier()
    mc.fit(X_train, y_train)
    print mc.proba(X_test)
    print mc.predict(X_test)
    print mc.predict_proba(X_test)

if __name__ == '__main__':
    main()

