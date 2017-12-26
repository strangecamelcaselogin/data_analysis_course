import numpy as np
from sklearn.svm import SVC

from common import Titanic


def main():
    titanic = Titanic('../data/titanic/')
    train, test = titanic.load_train(False), titanic.load_test(False)

    params = [
        #1
        {},
        #2
        {
           "kernel": "rbf",
            "C": 1,
            "max_iter": -1,
            "tol": 1e-3
        },
        #3
        {
            "kernel": "rbf",
            "C": 1,
            "max_iter": -1,
            "tol": 1e-5
        },
        #4
        {
            "kernel": "rbf",
            "C": 3,
            "max_iter": -1,
            "tol": 1e-3
        },
        #5
        {
            "kernel": "rbf",
            "C": 3,
            "max_iter": -1,
            "tol": 1e-5
        },
        #6
        {
            "kernel": "linear",
            "C": 1,
            "max_iter": 10 * 10**6,
            "tol": 1e-3
        },
        #7
        {
            "kernel": "linear",
            "C": 1,
            "max_iter": 10 * 10 ** 6,
            "tol": 1e-5
        },
        #8
        {
            "kernel": "linear",
            "C": 3,
            "max_iter": 10 * 10**6,
            "tol": 1e-3
        },
        #9
        {
            "kernel": "linear",
            "C": 3,
            "max_iter": 10 * 10 ** 6,
            "tol": 1e-5
        }
    ]
    params = [
        {
            "kernel": "linear",
            "C": 1,
            "max_iter": 10 * 10 ** 1,
            "tol": 0.001
        },
    ]
    for i, p in enumerate(params):
        m = SVC(random_state=0, **p)

        m.fit(train.data, train.target)

        survival_prediction = m.predict(test.data)

        print('SVM {}: acc = {}%, tested {} total.'.format(i+1, np.round((survival_prediction == test.target).mean(), 4) * 100,
                                                        len(survival_prediction)))

main()
