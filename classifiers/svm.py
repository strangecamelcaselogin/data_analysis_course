from sklearn.svm import SVC

from common import Titanic


def main():
    complete = True
    titanic = Titanic('../data/titanic/')
    train, test = titanic.load_train(complete), titanic.load_test(complete)

    m = SVC()
    m.fit(train.data, train.target)

    survival_prediction = m.predict(test.data)

    print('SVM: acc = {}%, tested {} total.'.format((survival_prediction == test.target).mean(),
                                                    len(survival_prediction)))

