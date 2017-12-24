import numpy as np

from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.linear_model import LogisticRegression, LinearRegression

from common import Titanic


def nb(train_data, test_data):
    print("Naive Bayes")

    m = GaussianNB()
    m.fit(train_data.data, train_data.target)
    survival_prediction = m.predict(test_data.data)

    print('GaussianNB: acc = {}%, tested {} total.'.format((survival_prediction == test_data.target).mean(), len(survival_prediction)))

    m = BernoulliNB()
    m.fit(train_data.data, train_data.target)
    survival_prediction = m.predict(test_data.data)

    print('BernoulliNB: acc = {}%, tested {} total.'.format((survival_prediction == test_data.target).mean(),
                                                           len(survival_prediction)))

    m = MultinomialNB()
    m.fit(train_data.data, train_data.target)
    survival_prediction = m.predict(test_data.data)

    print('MultinomialNB: acc = {}%, tested {} total.'.format((survival_prediction == test_data.target).mean(),
                                                            len(survival_prediction)))


def regressions(train_data, test_data):
    # todo  http://scikit-learn.org/stable/auto_examples/linear_model/plot_iris_logistic.html
    m = LinearRegression()
    m.fit(train_data.data, train_data.target)
    survival_prediction = np.round(m.predict(test_data.data))  # round потому что возвращает float

    print('Linear Regression: acc = {}%, tested {} total.'.format((survival_prediction == test_data.target).mean(),
                                                            len(survival_prediction)))

    m = LogisticRegression(random_state=0)
    m.fit(train_data.data, train_data.target)
    survival_prediction = np.round(m.predict(test_data.data))  # round потому что возвращает float

    print('Logistic Regression: acc = {}%, tested {} total.'.format((survival_prediction == test_data.target).mean(),
                                                                  len(survival_prediction)))


def main():
    complete = True
    titanic = Titanic('../data/titanic/')
    train, test = titanic.load_train(complete), titanic.load_test(complete)

    nb(train, test)
    regressions(train, test)


if __name__ == '__main__':
    main()