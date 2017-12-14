import numpy as np

from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression

from common import load_data


def nb(train_data, test_data):
    m = GaussianNB()

    m.fit(train_data.data, train_data.target)

    survival_prediction = m.predict(test_data.data)

    print('Naive bayes: acc = {}%, tested {} total.'.format((survival_prediction == test_data.target).mean(), len(survival_prediction)))


def lin_reg(train_data, test_data):
    # todo  http://scikit-learn.org/stable/auto_examples/linear_model/plot_iris_logistic.html
    m = LinearRegression()

    m.fit(train_data.data, train_data.target)

    survival_prediction = np.round(m.predict(test_data.data))  # round потому что возвращает float

    print('Linear Regression: acc = {}%, tested {} total.'.format((survival_prediction == test_data.target).mean(),
                                                            len(survival_prediction)))


if __name__ == '__main__':
    titanic_path = '../data/titanic/'

    complete = True
    train = load_data(titanic_path + 'train.csv', complete)
    test = load_data(titanic_path + 'test.csv', complete)

    nb(train, test)

    lin_reg(train, test)
