import numpy as np

import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC

import pandas as pd

from common import Titanic


def pca_viz(X, y):
    plt.figure()

    s = y == 1
    d = y == 0

    plt.scatter(X[s, 0], X[s, 1], marker='.', color='g', label='Survived')
    plt.scatter(X[d, 0], X[d, 1], marker='x', color='r', label='Died')

    plt.legend(loc='best', shadow=False, scatterpoints=1)
    plt.title('PCA for Titanic, persons={}'.format(len(y)))


def lda_viz(X, y):
    plt.figure()

    s = y == 1
    d = y == 0

    plt.scatter(X[s, 0], np.zeros(len(X[s])), marker ='.', color='g', label='Survived')
    plt.scatter(X[d, 0], np.zeros(len(X[d])), marker='x', color='r', label='Died')

    plt.legend(loc='best', shadow=False, scatterpoints=1)
    plt.title('LDA for Titanic, persons={}'.format(len(y)))


def iris_pca_viz(X, y):
    c1 = y == 0
    c2 = y == 1
    c3 = y == 2

    plt.figure()
    plt.title("PCA for Iris, samples={}".format(len(y)))

    plt.scatter(X[c1, 0], X[c1, 1], c='g', marker='.', label='setosa')
    plt.scatter(X[c2, 0], X[c2, 1], c='c', marker='.', label='versicolor')
    plt.scatter(X[c3, 0], X[c3, 1], c='b', marker='.', label='virginica')

    plt.legend(loc='best', shadow=False, scatterpoints=1)


def titanic_lda_pca():
    complete = False
    titanic = Titanic('../data/titanic/')
    train = titanic.load_train(complete)
    test = titanic.load_test(complete)
    X, y = train.data, train.target

    train_pca = PCA(n_components=2, random_state=0)
    train_pca_X = train_pca.fit_transform(X)

    test_pca = PCA(n_components=2, random_state=0)
    test_pca_X = test_pca.fit_transform(test.data)

    m = SVC()
    m.fit(train_pca_X, y)  # todo bokeh https://bokeh.pydata.org/en/latest/docs/gallery.html

    survival_prediction = m.predict(test_pca_X)

    # Статистика
    print('Titanic PCA stats:')
    print('explained variance ratio (first two components): {}'.format(train_pca.explained_variance_ratio_))
    print('Components_: \n', pd.DataFrame(train_pca.components_, ['PC1', 'PC2']))

    print('SVM for PCA data: acc = {}%, tested {} total.'.format((survival_prediction == test.target).mean(),
                                                                 len(survival_prediction)))

    train_lda = LinearDiscriminantAnalysis(n_components=2)
    traind_lda_X = train_lda.fit_transform(X, y)

    test_lda = LinearDiscriminantAnalysis(n_components=2)
    test_lda_X = test_lda.fit_transform(test.data, test.target)

    m = SVC()
    m.fit(traind_lda_X, y)

    survival_prediction = m.predict(test_lda_X)

    print('SVM for LDA data: acc = {}%, tested {} total.'.format((survival_prediction == test.target).mean(),
                                                                 len(survival_prediction)))

    pca_viz(train_pca_X, y)
    lda_viz(traind_lda_X, y)
    plt.show()


def iris_pca():
    dataset = load_iris()
    X, y = dataset.data, dataset.target

    pca = PCA(n_components=2, random_state=0)
    X_emb = pca.fit_transform(X)

    iris_pca_viz(X_emb, y)

    plt.show()


titanic_lda_pca()
iris_pca()
