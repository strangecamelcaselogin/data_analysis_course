import numpy as np

import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

import pandas as pd

from common import load_train


def pca_viz(y, X):
    plt.figure()

    s = y == 1
    d = y == 0

    plt.scatter(X[s, 0], X[s, 1], marker='.', color='g', label='Survived')
    plt.scatter(X[d, 0], X[d, 1], marker='x', color='r', label='Died')

    plt.legend(loc='best', shadow=False, scatterpoints=1)
    plt.title('PCA for Titanic, persons={}'.format(len(y)))


def lda_viz(y, X):
    plt.figure()

    s = y == 1
    d = y == 0

    plt.scatter(X[s, 0], np.zeros(len(X[s])), marker ='.', color='g', label='Survived')
    plt.scatter(X[d, 0], np.zeros(len(X[d])), marker='x', color='r', label='Died')

    plt.legend(loc='best', shadow=False, scatterpoints=1)
    plt.title('LDA for Titanic, persons={}'.format(len(y)))


def titanic():
    complete = False
    train = load_train(complete)

    pca = PCA(n_components=2, random_state=0)
    pca_r = pca.fit_transform(train.data)

    lda = LinearDiscriminantAnalysis(n_components=2)
    lda_r = lda.fit_transform(train.data, train.target)

    # Статистика
    print('explained variance ratio (first two components): {}'.format(pca.explained_variance_ratio_))
    print('Components_: \n', pd.DataFrame(pca.components_, ['PC1', 'PC2']))

    pca_viz(train.target, pca_r)
    lda_viz(train.target, lda_r)
    plt.show()


def iris():
    dataset = load_iris()
    X, y = dataset.data, dataset.target

    pca = PCA(n_components=2, random_state=0)
    X_emb = pca.fit_transform(X)

    c1 = y == 0
    c2 = y == 1
    c3 = y == 2

    plt.figure()
    plt.title("PCA for Iris, samples={}".format(len(y)))

    plt.scatter(X_emb[c1, 0], X_emb[c1, 1], c='g', marker='.', label='setosa')
    plt.scatter(X_emb[c2, 0], X_emb[c2, 1], c='c', marker='.', label='versicolor')
    plt.scatter(X_emb[c3, 0], X_emb[c3, 1], c='b', marker='.', label='virginica')

    plt.legend(loc='best', shadow=False, scatterpoints=1)
    plt.show()


if __name__ == '__main__':
    titanic()
    iris()
