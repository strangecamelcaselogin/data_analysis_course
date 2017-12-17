import numpy as np

import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

import pandas as pd

from common import load_train


def pca_viz(dataset, res):
    colors = ['navy', 'darkorange']

    plt.figure()
    for color, i, target_name in zip(colors, [0, 1], dataset.target_names):
        plt.scatter(res[dataset.target == i, 0], res[dataset.target == i, 1], marker='x', color=color, alpha=.8, lw=2,
                    label=target_name)

    plt.legend(loc='best', shadow=False, scatterpoints=1)
    plt.title('PCA')


def lda_viz(dataset, res):
    colors = ['navy', 'darkorange']

    plt.figure()
    for color, i, target_name in zip(colors, [0, 1], dataset.target_names):
        d = res[dataset.target == i, 0]
        plt.plot(d, np.zeros(len(d)), alpha=.8, marker='x', color=color, label=target_name)

    plt.legend(loc='best', shadow=False, scatterpoints=1)
    plt.title('LDA')


def titanic():
    complete = False
    train = load_train(complete)
    # test = load_data(titanic_path + 'test.csv', complete)  # todo train + test

    pca = PCA(n_components=2)
    pca_r = pca.fit(train.data).transform(train.data)  # todo ? transform

    lda = LinearDiscriminantAnalysis(n_components=2)
    lda_r = lda.fit(train.data, train.target).transform(train.data)

    # Статистика
    print('explained variance ratio (first two components): {}'.format(pca.explained_variance_ratio_))
    print('Components_: \n', pd.DataFrame(pca.components_, ['PC1', 'PC2']))

    pca_viz(train, pca_r)
    lda_viz(train, lda_r)
    plt.show()


if __name__ == '__main__':
    titanic()
