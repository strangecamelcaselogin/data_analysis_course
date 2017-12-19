import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.manifold import TSNE

from common import Titanic

def titanic():
    titanic_train = Titanic('../../data/titanic/').load_train(complete=False)

    X, y = titanic_train.data, titanic_train.target
    survived = y == 1
    died = y == 0

    p = 35
    X_emb = TSNE(n_components=2,
                 perplexity=p,
                 random_state=0,
                 verbose=1).fit_transform(X)

    plt.figure()
    plt.title("t-SNE for Titanic. perplexity={}, persons={}".format(p, len(y)))

    plt.scatter(X_emb[survived, 0], X_emb[survived, 1], c='g', marker='.', label='Survived')
    plt.scatter(X_emb[died, 0], X_emb[died, 1], c='r', marker='x', label='Died')

    plt.legend(loc='best', shadow=False, scatterpoints=1)
    plt.show()


def iris():
    iris = load_iris()
    X, y = iris.data, iris.target

    c1 = y == 0
    c2 = y == 1
    c3 = y == 2

    p = 10
    X_emb = TSNE(n_components=2,
                 perplexity=p,
                 random_state=0,
                 verbose=1).fit_transform(X)

    plt.figure()
    plt.title("t-SNE for Iris. perplexity={}, samples={}".format(p, len(y)))

    plt.scatter(X_emb[c1, 0], X_emb[c1, 1], c='g', marker='.', label='setosa')
    plt.scatter(X_emb[c2, 0], X_emb[c2, 1], c='c', marker='.', label='versicolor')
    plt.scatter(X_emb[c3, 0], X_emb[c3, 1], c='b', marker='.', label='virginica')

    plt.legend(loc='best', shadow=False, scatterpoints=1)
    plt.show()


if __name__ == '__main__':
    iris()
    titanic()
