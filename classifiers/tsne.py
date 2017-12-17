import matplotlib.pyplot as plt

from sklearn.manifold import TSNE

from common import load_train

if __name__ == '__main__':
    train = load_train(complete=False)

    X, y = train.data, train.target
    survived = y == 1
    died = y == 0

    p = 200
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
