from sklearn import svm

from common import load_data


if __name__ == '__main__':
    titanic_path = '../data/titanic/'

    complete = True
    train = load_data(titanic_path + 'train.csv', complete)
    test = load_data(titanic_path + 'test.csv', complete)

    m = svm.SVC()
    m.fit(train.data, train.target)

    survival_prediction = m.predict(test.data)

    print('SVM: acc = {}%, tested {} total.'.format((survival_prediction == test.target).mean(), len(survival_prediction)))
