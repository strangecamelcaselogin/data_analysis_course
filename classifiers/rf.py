from sklearn.ensemble import RandomForestClassifier

from common import load_data


if __name__ == '__main__':
    titanic_path = '../data/titanic/'

    complete = True
    train = load_data(titanic_path + 'train.csv', complete)
    test = load_data(titanic_path + 'test.csv', complete)

    m = RandomForestClassifier()

    m.fit(train.data, train.target)

    survival_prediction = m.predict(test.data)

    print('Random Forest: acc = {}%, tested {} total.'.format((survival_prediction == test.target).mean(),
                                                            len(survival_prediction)))
