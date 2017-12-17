from sklearn.ensemble import RandomForestClassifier

from common import load_train, load_test

if __name__ == '__main__':
    complete = True
    train = load_train(complete)
    test = load_test(complete)

    m = RandomForestClassifier()

    m.fit(train.data, train.target)

    survival_prediction = m.predict(test.data)

    print('Random Forest: acc = {}%, tested {} total.'.format((survival_prediction == test.target).mean(),
                                                            len(survival_prediction)))
