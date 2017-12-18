from sklearn.ensemble import RandomForestClassifier

from common import Titanic


def main():
    complete = True
    train, test = Titanic.load_train(complete), Titanic.load_test(complete)

    m = RandomForestClassifier()

    m.fit(train.data, train.target)

    survival_prediction = m.predict(test.data)

    print('Random Forest: acc = {}%, tested {} total.'.format((survival_prediction == test.target).mean(),
                                                              len(survival_prediction)))


main()