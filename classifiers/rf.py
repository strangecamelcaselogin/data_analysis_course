from sklearn.ensemble import RandomForestClassifier

from common import Titanic


def main():
    complete = True
    titanic = Titanic('../data/titanic/')
    train, test = titanic.load_train(complete), titanic.load_test(complete)

    m = RandomForestClassifier(random_state=0,
                               max_depth=4,
                               min_samples_split=25,
                               min_samples_leaf=10)

    m.fit(train.data, train.target)

    survival_prediction = m.predict(test.data)

    print('Random Forest: acc = {}%, tested {} total.'.format((survival_prediction == test.target).mean(),
                                                              len(survival_prediction)))


main()