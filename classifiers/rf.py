from sklearn.ensemble import RandomForestClassifier

from common import Titanic


def main():

    params = [
        {},
        {
            "max_depth": 15,
            "n_estimators": 13,
            "min_samples_split": 50,
            "min_samples_leaf": 10,
            "max_features": 2,
        },
        {
            "max_depth": 15,
            "n_estimators": 5,
            "min_samples_split": 50,
            "min_samples_leaf": 10,
            "max_features": 2
        },
        {
            "max_depth": 15,
            "n_estimators": 2,
            "min_samples_split": 50,
            "min_samples_leaf": 10,
            "max_features": 2
        },
        {
            "max_depth": 15,
            "n_estimators": 13,
            "min_samples_split": 50,
            "min_samples_leaf": 10,
            "max_features": 3
        },
        {
            "max_depth": 15,
            "n_estimators": 5,
            "min_samples_split": 50,
            "min_samples_leaf": 10,
            "max_features": 3
        },
        {
            "max_depth": 15,
            "n_estimators": 2,
            "min_samples_split": 50,
            "min_samples_leaf": 10,
            "max_features": 3
        },
        {
            "max_depth": 15,
            "n_estimators": 13,
            "min_samples_split": 50,
            "min_samples_leaf": 10,
            "max_features": 7
        },
        {
            "max_depth": 15,
            "n_estimators": 5,
            "min_samples_split": 50,
            "min_samples_leaf": 10,
            "max_features": 7
        },
        {
            "max_depth": 15,
            "n_estimators": 2,
            "min_samples_split": 50,
            "min_samples_leaf": 10,
            "max_features": 7
        }
    ]

    complete = True
    titanic = Titanic('../data/titanic/')
    train, test = titanic.load_train(complete), titanic.load_test(complete)

    for i, p in enumerate(params):
        m = RandomForestClassifier(random_state=0, **p)

        m.fit(train.data, train.target)

        survival_prediction = m.predict(test.data)

        print('Random Forest: acc = {}%, tested {} total.'.format((survival_prediction == test.target).mean(),
                                                                  len(survival_prediction)))


main()