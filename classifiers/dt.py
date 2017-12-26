import graphviz

import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_graphviz

from common import Titanic


def dt_viz(model, feature_names, target_names, name):
    """
    визуализация дерева http://scikit-learn.org/stable/modules/tree.html
    :param model: модель DecisionTreeClassifier
    :param feature_names: имена признаков
    :param target_names: имена классов
    """
    dot_data = export_graphviz(model,
                               out_file=None,
                               feature_names=feature_names,
                               class_names=target_names,
                               filled=True,
                               rounded=True,
                               special_characters=True)
    graph = graphviz.Source(dot_data)
    graph.render(name)


def main():
    """
    https://www.kaggle.com/c/titanic/data
    """

    complete = False
    titanic = Titanic('../data/titanic/')
    train, test = titanic.load_train(complete), titanic.load_test(complete)


    params = [
        # 1
        {},
        # 2
        {
            "max_depth": 10,
            "min_samples_split": 10,
            "min_samples_leaf": 50,
            "max_features": 2
        },
        # 3
        {
            "max_depth": 4,
            "min_samples_split": 10,
            "min_samples_leaf": 50,
            "max_features": 3
        },
        # 4
        {
            "max_depth": 10,
            "min_samples_split": 40,
            "min_samples_leaf": 10,
            "max_features": 2
        },
        # 5
        {
            "max_depth": 10,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "max_features": 3
        },
        # 6
        {
            "max_depth": 10,
            "min_samples_split": 25,
            "min_samples_leaf": 10,
            "max_features": None
        },
        # 7
        {
            "max_depth": 10,
            "min_samples_split": 25,
            "min_samples_leaf": 10,
            "max_features": 6
        },
    ]

    for i, p in enumerate(params):
        dt = DecisionTreeClassifier(random_state=5, **p)

        dt.fit(train.data, train.target)

        survival_prediction = dt.predict(test.data)

        print('DT {}: acc = {}%, tested {} total.'.format(i+1, np.round((survival_prediction == test.target).mean(), 4) * 100, len(survival_prediction)))

        dt_viz(dt, train.feature_names, train.target_names, "dt_titanic_{}_model.tmp".format(i+1))


main()
