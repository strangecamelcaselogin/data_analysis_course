import graphviz

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

    complete = True
    titanic = Titanic('../data/titanic/')
    train, test = titanic.load_train(complete), titanic.load_test(complete)

    dt = DecisionTreeClassifier(random_state=0,
                                max_depth=4,
                                min_samples_split=25,
                                min_samples_leaf=10)

    dt.fit(train.data, train.target)

    survival_prediction = dt.predict(test.data)

    print('DT: acc = {}%, tested {} total.'.format((survival_prediction == test.target).mean(), len(survival_prediction)))

    dt_viz(dt, train.feature_names, train.target_names, "titanic.tmp")


main()
