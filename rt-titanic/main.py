from collections import namedtuple

from sklearn.tree import DecisionTreeClassifier, export_graphviz
import pandas as pd

import graphviz

# данные, ответы, имена столбцов из данных, имена классов
Dataset = namedtuple('Dataset', ['data', 'target', 'feature_names', 'target_names'])


def viz(model, feature_names, target_names, name):
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


def load_data(path: str):
    columns = ['Survived', 'Pclass', 'Sex', 'Age', 'Fare', 'Embarked']

    df = pd.read_csv(path)[columns]  # выберем нужные столбцы

    # заменим строки на числа
    df.Sex.replace({'male': 1, 'female': 0}, inplace=True)
    df.Embarked.replace({'C': 0, 'Q': 1, 'S': 2}, inplace=True)

    df = df[~df.isnull().any(axis=1)]  # уберем все строки, где есть 'nan'

    # после обработки, разделим
    survived = df.Survived.as_matrix()
    persons = df.iloc[:, 1:].as_matrix()

    return Dataset(persons, survived, columns[1:], ['Survived', 'Died'])


if __name__ == '__main__':
    """
    https://www.kaggle.com/c/titanic/data
    """

    titanic_path = '../data/titanic/'

    train = load_data(titanic_path + 'train.csv')
    test = load_data(titanic_path + 'test.csv')

    dt = DecisionTreeClassifier(min_samples_split=10, random_state=0)

    dt.fit(train.data, train.target)

    survival_prediction = dt.predict(test.data)

    print('acc = {}%, tested {} total.'.format((survival_prediction == test.target).mean(), len(survival_prediction)))

    viz(dt, train.feature_names, train.target_names, "titanic.tmp")

