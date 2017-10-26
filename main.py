import csv
from collections import namedtuple

import numpy as np

from sklearn.tree import DecisionTreeClassifier

Dataset = namedtuple('TrainDataset', ['data', 'target', 'feature_names', 'target_names'])


def make_persons(columns, raw_data, include_columns, replace_columns_rules):
    prepared_data = []

    for record in raw_data:
        new_record = []
        # бежим по столбцам одной записи
        for col_idx, column in enumerate(columns):
            # если имя столбца в списке тех, которые нам нужны
            if column in include_columns:
                v = record[col_idx]  # возьмем значение с номером этого столбца
                if v:  # если там не пусто
                    if col_idx in replace_columns_rules.keys():  # и если номер этого столбца есть среди словаря замены значений
                        v = replace_columns_rules[col_idx][v]
                else:  # иначе будет 0
                    v = .0

                new_record.append(float(v))  # приведем каждый элемент новой записи к float

        prepared_data.append(new_record)

    return prepared_data


def load_data(file_name: str, is_train_data):
    with open(file_name, 'r') as csv_file:
        include = ['Survived', 'Pclass', 'Sex', 'Age', 'Fare', 'Embarked']
        replace = {
            2: {'male': 0, 'female': 1},
            9: {'C': 0, 'Q': 1, 'S': 2}  # C = Cherbourg, Q = Queenstown, S = Southampton
        }
        columns, *data = csv.reader(csv_file)  # 'PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'

        data = np.array(data)

        if is_train_data:
            columns = columns[2:]
            survived, persons = data[:, 1], make_persons(columns, data[:, 2:], include, replace)
        else:
            with open('data/gender_submission.csv', 'r') as gs:
                columns = columns[1:]
                survived = list(map(lambda e:e[1], list(csv.reader(gs))[1:]))

                persons = make_persons(columns, data[:, 1:], include, replace)

        return Dataset(persons, survived, [], ['Survived', 'Died'])


def test_prediction(valid_data, predicted):
    """
    Сосчитаем количество ошибок прогноза
    """
    total = len(valid_data)
    errors = 0
    for v, p in zip(valid_data, predicted):
        if int(v) != int(p):
            errors += 1

    return (1 - errors/total) * 100, errors, total


if __name__ == '__main__':
    """
    https://www.kaggle.com/c/titanic/data
    """

    titanic_train = load_data('data/train.csv', is_train_data=True)
    titanic_test = load_data('data/test.csv', is_train_data=False)

    dt = DecisionTreeClassifier(min_samples_split=10, random_state=0)

    dt.fit(titanic_train.data, titanic_train.target)

    survival_prediction = dt.predict(titanic_test.data)

    print('acc = {}%, {} errors, {} total.'.format(*test_prediction(titanic_test.target, survival_prediction)))
