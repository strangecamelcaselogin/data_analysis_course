from collections import namedtuple

from sklearn import cross_validation, svm
import pandas as pd


# данные, ответы, имена столбцов из данных, имена классов
Dataset = namedtuple('Dataset', ['data', 'target', 'feature_names', 'target_names'])


def load_data(path: str, complete_data=False):
    """
    Загрузим данные из csv, complete_data означает замену nan значений, а не удаление таких строк
    """
    columns = ['Survived', 'Pclass', 'Sex', 'Age', 'Fare', 'Embarked']

    df = pd.read_csv(path)[columns]  # выберем нужные столбцы

    # заменим строки на числа
    df.Sex.replace({'male': 1, 'female': 0}, inplace=True)
    df.Embarked.replace({'C': 0, 'Q': 1, 'S': 2}, inplace=True)

    if complete_data:
        df.Age = df.Age.fillna(df.Age.median())  # замена не указанного возраста на среднее

    df = df[~df.isnull().any(axis=1)]  # уберем все строки, где есть 'nan'

    # после обработки, разделим
    survived = df.Survived.as_matrix()
    persons = df.iloc[:, 1:].as_matrix()

    return Dataset(persons, survived, columns[1:], ['Survived', 'Died'])


if __name__ == '__main__':
    titanic_path = '../data/titanic/'

    complete = True
    train = load_data(titanic_path + 'train.csv', complete)
    test = load_data(titanic_path + 'test.csv', complete)

    m = svm.SVC()
    m.fit(train.data, train.target)

    survival_prediction = m.predict(test.data)

    print('acc = {}%, tested {} total.'.format((survival_prediction == test.target).mean(), len(survival_prediction)))
