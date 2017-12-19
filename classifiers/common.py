import pandas as pd

class DataSet:
    """
    Данные, ответы, имена столбцов из данных, имена классов
    """
    def __init__(self, data, target, feature_names, target_names):
        self.data = data
        self.target = target
        self.feature_names = feature_names #  if feature_names else [str(j) for j in range(len(data[0]))]
        self.target_names = target_names #  if target_names else []


class Titanic:
    titanic_path = '../data/titanic/'

    @classmethod
    def load_train(cls, complete):
        return Titanic.load_data(cls.titanic_path + 'train.csv', complete)

    @classmethod
    def load_test(cls, complete):
        return Titanic.load_data(cls.titanic_path + 'test.csv', complete)

    @staticmethod
    def load_data(path: str, complete_data=False):
        """
        Загрузим данные из csv, complete_data означает замену nan значений, а не удаление таких строк
        """
        columns = ['Survived', 'Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'SibSp', 'Parch']

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

        return DataSet(persons, survived, columns[1:], ['Died', 'Survived'])
