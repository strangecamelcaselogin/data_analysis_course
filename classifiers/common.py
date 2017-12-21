from os.path import join as pathjoin
import gzip
from struct import unpack

import numpy as np
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


# todo http://www.ritchieng.com/machinelearning-one-hot-encoding/

class Titanic:
    def __init__(self, path):
        self.path = path

    def load_train(self, complete):
        return self._load_data(pathjoin(self.path, 'train.csv'), complete)

    def load_test(self, complete):
        return self._load_data(pathjoin(self.path, 'test.csv'), complete)

    @staticmethod
    def _load_data(path: str, complete_data=False):
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


class MNIST:
    def __init__(self, path):
        self.path = path

    @staticmethod
    def _open_idx_images(images_archive):
        with gzip.open(images_archive) as byte_stream:
            _, total_count, width, height = unpack('>IIII', byte_stream.read(4 * 4))

            stream_len = total_count * width * height
            return (np.array(unpack('>{}B'.format(stream_len), byte_stream.read(stream_len)), dtype=np.ubyte)
                .reshape(total_count, width, height))

    @staticmethod
    def _open_idx_labels(labels_archive):
        with gzip.open(labels_archive) as byte_stream:
            _, total_count = unpack('>II', byte_stream.read(2 * 4))

            return unpack('>{}B'.format(total_count), byte_stream.read(total_count))

    def load_test(self):
        test_imgs = pathjoin(self.path, 't10k-images-idx3-ubyte.gz')
        test_labels = pathjoin(self.path, 't10k-labels-idx1-ubyte.gz')

        images = self._open_idx_images(test_imgs)
        labels = self._open_idx_labels(test_labels)

        return DataSet(data=images, target=labels, feature_names=[str(i) for i in range(784)], target_names=[str(i) for i in range(10)])

    def load_train(self):
        train_imgs = pathjoin(self.path, 'train-images-idx3-ubyte.gz')
        train_labels = pathjoin(self.path, 'train-labels-idx1-ubyte.gz')

        images = self._open_idx_images(train_imgs)
        labels = self._open_idx_labels(train_labels)

        return DataSet(data=images, target=labels, feature_names=[str(i) for i in range(784)],
                       target_names=[str(i) for i in range(10)])
