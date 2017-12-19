from sklearn.neural_network import MLPClassifier
from common import Titanic


def model1(train, test, constant_params):
    # два скрытых слоя в 30 и 10 нейрогов
    mlp = MLPClassifier(hidden_layer_sizes=(30, 10),
                        **constant_params)

    mlp.fit(train.data, train.target)

    survival_prediction = mlp.predict(test.data)

    print('MLP-{}: acc = {}%, tested {} total.'.format(
        model1.__name__,
        (survival_prediction == test.target).mean(),
        len(survival_prediction)))


def model2(train, test, constant_params):
    # три скрытых слоя (итого 5)
    mlp = MLPClassifier(hidden_layer_sizes=(25, 10, 10),
                        **constant_params)

    mlp.fit(train.data, train.target)

    survival_prediction = mlp.predict(test.data)

    print('MLP-{}: acc = {}%, tested {} total.'.format(
        model2.__name__,
        (survival_prediction == test.target).mean(),
        len(survival_prediction)))


def model3(train, test, constant_params):
    # один скрытый слой
    mlp = MLPClassifier(hidden_layer_sizes=(30,),
                        **constant_params)

    mlp.fit(train.data, train.target)

    survival_prediction = mlp.predict(test.data)

    print('MLP-{}: acc = {}%, tested {} total.'.format(
        model3.__name__,
        (survival_prediction == test.target).mean(),
        len(survival_prediction)))


def mlp_classifier():
    """
    constant_params = {
        'random_state': 0,
        'solver': 'lbfgs',  # показал лучше всех
        'early_stopping': False,  # флаг возможности ранней остановки градиентного спуска
        'learning_rate': 'invscaling',
        'max_iter': 1000,  # максимальное количество итераций градиентного спуска
    }
    MLP - model1: acc = 0.8731117824773413 %, tested 331 total.
    MLP - model2: acc = 0.9063444108761329 %, tested 331 total.
    MLP - model3: acc = 0.9274924471299094 %, tested 331 total.
    """
    complete = False
    train = Titanic.load_train(complete)
    test = Titanic.load_test(complete)

    constant_params = {
        'random_state': 0,
        'solver': 'lbfgs',
        'early_stopping': False,
        'learning_rate': 'invscaling',
        'learning_rate_init': 0.001,
        'max_iter': 1000,
        'activation': 'relu'
    }

    model1(train, test, constant_params)
    model2(train, test, constant_params)
    model3(train, test, constant_params)


mlp_classifier()  # todo тоже самое для MNIST
