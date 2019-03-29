import numpy as np


# y_true就是在监督学习中提供的分类的真值
def accuracy_score(y_true, y_predict):
    """计算y_true和y_predict之间的准确率"""
    assert y_true.shape[0] == y_predict.shape[0], \
        "the size of y_true must be equal to the size of y_predict"

    return sum(y_true == y_predict) / len(y_true)


def mean_square_error(y_test, y_predict):
    """
    compute the y_predict MSE
    :param y_predict:
    :param y_test:
    :return:
    """
    assert len(y_predict) == len(y_test), \
        "the y_predict size must be equal to the size of y_test"

    return np.sum(np.square(y_predict - y_test)) / len(y_test)


def root_mean_square_error(y_test, y_predict):
    """
    compute the y_predict RMSE
    :param y_predict:
    :param y_test:
    :return:
    """
    assert len(y_predict) == len(y_test), \
        "the size of y_predict must be equal to the size of y_test"

    return np.sqrt(mean_square_error(y_predict, y_test))


def mean_absolute_error(y_test, y_predict):
    """
    compute the y_predict MAE
    :param y_predict:
    :param y_test:
    :return:
    """

    assert len(y_predict) == len(y_test), \
        "the size of y_predict must be equal to the size of y_test"

    return np.sum(np.absolute(y_predict - y_test)) / len(y_test)


def r2_score(y_test, y_predict):
    """
    compute the model R square
    :param y_predict:
    :param y_test:
    :return:
    """
    return 1 - mean_square_error(y_test, y_predict) / np.var(y_test)

