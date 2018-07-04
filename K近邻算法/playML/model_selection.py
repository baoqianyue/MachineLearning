import numpy as np


def train_test_split(X, y, test_ratio=0.2):
    """将传入的X，y原始训练集分割按照比例分割成新训练集和测试集"""

    assert X.shape[0] == y.shape[0], \
        "the size of X must be equal to the size of y"
    assert 0.0 <= test_ratio <= 1.0, \
        "the test_radio must be valid"

    # 将X的索引乱序

    shuffled_indexes = np.random.permutation(len(X))
    test_size = int(test_ratio * len(X))

    test_indexes = shuffled_indexes[:test_size]
    train_indexes = shuffled_indexes[test_size:]

    X_train = X[train_indexes]
    y_train = y[train_indexes]

    X_test = X[test_indexes]
    y_test = y[test_indexes]

    return X_train, X_test, y_train, y_test
