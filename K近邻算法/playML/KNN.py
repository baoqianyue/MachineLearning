import numpy as np
from math import sqrt
from collections import Counter


# def kNN_Classify(k, X_train, y_train, x):
#     assert 1 <= k <= X_train.shape[0], "k must be valid"
#     assert X_train.shape[0] == y_train.shape[0], \
#         "the size of X_train must equal to the size of y_train"
#     assert X_train.shape[1] == x.shape[0], \
#         "the feature number of x must be equal to X_train"
#
#     distances = [sqrt(np.sum((x_train - x) ** 2)) for x_train in X_train]
#     nearest = np.argsort(distances)
#     TopK_y = [y_train[i] for i in nearest[:k]]
#     votes = Counter(TopK_y)
#     return votes.most_common(1)[0][0]

# 重新封装Knn算法

class KnnClassifier:

    # 构造函数
    def __init__(self, k):
        assert k >= 1, "k must be vaild"
        self.k = k
        self._X_train = None
        self._y_train = None

    # 训练数据
    def fit(self, X_train, y_train):
        """根据训练集X_train和y_train训练KNN分类器"""
        assert X_train.shape[0] == y_train.shape[0], \
            "the size of X_train must be equal to the size of y_train"
        assert X_train.shape[0] >= self.k, \
            "the size of X_train must be at least k"
        self._X_train = X_train
        self._y_train = y_train
        return self

    # 预测函数
    def predict(self, X_predict):
        """给定待测数据集X_predict,返回表示X_predict的结果向量"""
        assert self._X_train is not None and self._y_train is not None, \
            "must fit before predict"
        assert X_predict.shape[1] == self._X_train.shape[1], \
            "the feature mumber of X_predict must be equal to X_train"
        # 调用私有函数对每个测试数据进行predict操作
        y_predict = [self._predict(x) for x in X_predict]
        return np.array(y_predict)

    def _predict(self, x):
        """给定单个数据，进行predict操作"""
        assert x.shape[0] == self._X_train.shape[1], \
            "the feature number of x must be equal to X_train"
        distance = [sqrt(np.sum((x_train - x) ** 2)) for x_train in self._X_train]
        nearest = np.argsort(distance)
        topK_y = [self._y_train[i] for i in nearest[:self.k]]
        votes = Counter(topK_y)
        return votes.most_common(1)[0][0]

    def score(self, X_test, y_test):

    def __repr__(self):
        return "Knn(k=%d)" % self.k
