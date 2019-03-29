import numpy as np
from .metrics import r2_score


class SimpleLinearRegression:

    def __init__(self):
        """初始化Simple Linear Regression 模型"""
        self.x_ = None
        self.y_ = None

    def fit(self, x_train, y_train):
        """根据训练数据集训练模型"""
        assert y_train.ndim == 1, \
            "simple linear regressor can only solve single feature train data"
        assert len(x_train) == len(y_train), \
            "the size of x_train must be equal to the size of y_train"

        self.x_ = x_train
        self.y_ = y_train

        x_mean = np.mean(self.x_)
        y_mean = np.mean(self.y_)

        # 使用向量化计算 代替循环
        # a的分子
        num = (x_train - x_mean).dot(y_train - y_mean)
        # a的分母
        den = (x_train - x_mean).dot(x_train - x_mean)

        self.a_ = num / den
        self.b_ = y_mean - self.a_ * x_mean
        return self

    def predict(self, x_predict):
        """给定测试集x_predict,返回结果向量"""
        assert x_predict.ndim == 1, \
            "Simple Linear regression can only solve single feature train data"
        assert self.x_ is not None and self.y_ is not None, \
            "must fit before predict"
        # 使用一个私有的预测函数，对一个具体的数据进行预测
        return np.array([self._predict(x) for x in x_predict])

    def _predict(self, single_x):
        """
        给定单个数据，带入简单线性回归模型中进行预测，并返回结果
        :param single_x:
        :return: res
        """
        return self.a_ * single_x + self.b_

    def score(self, x_test, y_test):
        """
        根据给定的测试集，确定当前模型的准确度
        :param x_test:
        :param y_test:
        :return:
        """
        y_predict = self.predict(x_test)
        return r2_score(y_test, y_predict)

    def __repr__(self):
        return "SimpleLinearRegression()"
