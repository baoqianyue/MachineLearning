import numpy as np
from .metrics import accuracy_score


class LogisticRegression:

    def __init__(self):
        """初始化模型"""
        self.coef_ = None
        self.intercept_ = None
        self._theta = None

    def _sigmoid(self, x):
        return 1. / (1 + np.exp(-x))

    def fit(self, X_train, y_train, eta=0.01, n_iters=1e4):
        """根据X_train, y_train，使用梯度下降法训练Logistic Regression模型"""
        assert X_train.shape[0] == y_train.shape[0], \
            "the size of X_train must be equal to the size of y_train"

        def J(theta, X_b, y):
            """损失函数"""
            y_hat = self._sigmoid(X_b.dot(theta))

            try:
                return - np.sum(y * np.log(y_hat) + (1 - y) * (1 - y_hat)) / len(y)
            except:
                return float('inf')

        def dJ(theta, X_b, y):
            """对损失函数求导"""
            return X_b.T.dot(self._sigmoid(X_b.dot(theta)) - y) / len(y)

        def gradient_descent(X_b, y, initial_theta, eta, n_iters=1e4, epsilon=1e-8):
            """
            梯度下降
            :param X_b:
            :param y:
            :param initial_theta: 初始化的参数矩阵
            :param eta: 学习率
            :param n_iters: 训练迭代次数
            :param epsilon: 代价函数的最小误差
            :return:
            """

            theta = initial_theta
            cur_iter = 0

            while cur_iter < n_iters:
                gradient = dJ(theta, X_b, y)
                last_theta = theta
                theta = theta - eta * gradient
                # 如果两次损失函数差距在epsilon就停止梯度下降
                if abs(J(theta, X_b, y) - J(last_theta, X_b, y)) < epsilon:
                    break

                cur_iter += 1

            return theta

        # 在训练样本X矩阵上加入一列1
        X_b = np.hstack([np.ones((len(X_train), 1)), X_train])
        initial_theta = np.zeros(X_b.shape[1])
        self._theta = gradient_descent(X_b, y_train, initial_theta, eta, n_iters)
        # 参数矩阵
        self.coef_ = self._theta[1:]
        # 偏置项，截距
        self.intercept_ = self._theta[0]    
        return self

    def predict_proba(self, X_predict):
        """给定待预测数据集X_predict，返回表示X_predict的结果概率向量"""
        assert self.intercept_ is not None and self.coef_ is not None, \
            "must fit before predict!"
        assert X_predict.shape[1] == len(self.coef_), \
            "the feature number of X_predict must be equal to X_train"

        X_b = np.hstack([np.ones((len(X_predict), 1)), X_predict])
        return self._sigmoid(X_b.dot(self._theta))

    def predict(self, X_predict):
        """给定待预测数据集X_predict，返回表示X_predict的结果向量"""
        assert self.intercept_ is not None and self.coef_ is not None, \
            "must fit before predict!"
        assert X_predict.shape[1] == len(self.coef_), \
            "the feature number of X_predict must be equal to X_train"

        proba = self.predict_proba(X_predict)
        return np.array(proba >= 0.5, dtype='int')

    def score(self, X_test, y_test):
        """根据输入的预测集，评价当前模型的性能"""
        y_predict = self.predict(X_test)
        return accuracy_score(y_test, y_predict)

    def __repr__(self):
        return "LogisticRegression"
