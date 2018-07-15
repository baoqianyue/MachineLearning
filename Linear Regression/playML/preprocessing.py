import numpy as np


class StandardScaler:

    def __init__(self):
        self.mean_ = None
        self.scale_ = None

    def fit(self, X):
        """根据训练数据得到每个特征维度上的数据的均值和方差"""
        """这里只针对二维向量进行fit"""
        assert X.ndim == 2, "The dimension of X must be 2"

        self.mean_ = np.array([np.mean(X[:, i]) for i in range(X.shape[1])])
        self.scale_ = np.array([np.std(X[:, i]) for i in range(X.shape[1])])
        return self

    def transform(self, X):
        """将Ｘ根据StandardScaler进行均值方差归一化操作"""
        assert X.ndim == 2, "The dimension of X must be 2"
        assert self.mean_ is not None and self.scale_ is not None, \
            "must fit before transform"
        assert X.shape[1] == len(self.mean_)

        resX = np.empty(shape=X.shape, dtype=float)
        for col in X.shape[1]:
            # 对矩阵的每一列进行归一化操作
            resX[:, col] = (X[:, col] - self.mean_[col]) / self.scale_[col]
        return resX
