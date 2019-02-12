import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import Perceptron as skPerceptron


def preprocess(df):
    """
    对传入的数据进行预处理
    取100组样本数据，并将X和y分离，将y值规范化为1和-1
    :param df:
    :return:
    """
    # 进行sepal length和sepal width两个特征的可视化
    # iris数据集中标签有三种0，1，2，每种各有50个样本，按照标签升序排列
    # 这里先绘制标签为0的50个样本

    plt.scatter(df[:50]['sepal length'], df[:50]['sepal width'], label='0')
    # 这里绘制标签为1的50个样本
    plt.scatter(df[50:100]['sepal length'], df[50:100]['sepal width'], label='1')
    plt.xlabel('sepal length')
    plt.ylabel('sepal width')
    plt.show()

    # 将前100个样本分离出来，并进行X和y的初始化
    # 只取前两个特征和标签数据
    data = np.array(df.iloc[:100, [0, 1, -1]])
    X = data[:, :-1]
    y = data[:, -1]
    # 将y进行规范化
    y = np.array([1 if i == 1 else -1 for i in y])
    return data, X, y


class Perceptron:
    def __init__(self, data):
        # 初始化参数
        self.w = np.ones(len(data[0]) - 1, dtype=np.float32)
        self.b = 0
        self._learning_rate = 0.1

    def sign(self, X, w, b):
        y = np.dot(X, w) + b
        return y

    def fit(self, X_train, y_train):
        """
        随机梯度下降
        :param X_train:
        :param y_train:
        :return:
        """
        is_fitting = False
        while not is_fitting:
            fitting_count = 0
            for i in range(len(X_train)):
                X = X_train[i]
                y = y_train[i]
                if self.sign(X, self.w, self.b) * y < 0:
                    # 进行参数的更新
                    self.w = self.w + self._learning_rate * np.dot(y, X)
                    self.b = self.b + self._learning_rate * y
                    fitting_count += 1
            if fitting_count == 0:
                is_fitting = True
        print("Model fitting finished")
        return is_fitting


def train(data, X_train, y_train):
    """
    进行模型的训练和结果可视化
    :param X_train:
    :param y_train:
    :return:
    """
    perceptron = Perceptron(data)
    perceptron.fit(X_train, y_train)

    # 训练结束得到分隔直线的系数
    x_points = np.linspace(4, 7, 10)
    # 分隔直线方程为w1*x + w2*y + b = 0
    # 为了绘制，将y提到方程左边，所以方程变为下式
    y_ = -(x_points * perceptron.w[0] + perceptron.b) / perceptron.w[1]
    plt.plot(x_points, y_)

    plt.plot(X_train[:50, 0], X_train[:50, 1], 'bo', label='sepal length')
    plt.plot(X_train[50:, 0], X_train[50:, 1], 'ro', label='sepal width')
    plt.xlabel('sepal length')
    plt.ylabel('sepal width')
    plt.show()


if __name__ == '__main__':
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['label'] = iris.target
    df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']
    data, X, y = preprocess(df)
    # train(data, X, y)
    model = skPerceptron(fit_intercept=False, max_iter=1000, shuffle=False)
    model.fit(X, y)
    # 可视化
    x_points = np.linspace(4, 7, 10)
    y = -(model.coef_[0][0] * x_points + model.intercept_) / model.coef_[0][1]
    plt.plot(x_points, y)
    plt.plot(X[:50, 0], X[:50, 1], 'bo', label='0')
    plt.plot(X[50:, 0], X[50:, 1], 'bo', color='green', label='1')
    plt.xlabel('sepal length')
    plt.ylabel('sepal width')
    plt.show()
