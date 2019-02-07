# 使用最小二乘法拟合曲线
# 最小化残差平方和(L2范数)
import numpy as np
from scipy.optimize import leastsq
import matplotlib.pyplot as plt


def goal_fun(x):
    """
    目标函数
    :param x:
    :return: 目标函数设置为y = sin2Πx
    """
    return np.sin(2 * np.pi * x)


def fit_fun(p, x):
    """
    多项式函数
    eg: np.poly1d([1, 2, 3])生成1x^2 + 2x^1 + 3x^0
    :param p: 各次项前面的系数或者说是权重
    :param x:
    :return:
    """
    f = np.poly1d(p)
    return f(x)


def residual_fun(p, x, y):
    """
    计算残差
    :param p:
    :param x:
    :param y: y为观测值(真实值)
    :return:
    """
    res = fit_fun(p, x) - y
    return res


def fitting(x, y, x_points, M=0):
    """
    使用最小二乘法进行拟合并进行可视化
    :param x: 观测点的x坐标
    :param y: y坐标
    :param x_points: 可视化的x坐标
    :param M: 最高次数
    :return:
    """
    # 随机初始化多项式各项系数, M+1表示有一个常数项
    p = np.random.rand(M + 1)
    # 最小二乘法， 传入残差计算函数
    p_lsq = leastsq(residual_fun, p, args=(x, y))
    print('fitting param:', p_lsq[0])

    # 可视化
    plt.title('M={}'.format(M))
    plt.plot(x, y, 'ro', label='noise')
    plt.plot(x_points, goal_fun(x_points), label='goal')
    plt.plot(x_points, fit_fun(p_lsq[0], x_points), label='fitted curve')
    plt.legend()
    plt.show()
    return p_lsq


def residual_with_regularization(p, x, y, regularization=0.0001):
    """
    在误差计算时加上正则项
    :param p:
    :param x:
    :param y:
    :return:
    """
    # 计算基本误差
    res = fit_fun(p, x) - y
    # 加上正则项
    res = np.append(res, np.sqrt(0.5 * regularization * np.square(p)))
    return res


def fitting_with_regularization(x, y, x_points, M):
    # 随机初始化各项的系数，M + 1代表初始化偏置项常数
    p_init = np.random.rand(M + 1)
    p_lsq_with_regularization = leastsq(residual_with_regularization, p_init, args=(x, y))
    plt.plot(x_points, goal_fun(x_points), label='goal')
    plt.plot(x_points, fit_fun(p_lsq_9[0], x_points), label='fitted curve')
    plt.plot(x_points, fit_fun(p_lsq_with_regularization[0], x_points), label='regularization')
    plt.plot(x, y, 'ro', label='noise')
    plt.legend()
    plt.show()
    return p_lsq_with_regularization


if __name__ == '__main__':
    # 给定10个数据点
    x = np.linspace(0, 1, 10)
    y_ = goal_fun(x)
    # 给10个坐标点加入随机噪声
    y = [i + np.random.normal(0, 0.1) for i in y_]
    # 可视化点x坐标
    x_points = np.linspace(0, 1, 1000)
    # M = 0
    p_lsq_0 = fitting(x, y, x_points, M=0)
    # M = 1
    p_lsq_1 = fitting(x, y, x_points, M=1)
    # M = 3
    p_lsq_3 = fitting(x, y, x_points, M=3)
    # M = 9
    p_lsq_9 = fitting(x, y, x_points, M=9)
    # with regularization
    p_lsq_regularization = fitting_with_regularization(x, y, x_points, M=9)
