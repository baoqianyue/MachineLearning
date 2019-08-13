import matplotlib.pyplot as plt
from getLA.Matrix import Matrix
import math

if __name__ == "__main__":
    # 模拟图形坐标
    points = [[0, 0], [0, 5], [3, 5], [3, 4], [1, 4],
              [1, 3], [2, 3], [2, 2], [1, 2], [1, 0]]

    x = [point[0] for point in points]
    y = [point[1] for point in points]

    plt.figure(figsize=(5, 5))
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)

    plt.plot(x, y)

    # 首先将原图形坐标封装成一个矩阵，然后设计变换矩阵
    P = Matrix(points)

    # 1. 缩放变换
    # T = Matrix([[2, 0], [0, 1.5]])
    # 2. x轴反转
    # T = Matrix([[1, 0], [0, -1]])
    # 3. y轴反转
    # T = Matrix([[-1, 0], [0, 1]])
    # 4. 原点反转
    # T = Matrix([[-1, 0], [0, -1]])
    # 5. x轴错切
    # T = Matrix([[1, 0.5], [0, 1]])
    # 6. y轴错切
    # T = Matrix([[1, 0], [0.5, 1]])
    # 7. 旋转
    theta = math.pi / 3
    T = Matrix([[math.cos(theta), math.sin(theta)], [-math.sin(theta), math.cos(theta)]])

    # 矩阵乘法时，被转换的坐标item应以列向量的形式出现
    P2 = T.dot(P.T())
    plt.plot(
        [P2.col_vector(i)[0] for i in range(P2.col_num())],
        [P2.col_vector(i)[1] for i in range(P2.col_num())]
    )
    plt.show()
