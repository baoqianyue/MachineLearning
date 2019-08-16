from .Matrix import Matrix
from .Vector import Vector


# 在该线性系统中模拟高斯-约旦消元法
# 目前有三个限制：1. 系数矩阵的行数m必须等于未知数的数目n
# 2. 系数矩阵的每一行的主元都不为0
# 3. 每个线性系统都有唯一解
class SimpleLinearSystem:

    def __init__(self, A, b):
        """构建增广矩阵，用户无需关系增广矩阵的构建，只需传入系数矩阵A和结果向量b"""
        assert A.row_num() == len(b), \
            "row number of A must be equal to the length of b"
        # 之后需多次访问，这里将这两个参数作为线性系统的一个属性
        self._m = A.row_num()
        self._n = A.col_num()
        assert self._m == self._n  # 限制1 TODO:no this restriction

        # 增广矩阵，先返回矩阵A每一个行向量(需要list类型才能够拼接元素)，然后拼接上b向量中的某个元素
        self.Ab = [Vector(A.row_vector(i).underlying_list() + [b[i]])
                   for i in range(self._m)]

    def _max_row(self, index, n):
        """从index行到第n行查找主元(index，index)最大的行向量"""
        # 初始时保存最佳主元和主元所在行数
        best, target = self.Ab[index][index], index
        for i in range(index + 1, n):
            # 若当前行对应的主元值大于best，则更新best和目标行数target
            if self.Ab[i][index] > best:
                best, target = self.Ab[i][index], i
        return target

    def _forward(self):
        """高斯-约旦消元法的前向操作"""
        n = self._m
        # 外层循环每次不仅向下推进，还会将主元的位置向右推进一位
        # 每次循环首先确定了该行的主元为self.Ab[i][i]
        for i in range(n):
            # 1. 先将主元最大的行向量提到第一行
            max_row = self._max_row(i, n)
            # 交换两行
            self.Ab[i], self.Ab[max_row] = self.Ab[max_row], self.Ab[i]

            # 2. 将主元化简为1
            self.Ab[i] = self.Ab[i] / self.Ab[i][i]  # TODO:self.Ab[i][i] == 0?
            # 3. 将主元下面的所有行都减去主元所在行的某个倍数，使主元下面的所有元素都为0
            for j in range(i + 1, n):
                # 将当前主元下面的元素消为0，对应主元列数的元素乘上主元所在的行向量(主元现在为1)
                self.Ab[j] = self.Ab[j] - self.Ab[i] * self.Ab[j][i]

    def _backward(self):
        """高斯-约旦消元法的后向操作"""
        n = self._m
        # 从下面往上遍历
        # 外层循环每次将主元的位置向左推进一位
        # 每次循环确定了该行的主元为self.Ab[i][i]
        for i in range(n - 1, -1, -1):
            # 内层循环将主元上面的所有元素都化为0
            for j in range(i - 1, -1, -1):
                # 主元上面的所有行减去主元所在行的某个倍数(每一行的第i列，对应着主元所在的列数)
                self.Ab[j] = self.Ab[j] - self.Ab[i] * self.Ab[j][i]

    def gauss_jordan_elimination(self):
        """执行高斯-约旦消元法"""
        self._forward()
        self._backward()

    def fancy_print(self):
        """打印当前的增广矩阵"""

        for i in range(self._m):
            print(" ".join(str(self.Ab[i][j]) for j in range(self._n)), end=" ")
            print("|", self.Ab[i][-1])



