from .Matrix import Matrix
from .Vector import Vector
from ._global import is_zero


# 在该线性系统中模拟高斯-约旦消元法
# 消元法后得到当前增广矩阵的行最简形式，可以判断线性系统是否有解
class LinearSystem:

    def __init__(self, A, b):
        """构建增广矩阵，用户无需关系增广矩阵的构建，只需传入系数矩阵A和结果向量b"""
        assert A.row_num() == len(b), \
            "row number of A must be equal to the length of b"
        self._m = A.row_num()
        self._n = A.col_num()

        self.Ab = [Vector(A.row_vector(i).underlying_list() + [b[i]])
                   for i in range(self._m)]
        # 保存每行的主元所在的列数
        self.pivots = []

    def _max_row(self, i_index, j_index, n):
        best, target = self.Ab[i_index][j_index], i_index
        for i in range(i_index + 1, n):
            if self.Ab[i][j_index] > best:
                best, target = self.Ab[i][j_index], i
        return target

    def _forward(self):
        """高斯-约旦消元法的前向操作"""
        # i指向当前处于第i行，k指向当前的主元所在列数
        i, k = 0, 0
        while i < self._m and k < self._n:
            # 看self.Ab[i][k]位置是否可以是主元,先找该列最大的元素
            max_row = self._max_row(i, k, self._m)
            self.Ab[i], self.Ab[max_row] = self.Ab[max_row], self.Ab[i]

            # 判断当前行的主元位置是否为0，如果为0，主元位置向右推进一位
            if is_zero(self.Ab[i][k]):
                k += 1
            else:
                # 将主元化简为1
                self.Ab[i] = self.Ab[i] / self.Ab[i][k]
                # 将主元下面的所有行都减去主元所在行的某个倍数，使主元下面的所有元素都为0
                for j in range(i + 1, self._m):
                    self.Ab[j] = self.Ab[j] - self.Ab[j][k] * self.Ab[i]
                # 将当前行的主元位置保存
                self.pivots.append(k)
                i += 1

    def _backward(self):
        """高斯-约旦消元法的后向操作"""
        # 判断主元个数
        n = len(self.pivots)
        for i in range(n - 1, -1, -1):
            # 取出主元
            k = self.pivots[i]
            # 主元为self.Ab[i][k]
            for j in range(i - 1, -1, -1):
                self.Ab[j] = self.Ab[j] - self.Ab[i] * self.Ab[j][k]

    def gauss_jordan_elimination(self):
        """如果当前线性系统有解，返回True，如果无解，返回False"""
        self._forward()
        self._backward()

        # 根据增广矩阵判断是否有解
        for i in range(len(self.pivots), self._m):
            if not is_zero(self.Ab[i][-1]):
                return False
        return True

    def fancy_print(self):
        """打印当前的增广矩阵"""
        for i in range(self._m):
            print(" ".join(str(self.Ab[i][j]) for j in range(self._n)), end=" ")
            print("|", self.Ab[i][-1])
