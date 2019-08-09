from .Vector import Vector


class Matrix:

    def __init__(self, list2d):
        # 复制传入的二维数组，防止外部修改，保证数据的一致性
        self._values = [row[:] for row in list2d]

    def __repr__(self):
        return "Matrix({})".format(self._values)

    __str__ = __repr__

    @classmethod
    def zero(cls, r, c):
        """返回一个r行c列的0矩阵"""
        # 每行重复c次
        return cls([[0] * c for _ in range(r)])

    def shape(self):
        """返回矩阵的形状:(行数，列数)"""
        return (len(self._values), len(self._values[0]))

    def row_num(self):
        """返回矩阵行数"""
        return self.shape()[0]

    # 在矩阵中，矩阵的len倾向于描述当前矩阵的第一个维度情况，即有多少行
    __len__ = row_num

    def col_num(self):
        """返回矩阵列数"""
        return self.shape()[1]

    def size(self):
        """返回矩阵元素个数"""
        r, c = self.shape()
        return r * c

    def __getitem__(self, pos):
        """该魔法方法传入一个参数元组,返回具体的某一个元素"""
        r, c = pos
        return self._values[r][c]

    def row_vector(self, index):
        """返回矩阵的第index个行向量"""
        return Vector(self._values[index])

    def col_vector(self, index):
        """返回矩阵的第index个列向量"""
        # 列向量需要手动构建，返回每一行中的第index个元素，然后构成一个向量
        return Vector([row[index] for row in self._values])

    def __add__(self, another):
        """返回两个矩阵相加的结果"""
        assert self.shape() == another.shape(), \
            "Error in adding. Shape of matrix must be same"
        # 每次将两个矩阵的中的第i行向量取出，然后zip打包，然后遍历两个行向量中的每个元素，进行相加
        return Matrix([[a + b for a, b in zip(self.row_vector(i), another.row_vector(i))]
                       for i in range(self.row_num())])

    def __sub__(self, another):
        """返回两个矩阵相减的结果"""
        assert self.shape() == another.shape(), \
            "Error in subtracting. Shape of Matrix must be same"
        return Matrix([[a - b for a, b in zip(self.row_vector(i), another.row_vector(i))]
                       for i in range(self.row_num())])

    def __mul__(self, k):
        """返回矩阵数量乘的结果: self * k"""
        # 每次取出矩阵中的一行，然后遍历该行中的每个元素，然后乘上k即可
        return Matrix([[k * e for e in self.row_vector(i)] for i in range(self.row_num())])

    def __rmul__(self, k):
        """返回矩阵数量乘的结果: k * self"""
        return self * k

    def __truediv__(self, k):
        """返回矩阵数量除的结果: self / k"""
        return (1 / k) * self

    def __pos__(self):
        """返回矩阵取正的结果"""
        return 1 * self

    def __neg__(self):
        """返回矩阵取负的结果"""
        return -1 * self
