# 定义向量类，实现简单的向量运算，重载相关运算符
class Vector:

    def __init__(self, lst):
        # 这里对传入的lst进行复制,防止外部修改,保证对象的不可变性
        self._values = list(lst)

    def __len__(self):
        """返回向量的长度(维度)"""
        return len(self._values)

    def __add__(self, another):
        """向量加法,返回结果向量,相当于重载+"""
        assert len(self) == len(another), \
            "Error in adding, length of vectors must be same"
        # return Vector([a + b for a , b in zip(self._values, another._values)])
        return Vector([a + b for a, b in zip(self, another)])

    def __sub__(self, another):
        """向量减法,相当于重载-"""
        assert len(self) == len(another), \
            "Error in subtracting, Length of vectors must be same"
        return Vector([a - b for a, b in zip(self, another)])

    def __mul__(self, k):
        """向量数乘, self * k"""
        return Vector([k * e for e in self])

    def __rmul__(self, k):
        """向量数乘, k * self"""
        return self * k

    def __pos__(self):
        """返回向量取正的结果"""
        return 1 * self

    def __neg__(self):
        """返回向量取负的结果"""
        return -1 * self

    def __iter__(self):
        """返回list的迭代器,覆盖该类的迭代器方法,使该类对象可以直接遍历"""
        return self._values.__iter__()

    def __getitem__(self, index):
        """取出向量的第index个元素"""
        return self._values[index]

    # 系统调用展示方法，对应该类的构造方法
    def __repr__(self):
        return "Vector({})".format(self._values)

    # 用户调用展示方法
    def __str__(self):
        return "({})".format(", ".join(str(e) for e in self._values))
