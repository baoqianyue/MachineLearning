from getLA.Matrix import Matrix
from getLA.Vector import Vector

if __name__ == "__main__":
    mat = Matrix([[2, 3], [2, 1]])
    print(mat)
    print("Matrix shape: {}".format(mat.shape()))
    print("Matrix size: {}".format(mat.size()))
    print("Matrix row num: {}".format(mat.row_num()))
    print("Matrix col num: {}".format(mat.col_num()))
    print("Matrix[0][0]: {}".format(mat[0, 0]))
    print("Matrix:1 row_vector: {}".format(mat.row_vector(1)))

    mat2 = Matrix([[1, 2], [2, 2]])
    print("add: {}".format(mat + mat2))
    print("subtract: {}".format(mat - mat2))
    print("mul:mat * 2 = {}".format(mat * 2))
    print("scalar-mul: 2 * mat = {}".format(2 * mat))
    print("zero_2_3: {}".format(Matrix.zero(2, 3)))

    # 模拟转换矩阵(函数)
    T = Matrix([[1.5, 0], [0, 2]])
    # 模拟二维平面中某点坐标
    p = Vector([5, 3])
    print("T.dot(p) = {}".format(T.dot(p)))

    # 模拟二维平面中一个三角形
    P = Matrix([[0, 4, 5], [0, 0, 3]])
    print("T.dot(P) = {}".format(T.dot(P)))

    print("P.T = {}".format(P.T()))

    I = Matrix.identity(2)
    print("I = {}".format(I))

    print("mat.dot(I) = {}".format(mat.dot(I)))
    print("I.dot(mat) = {}".format(I.dot(mat)))
