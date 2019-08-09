from getLA.Matrix import Matrix

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
