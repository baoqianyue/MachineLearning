import tensorflow as tf
import numpy as np

"""实现一个卷积层的demo"""

# 先创建一个4x4大小的数组作为输入数据X
# 然后reshape成tf要求的数据格式，四维矩阵，第一个维度表示在batch中的index
# 第二第三个维度表示数据的size
# 第四个维度表示数据的深度
X = np.array([[[2], [1], [2], [-1]],
              [[0], [-1], [3], [0]],
              [[2], [1], [-1], [4]],
              [[-2], [0], [-3], [4]]], dtype=np.float32).reshape([1, 4, 4, 1])
# 创建卷积核，这里的shape参数前两个参数表示了卷积核的大小
# 第三个参数代表当前层的深度，第四个参数代表卷积核的深度
# 第四个参数同时也确定了该卷积层的输出深度，也决定了下一层输入的深度
filter_weights = tf.get_variable("weights", shape=[2, 2, 1, 1],
                                 initializer=tf.constant_initializer([[-1, 4], [2, 1]]))

# 创建偏置项
biases = tf.get_variable("biase", shape=[1], initializer=tf.constant_initializer(1))

# 输入数据的placeholder
x = tf.placeholder('float32', [1, None, None, 1])

# 卷积层
# 将输入数据和卷积核参数带入
conv = tf.nn.conv2d(x, filter=filter_weights, strides=[1, 1, 1, 1], padding='SAME')

# bias_add函数具有给每个单元加上偏置项的功能
# 这里要给卷积操作后的feature_map中的每个值都加上bias
# 因为输出的map深度只有1，所以偏置项只有一个数
add_bias = tf.nn.bias_add(conv, biases)

# 全局参数初始化op
init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    init_op.run()
    # 将要计算的op输入，并填补数据
    X_conv = sess.run(add_bias, feed_dict={x: X})

print("X after convolution: \n", X_conv)
