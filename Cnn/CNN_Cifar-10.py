import tensorflow as tf
import numpy as np
import time
import math
from Cnn_cifar10 import Cifar10_data

max_steps = 4000
batch_size = 100
num_examples_for_eval = 10000

data_dir = "../datasets/cifar-10-batches-bin"


def variable_with_weight_loss(shape, stddev, w1):
    """
    定义权重参数
    :param shape:
    :param stddev:
    :param w1: 为正则项的系数
    :return:
    """
    # 使用正太分布初始化
    var = tf.Variable(tf.truncated_normal(shape, stddev=stddev))
    if w1 is not None:
        weights_loss = tf.multiply(tf.nn.l2_loss(var), w1, name="weights_loss")
        tf.add_to_collection("losses", weights_loss)
    return var


# 对于训练图片进行数据增强
images_train, labels_train = Cifar10_data.inputs(data_dir, batch_size, True)
images_test, labels_test = Cifar10_data.inputs(data_dir, batch_size, False)

# 创建placeholder
x = tf.placeholder(tf.float32, shape=[batch_size, 24, 24, 3])
y_ = tf.placeholder(tf.int32, shape=[batch_size])

# 第一个卷积层,size为5x5，输入通道为3（图像通道为3），64个卷积核
kernel1 = variable_with_weight_loss(shape=[5, 5, 3, 64], stddev=5e-2, w1=0.0)
# 卷积步长为1，使用padding
conv1 = tf.nn.conv2d(x, kernel1, strides=[1, 1, 1, 1], padding='SAME')
# 偏置项参数初始化为0，给每个卷积核参数都加上偏置项
bias1 = tf.Variable(tf.constant(0.0, shape=[64]))
relu1 = tf.nn.relu(tf.nn.bias_add(conv1, bias1))
# 池化size为3x3，步长为2x2，图像通过池化，size减半
# 使用最大池化的尺寸大于步长（相当于池化核发生了部分重叠）可以增加数据的丰富性
pool1 = tf.nn.max_pool(relu1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="SAME")

# 第二个卷积层
kernel2 = variable_with_weight_loss(shape=[5, 5, 64, 64], stddev=5e-2, w1=0.0)
conv2 = tf.nn.conv2d(pool1, kernel2, strides=[1, 1, 1, 1], padding='SAME')
bias2 = tf.Variable(tf.constant(0.1, shape=[64]))
relu2 = tf.nn.relu(tf.nn.bias_add(conv2, bias2))
pool2 = tf.nn.max_pool(relu2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="SAME")

# 将数据拉平
# 第二个参数取-1，表明将数据拉成一维结构
reshape = tf.reshape(pool2, shape=[batch_size, -1])
# 获取拉平后的数据维度
dim = reshape.get_shape()[1].value

# 第一个全联接层
# 第一个全联接层的隐藏神经元数为384
# 使用正则化
weight1 = variable_with_weight_loss(shape=[dim, 384], stddev=0.04, w1=0.004)
fc_bias1 = tf.Variable(tf.constant(0.1, shape=[384]))
fc_1 = tf.nn.relu(tf.matmul(reshape, weight1) + fc_bias1)

# 第二个全联接层
weight2 = variable_with_weight_loss(shape=[384, 192], stddev=0.04, w1=0.004)
fc_bias2 = tf.Variable(tf.constant(0.1, shape=[192]))
fc_2 = tf.nn.relu(tf.matmul(fc_1, weight2) + fc_bias2)

# 第三个全联接层
# 输出神经元个数为10
weight3 = variable_with_weight_loss(shape=[192, 10], stddev=1 / 192.0, w1=0.0)
fc_bias3 = tf.Variable(tf.constant(0.0, shape=[10]))
result = tf.add(tf.matmul(fc_2, weight3), fc_bias3)

# 计算损失，包括权重参数的正则化损失和交叉熵损失
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=result,
                                                               labels=tf.cast(y_, tf.int64))
# 将每个权重参数的正则化损失求和
weights_with_l2_loss = tf.add_n(tf.get_collection("losses"))
loss = tf.reduce_mean(cross_entropy) + weights_with_l2_loss
train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)

# 该函数用来输出topk的准确率，k默认为1，就是输出分类准确度最大的一个数值
top_k_op = tf.nn.in_top_k(result, y_, 1)

with tf.Session() as sess:
    tf.global_variables_initializer().run()

    # 开启多线程，因为在构建batch中使用了多线程
    tf.train.start_queue_runners()

    for step in range(max_steps):
        start_time = time.time()
        # 先生成训练数据
        image_batch, label_batch = sess.run([images_train, labels_train])
        _, loss_value = sess.run([train_op, loss], feed_dict={x: image_batch, y_: label_batch})
        duration = time.time() - start_time

        if step % 100 == 0:
            # 计算每秒能训练的样本数量
            examples_per_sec = batch_size / duration
            sec_per_batch = float(duration)

            # 打印每一batch训练的耗时：
            print("step %d, loss = %.2f (%.1f examples/sec; %.3f sec/batch)" % (step, loss_value, examples_per_sec,
                                                                                sec_per_batch))
