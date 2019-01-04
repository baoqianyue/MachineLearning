# -*- coding: UTF-8 -*-  

"""
DCGAN
"""
import tensorflow as tf


# 定义判别器模型
def discriminator_model():
    # 定义一个网络层序列
    model = tf.keras.models.Sequential()
    # 第一个卷积层
    model.add(tf.keras.layers.Conv2D(
        64,  # 64个filter,输出depth为64
        (5, 5),  # 卷积核大小5x5
        padding='same',  # 填充两圈0
        input_shape=(64, 64, 3)  # 输入图像为64x64x3彩色图像
    ))
    model.add(tf.keras.layers.Activation("tanh"))  # 激活函数tanH
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))  # 图像宽高都缩小2
    model.add(tf.keras.layers.Conv2D(128, (5, 5)))  # 第二个卷积层
    model.add(tf.keras.layers.Activation("tanh"))
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Conv2D(128, (5, 5)))
    model.add(tf.keras.layers.Activation("tanh"))
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(1024))  # 1024全连接层
    model.add(tf.keras.layers.Activation("tanh"))
    model.add(tf.keras.layers.Dense(1))
    model.add(tf.keras.layers.Activation("sigmoid"))

    return model


# 定义生成器模型
# 给定随机噪声来生成图像
def generator_model():
    model = tf.keras.models.Sequential()

    # 输入维度100的随机噪声,输出维度为(神经元个数)是1024的fc层
    model.add(tf.keras.layers.Dense(input_dim=100, units=1024))
    model.add(tf.keras.layers.Activation("tanh"))
    model.add(tf.keras.layers.Dense(128 * 8 * 8))  # 8192神经元
    model.add(tf.keras.layers.BatchNormalization())  # 标准化
    model.add(tf.keras.layers.Activation("tanh"))
