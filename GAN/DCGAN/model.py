import tensorflow as tf
import numpy as np


class batch_norm(object):
    def __init__(self, epsilon=1e-5, momentum=0.9, name="batch_norm"):
        with tf.variable_scope(name):
            self.epsilon = epsilon
            self.momentum = momentum
            self.name = name

    def __call__(self, x, train=True):
        return tf.contrib.layers.batch_norm(x,
                                            decay=self.momentum,
                                            updates_collections=None,
                                            epsilon=self.epsilon,
                                            scale=True,
                                            is_training=train,
                                            scope=self.name,
                                            reuse=tf.AUTO_REUSE
                                            )


class DCGAN():
    def __init__(self, sess, batch_size):
        self.sess = sess
        self.batch_size = batch_size
        self.learning_rate = 0.0002
        self.beta1 = 0.5

        self.image_shape = [64, 64, 3]  # G生成的图像shape
        self.z_dim = 100  # 随机噪声维度
        self.w1_dim = 1024
        self.w2_dim = 512
        self.w3_dim = 256
        self.w4_dim = 128
        self.w5_dim = 3  # G中每层的通道数

        self.G_w1 = tf.Variable(tf.truncated_normal([4, 4, self.w1_dim, self.z_dim], stddev=0.02), name='G_w1')
        self.G_bn1 = batch_norm(name='G_bn1')
        self.G_w2 = tf.Variable(tf.truncated_normal([4, 4, self.w2_dim, self.w1_dim], stddev=0.02), name='G_w2')
        self.G_bn2 = batch_norm(name='G_bn2')
        self.G_w3 = tf.Variable(tf.truncated_normal([4, 4, self.w3_dim, self.w2_dim], stddev=0.02), name='G_w3')
        self.G_bn3 = batch_norm(name='G_bn3')
        self.G_w4 = tf.Variable(tf.truncated_normal([4, 4, self.w4_dim, self.w3_dim], stddev=0.02), name='G_w4')
        self.G_bn4 = batch_norm(name='G_bn4')
        self.G_w5 = tf.Variable(tf.truncated_normal([4, 4, self.w5_dim, self.w4_dim], stddev=0.02), name='G_w5')

        self.D_w1 = tf.Variable(tf.truncated_normal([4, 4, self.w5_dim, self.w4_dim], stddev=0.02), name='D_w1')

        self.D_w2 = tf.Variable(tf.truncated_normal([4, 4, self.w4_dim, self.w3_dim], stddev=0.02), name='D_w2')
        self.D_bn2 = batch_norm(name='D_bn2')
        self.D_w3 = tf.Variable(tf.truncated_normal([4, 4, self.w3_dim, self.w2_dim], stddev=0.02), name='D_w3')
        self.D_bn3 = batch_norm(name='D_bn3')
        self.D_w4 = tf.Variable(tf.truncated_normal([4, 4, self.w2_dim, self.w1_dim], stddev=0.02), name='D_w4')
        self.D_bn4 = batch_norm(name='D_bn4')

        self.D_w5 = tf.Variable(tf.truncated_normal([4, 4, self.w1_dim, 1], stddev=0.02), name='D_W5')

        self.G_params = [
            self.G_w1,
            self.G_w2,
            self.G_w3,
            self.G_w4,
            self.G_w5
        ]

        self.D_params = [
            self.D_w1,
            self.D_w2,
            self.D_w3,
            self.D_w4,
            self.D_w5
        ]

        self._build_model()

    def generate(self, z):
        h1 = tf.reshape(z, [self.batch_size, 1, 1, self.z_dim])
        h1 = tf.nn.conv2d_transpose(h1, self.G_w1, output_shape=[self.batch_size, 4, 4, self.w1_dim],
                                    strides=[1, 4, 4, 1])
        h1 = tf.nn.relu(self.G_bn1(h1))

        h2 = tf.nn.conv2d_transpose(h1, self.G_w2, output_shape=[self.batch_size, 8, 8, self.w2_dim],
                                    strides=[1, 2, 2, 1])
        h2 = tf.nn.relu(self.G_bn2(h2))

        h3 = tf.nn.conv2d_transpose(h2, self.G_w3, output_shape=[self.batch_size, 16, 16, self.w3_dim],
                                    strides=[1, 2, 2, 1])
        h3 = tf.nn.relu(self.G_bn3(h3))

        h4 = tf.nn.conv2d_transpose(h3, self.G_w4, output_shape=[self.batch_size, 32, 32, self.w4_dim],
                                    strides=[1, 2, 2, 1])
        h4 = tf.nn.relu(self.G_bn4(h4))
        h5 = tf.nn.conv2d_transpose(h4, self.G_w5, output_shape=[self.batch_size, 64, 64, self.w5_dim],
                                    strides=[1, 2, 2, 1])
        x = tf.nn.tanh(h5)
        return x

    def generator_sample(self, noise_z, batch_size=1):
        """生成图像"""
        noise_z = np.array(noise_z).reshape([batch_size, self.z_dim])
        z = tf.placeholder(tf.float32, [batch_size, self.z_dim])
        # 1.先将随机向量reshape为[batch_size, 1, 1, 100]
        l1 = tf.reshape(z, [batch_size, 1, 1, self.z_dim])
        l1 = tf.nn.conv2d_transpose(l1, self.G_w1, output_shape=[batch_size, 4, 4, self.w1_dim],
                                    strides=[1, 4, 4, 1])
        l1 = self.G_bn1(l1)
        l1 = tf.nn.relu(l1)

        l2_output_shape = [batch_size, 8, 8, self.w2_dim]
        l2 = tf.nn.conv2d_transpose(l1, self.G_w2, output_shape=l2_output_shape, strides=[1, 2, 2, 1])
        l2 = self.G_bn2(l2)
        l2 = tf.nn.relu(l2)

        l3_output_shape = [batch_size, 16, 16, self.w3_dim]
        l3 = tf.nn.conv2d_transpose(l2, self.G_w3, output_shape=l3_output_shape, strides=[1, 2, 2, 1])
        l3 = self.G_bn3(l3)
        l3 = tf.nn.relu(l3)

        l4_output_shape = [batch_size, 32, 32, self.w4_dim]
        l4 = tf.nn.conv2d_transpose(l3, self.G_w4, output_shape=l4_output_shape, strides=[1, 2, 2, 1])
        l4 = self.G_bn4(l4)
        l4 = tf.nn.relu(l4)

        l5_output_shape = [batch_size, 64, 64, self.w5_dim]
        l5 = tf.nn.conv2d_transpose(l4, self.G_w5, output_shape=l5_output_shape, strides=[1, 2, 2, 1])

        x = tf.nn.tanh(l5)

        generated_samples = self.sess.run(x, feed_dict={z: noise_z})
        generated_samples = (generated_samples + 1.) / 2
        return generated_samples

    def discriminator(self, image):
        def leaky_relu(x, leak=0.2):
            return tf.maximum(x, x * leak)

        l1 = leaky_relu(tf.nn.conv2d(image, self.D_w1, strides=[1, 2, 2, 1], padding='SAME'))
        l2 = self.D_bn2(tf.nn.conv2d(l1, self.D_w2, strides=[1, 2, 2, 1], padding='SAME'))
        l2 = leaky_relu(l2)
        l3 = self.D_bn3(tf.nn.conv2d(l2, self.D_w3, strides=[1, 2, 2, 1], padding='SAME'))
        l3 = leaky_relu(l3)
        l4 = self.D_bn4(tf.nn.conv2d(l3, self.D_w4, strides=[1, 2, 2, 1], padding='SAME'))
        l5 = leaky_relu(tf.nn.conv2d(l4, self.D_w5, strides=[1, 4, 4, 1], padding='SAME'))
        # flatten
        l5 = tf.reshape(l5, [self.batch_size, 1])
        # 输出判别结果
        y = tf.nn.sigmoid(l5)
        return y

    def _build_model(self):
        # 随机噪声
        self.z = tf.placeholder(tf.float32, [self.batch_size, self.z_dim])
        # 真实图像
        self.image_real = tf.placeholder(tf.float32, [self.batch_size] + self.image_shape)
        gen_image = self.generate(self.z)

        d_real = self.discriminator(self.image_real)
        d_fake = self.discriminator(gen_image)

        # D和G的损失函数, 交叉熵
        self.loss_g = -tf.reduce_mean(tf.log(d_fake))
        self.loss_d = -tf.reduce_mean(tf.log(d_real) + tf.log(1 - d_fake))

        # optimizer
        self.op_g = tf.train.AdamOptimizer(self.learning_rate, self.beta1).minimize(self.loss_g,
                                                                                    var_list=self.G_params)
        self.op_d = tf.train.AdamOptimizer(self.learning_rate, self.beta1).minimize(self.loss_d,
                                                                                    var_list=self.D_params)

    def train_generator(self, z):
        _, loss_val_g = self.sess.run([self.op_g, self.loss_g], feed_dict={self.z: z})
        return loss_val_g

    def train_discriminator(self, batch_img, z):
        _, loss_val_d = self.sess.run([self.op_d, self.loss_d], feed_dict={self.image_real: batch_img,
                                                                           self.z: z})
        return loss_val_d
