import os
import tensorflow as tf

num_classes = 10
num_examples_pre_epoch_for_train = 50000
num_examples_pre_epoch_for_eval = 10000


class CIFAR10Record(object):
    pass


def read_cifar10(file_queue):
    result = CIFAR10Record()
    label_bytes = 1  # cifar100数据集此处为2
    result.height = 32
    result.width = 32
    result.depth = 3
    # size为3072
    image_bytes = result.height * result.width * result.depth
    # record_bytes是图像数据加上标签数据:3073
    record_bytes = label_bytes + image_bytes

    # 该类适合读取固定长度字节数的数据
    reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
    # 将文件队列传入
    result.key, value = reader.read(file_queue)

    # 得到的value包含多个标签数据和多个图像数据的字符串
    # decode_raw方法可以将字符串解析成图像对应的像素数组
    record_bytes = tf.decode_raw(value, tf.uint8)

    # 将解析后得到的数组的第一个元素类型转换int类型，这是标签数据
    # tf.strided_slice函数截取范围[begin, end)
    result.label = tf.cast(tf.strided_slice(record_bytes, [0], [label_bytes]), tf.int32)

    # 截取剩下的图像数据，并转换shape为[depth, height, width]
    depth_major = tf.reshape(tf.strided_slice(record_bytes, [label_bytes], [label_bytes + image_bytes]),
                             shape=[result.depth, result.height, result.width])
    # 将shape修改为[height, width, depth]
    result.uint8image = tf.transpose(depth_major, [1, 2, 0])
    return result


def inputs(data_dir, batch_size, distorted):
    """
    输入函数，将原始数据传入，可以做数据增强
    :param data_dir:
    :param batch_size:
    :param distorted: 是否进行数据增强
    :return:
    """
    filenames = [os.path.join(data_dir, "data_batch_%d.bin" % i) for i in range(1, 6)]
    # 创建一个文件队列,并调用上面的读取方法
    file_queue = tf.train.string_input_producer(filenames)
    read_input = read_cifar10(file_queue)

    # 将图像数据转换为float
    reshaped_image = tf.cast(read_input.uint8image, tf.float32)
    num_examples_per_epoch = num_examples_pre_epoch_for_train

    # 对图像数据做增强
    if distorted is not None:
        # 将[32, 32, 3]的图像随机裁剪成[24, 24, 3]的图像
        cropped_image = tf.random_crop(reshaped_image, [24, 24, 3])
        # 随机左右翻转图像
        flipped_image = tf.image.random_flip_left_right(cropped_image)

        # 使用random_brightness调整图像亮度
        adjusted_brightness = tf.image.random_brightness(flipped_image, max_delta=0.8)

        # 调整对比度
        adjusted_contrast = tf.image.random_contrast(adjusted_brightness, lower=0.2, upper=1.8)

        # 标准化图像
        # 这个方法是对每一个像素减去均值并除以像素方差
        float_image = tf.image.per_image_standardization(adjusted_contrast)

        # 设置图像数据及label的形状
        float_image.set_shape([24, 24, 3])
        read_input.label.set_shape([1])

        # 最小队列文件数量4000
        min_queue_examples = int(num_examples_pre_epoch_for_eval * 0.4)
        print('Filling queue with %d CIFAR images before starting to train. ' % min_queue_examples)

        # 使用shuffle随机产生一个batch的data和label
        images_train, labels_train = tf.train.shuffle_batch([float_image, read_input.label],
                                                            batch_size=batch_size, num_threads=16,
                                                            capacity=min_queue_examples + 3 * batch_size,
                                                            min_after_dequeue=min_queue_examples)

        return images_train, tf.reshape(labels_train, [batch_size])

    # 不对图像做增强，直接随机裁剪，然后标准化
    else:
        resized_image = tf.image.resize_image_with_crop_or_pad(reshaped_image, 24, 24)
        float_image = tf.image.per_image_standardization(resized_image)

        # 设置图像数据与label
        float_image.set_shape([24, 24, 3])
        read_input.label.set_shape([1])

        min_queue_examples = int(num_examples_pre_epoch_for_eval * 0.4)

        # 使用batch函数生成test_batch
        images_test, labels_test = tf.train.batch([float_image, read_input.label],
                                                  batch_size=batch_size, num_threads=16,
                                                  capacity=min_queue_examples + 3 * batch_size)
        return images_test, tf.reshape(labels_test, [batch_size])


