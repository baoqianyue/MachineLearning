import os
import tensorflow as tf
import numpy as np
import PIL
from PIL import Image
from model import DCGAN
import scipy.misc
import provider

output_dir = './face_output'  # 输出文件夹
src_dir = 'D:/datasets/subCeleA/filelist.txt'  # 数据集文件夹
batch_size = 32
n_noise = 100
total_epoch = 5
img_size = 64


def save_img(x, nh_nw, save_path='./face_output/sample.jpg'):
    nh, nw = nh_nw
    h, w = x.shape[1], x.shape[2]
    img = np.zeros((h * nh, w * nw, 3))

    for n, x in enumerate(x):
        j = int(n / nw)
        i = int(n % nw)
        img[j * h: j * h + h, i * w: i * w + w, :] = x

    scipy.misc.imsave(save_path, img)


def main():
    # 创建输出文件夹
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    database = provider.DBreader(src_dir, batch_size, resize=[64, 64, 3], labeled=False)
    sess = tf.Session()
    model = DCGAN(sess, batch_size)
    sess.run(tf.global_variables_initializer())

    total_batch = database.total_batch
    save_img_shape = 14 * 14
    noise_z = np.random.normal(size=(save_img_shape, n_noise))

    loss_d = 0.0
    loss_g = 0.0
    for epoch in range(total_epoch):
        for step in range(total_batch):
            batch_imgs = database.next_batch()
            batch_imgs = batch_imgs / 127.5 - 1  # 规划到-1 ~ 1
            noise_g = np.random.normal(size=(batch_size, n_noise))
            noise_d = np.random.normal(size=(batch_size, n_noise))

            loss_d = model.train_discriminator(batch_imgs, noise_d)
            loss_g = model.train_generator(noise_g)

            if epoch == 0 and step < 200:
                adventage = 2
            else:
                adventage = 1

            if step % adventage == 0:
                loss_d = model.train_discriminator(batch_imgs, noise_d)  # Train Discriminator and get the loss value
            loss_g = model.train_generator(noise_g)  # Train Generator and get the loss value

            print('Epoch: [', epoch, '/', total_epoch, '], ', 'Step: [', step, '/', total_batch, '], D_loss: ',
                  loss_d, ', G_loss: ', loss_g)

            if step == 0 or (step + 1) % 10 == 0:
                generated_samples = model.generator_sample(noise_z, batch_size=save_img_shape)
                savepath = output_dir + '/output_' + 'EP' + str(epoch).zfill(3) + "_Batch" + str(step).zfill(6) + '.jpg'
                save_img(generated_samples, (14, 14), save_path=savepath)


if __name__ == "__main__":
    main()
