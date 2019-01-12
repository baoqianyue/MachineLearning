import numpy as np
from PIL import Image
from scipy import misc
import os


#
# def load_src(dir):
#     img_data = []
#     for img in os.listdir(dir):
#         img_path = os.path.join(dir, img)
#         src = Image.open(img_path)
#         arr = np.asarray(src, dtype=np.float32)
#         img_data.append(arr)
#     img_data = np.asarray(img_data)
#     return img_data


# class Provider(object):
#     def __init__(self, z_dim, train_data, img_size):
#         self._data = train_data
#         self._example_num = len(self._data)
#         self._z_data = np.random.standard_normal((self._example_num, z_dim))
#         self._indicator = 0
#         self._resize_img(img_size)
#         self._random_shuffle()
#
#     def _random_shuffle(self):
#         x = np.random.permutation(self._example_num)
#         self._data = self._data[x]
#         self._z_data = self._z_data[x]
#
#     def _resize_img(self, img_size):
#         """输入图像是96x96的,这里将他们resize为64x64"""
#         data = np.asarray(self._data, np.uint8)
#         new_data = []
#         for i in range(self._example_num):
#             img = data[i]
#             img = Image.fromarray(img)
#             img = img.resize((img_size, img_size))
#             img = np.asarray(img)
#             new_data.append(img)
#         new_data = np.asarray(new_data, dtype=np.float32)
#         self._data = new_data
#
#     def next_batch(self, batch_size):
#         end_indicator = self._indicator + batch_size
#         if end_indicator > self._example_num:
#             self._random_shuffle()
#             self._indicator = 0
#             end_indicator = self._indicator + batch_size
#         assert end_indicator < self._example_num
#
#         batch_data = self._data[self._indicator: end_indicator]
#         batch_z = self._z_data[self._indicator: end_indicator]
#         self._indicator = end_indicator
#         return batch_data, batch_z
class DBreader:
    def __init__(self, filename, batch_size, resize=0, labeled=True, color=True):
        self.color = color
        self.labeled = labeled

        self.batch_size = batch_size
        # filename: Directory of the filelist.txt(Database list)
        with open(filename) as f:
            tmp_filelist = f.readlines()
            tmp_filelist = [x.strip() for x in tmp_filelist]
            tmp_filelist = np.array(tmp_filelist)

        self.file_len = len(tmp_filelist)

        self.filelist = []
        self.labellist = []
        if self.labeled:
            for i in range(self.file_len):
                splited = (tmp_filelist[i]).split(" ")
                self.filelist.append(splited[0])
                self.labellist.append(splited[1])
        else:
            self.filelist = tmp_filelist

        self.batch_idx = 0
        self.total_batch = int(self.file_len / batch_size)
        self.idx_shuffled = np.arange(self.file_len)
        np.random.shuffle(self.idx_shuffled)
        self.resize = resize

        self.filelist = np.array(self.filelist)
        self.labellist = np.array(self.labellist)

    # Method for get the next batch
    def next_batch(self):
        if self.batch_idx == self.total_batch:
            np.random.shuffle(self.idx_shuffled)
            self.batch_idx = 0

        batch = []
        idx_set = self.idx_shuffled[self.batch_idx * self.batch_size:(self.batch_idx + 1) * self.batch_size]
        batch_filelist = self.filelist[idx_set]

        for i in range(self.batch_size):
            im = misc.imread(batch_filelist[i])
            if self.resize != 0:
                im = misc.imresize(im, self.resize)
                if self.color:
                    if im.shape[2] > 3:
                        im = im[:, :, 0:3]
            batch.append(im)

        if self.labeled:
            label = self.labellist[idx_set]
            self.batch_idx += 1
            return np.array(batch).astype(np.float32), np.array(label).astype(np.int32)

        self.batch_idx += 1
        return np.array(batch).astype(np.float32)
