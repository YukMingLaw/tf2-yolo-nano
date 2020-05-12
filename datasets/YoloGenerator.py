import tensorflow as tf
import numpy as np
import math
class YoloGenerator(tf.keras.utils.Sequence):
    def __init__(
            self,
            train_list=[],
            multi_scale=False,
            multi_image_sizes=(320, 352, 384, 416, 448, 480, 512, 544, 576, 608),
            batch_size=1,
            shuffle = True,
            random_transfer = False,
            input_size=416
    ):
        self.current_index = 0
        self.train_list = train_list
        self.multi_scale = multi_scale
        self.multi_image_sizes = multi_image_sizes
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.random_trans = random_transfer
        self.input_size = input_size

    def __len__(self):
        return math.ceil(len(self.train_list) / self.batch_size)

    def __getitem__(self, index):
        return self.load_batch()

    def load_batch(self):
        if self.multi_scale:
            input_size = np.random.choice(self.multi_image_sizes)
        else:
            input_size = self.input_size
        if self.shuffle:
            a = 1 + 1
        else:
            train_batch = self.train_list[self.current_index : self.current_index+self.batch_size]
            self.current_index = self.current_index + self.batch_size

        return np.array([self.load_img(file_name,input_size) for file_name in train_batch]),np.array([self.load_label(file_name) for file_name in train_batch])

    def load_img(self,img_path,img_size):
        return 0

    def load_label(self,path):
        return 0
