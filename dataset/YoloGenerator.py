import tensorflow as tf
import numpy as np
import math
import cv2
from utils.utils import preprocess_true_boxes
from model.base_layers import yolo_eval

class YoloGenerator(tf.keras.utils.Sequence):
    def __init__(
            self,
            train_list=[],
            multi_scale=False,
            multi_image_sizes=(320, 352, 384, 416, 448, 480, 512, 544, 576, 608),
            batch_size=1,
            shuffle = True,
            random_transfer = False,
            input_size = 416
    ):
        self.current_index = 0
        self.train_list = train_list
        self.multi_scale = multi_scale
        self.multi_image_sizes = multi_image_sizes
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.random_trans = random_transfer
        self.input_size = input_size
        if(len(train_list) == 0):
            print('error train set is empty!')
            exit()
        if(len(train_list) < batch_size):
            print('train set count {0} is less than batch size {1}'.format(len(train_list),batch_size))
            exit()

    def __len__(self):
        return math.ceil(len(self.train_list) / self.batch_size)

    def __getitem__(self, index):
        batch_img,batch_box = self.load_batch()
        gt = preprocess_true_boxes(batch_box,(self.input_size,self.input_size),np.array([[1.,2.],[1.,2.],[1.,2.],[1.,2.],[1.,2.],[1.,2.],[1.,2.],[1.,2.],[1.,2.]],dtype='float32'),3)
        return [batch_img,*gt],np.zeros(self.batch_size)

    def load_batch(self):
        if self.multi_scale:
            input_size = np.random.choice(self.multi_image_sizes)
        else:
            input_size = self.input_size
        if self.shuffle:
            train_batch = []
            for i in range(self.batch_size):
                random_index = np.random.randint(0,len(self.train_list))
                train_batch.append(self.train_list[random_index])
        else:
            train_batch = self.train_list[self.current_index : self.current_index+self.batch_size]
            self.current_index = self.current_index + self.batch_size

        return self.batch_img_label(train_batch,input_size)

    def batch_img_label(self,batch_list,img_size,max_num_box = 20):
        batch_img = []
        batch_label = []
        for path in batch_list:
            img = cv2.imread(path)
            img = cv2.resize(img,(img_size,img_size))
            batch_img.append(img)

            _path = path.split('.')
            with open(_path[0] + '.txt') as f:
                _line = f.readline()
                boxes = []
                while _line:
                    _line_split = _line.split()
                    obj_class = int(_line_split[0])
                    _box = [float(i) for i in _line_split[1:]]
                    _box.append(obj_class)
                    boxes.append(_box)
                    _line = f.readline()
            box_data = np.zeros((max_num_box,5))
            if len(boxes) > max_num_box: boxes = boxes[:max_num_box]
            box_data[:len(boxes)] = np.array(boxes)
            batch_label.append(box_data)

        batch_img = np.array(batch_img)
        batch_label = np.array(batch_label)
        return batch_img,batch_label


with open('../coco/train.txt') as f:
    _line = f.readlines()
train_set = [i.rstrip('\n') for i in _line]
train_generator = YoloGenerator(train_list=train_set,shuffle=True,batch_size=1)
_b_img,_ = train_generator.__getitem__(0)
boxes,scores,cls = yolo_eval(_b_img[1:],np.array([[1.,2.],[1.,2.],[1.,2.],[1.,2.],[1.,2.],[1.,2.],[1.,2.],[1.,2.],[1.,2.]],dtype='float32'),3,np.array([416,416]))
print(boxes)
print(scores)
print(cls)
# img_1 = _b_img[0]
# for box in _b_label[0]:
#     cv2.rectangle(img_1,(int((box[0]-box[2]/2)*416),int((box[1]-box[3]/2)*416)),(int((box[0]+box[2]/2)*416),int((box[1]+box[3]/2)*416)),(255,int(box[4]) * 50,0))
# cv2.imshow('vis',img_1)
# cv2.waitKey(0)
