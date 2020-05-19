import tensorflow as tf
import numpy as np
import math
import cv2
from utils.utils import preprocess_true_boxes

class YoloGenerator(tf.keras.utils.Sequence):
    def __init__(
            self,
            train_list=[],
            anchors=[],
            num_classes=1,
            multi_scale=False,
            multi_image_sizes=(320, 352, 384, 416, 448, 480, 512, 544, 576, 608),
            batch_size=1,
            shuffle = True,
            random_transfer = False,
            input_size = 416,
            debug=False
    ):
        self.current_index = 0
        self.train_list = train_list
        self.multi_scale = multi_scale
        self.multi_image_sizes = multi_image_sizes
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.random_trans = random_transfer
        self.input_size = input_size
        self.anchors = anchors
        self.num_classes = num_classes
        self.debug = debug
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
        if self.debug:
            return batch_img, batch_box
        else:
            gt = preprocess_true_boxes(batch_box,(self.input_size,self.input_size),self.anchors,self.num_classes)
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
            org_h = img.shape[0]
            org_w = img.shape[1]
            max_side = max(org_h,org_w)
            if org_h > org_w :
                scale = org_w / max_side
                pts1 = np.array([[0, 0], [org_w, 0], [0, org_h]], dtype=np.float32)
                offset1 = img_size * (1 - scale) / 2
                offset2 = img_size * (1 + scale) / 2
                pts2 = np.array([[offset1, 0], [offset2, 0], [offset1, img_size]],
                                dtype=np.float32)
                M = cv2.getAffineTransform(pts1, pts2)
                img = cv2.warpAffine(img, M, (img_size, img_size))
            else:
                scale = org_h / max_side
                pts1 = np.array([[0, 0], [org_w, 0], [0, org_h]], dtype=np.float32)
                offset1 = img_size * (1 - scale) / 2
                offset2 = img_size * (1 + scale) / 2
                pts2 = np.array([[0, offset1], [img_size, offset1], [0, offset2]],
                                dtype=np.float32)
                M = cv2.getAffineTransform(pts1, pts2)
                img = cv2.warpAffine(img, M, (img_size, img_size))

            img = img / 255.0
            batch_img.append(img)

            _path = path.split('.')
            with open(_path[0] + '.txt') as f:
                _line = f.readline()
                boxes = []
                while _line:
                    _line_split = _line.split()
                    obj_class = int(_line_split[0])
                    _box = [float(i) for i in _line_split[1:]]
                    if org_h > org_w:
                        _box[0] = (_box[0] * img_size * scale + offset1) / img_size
                        _box[2] = _box[2] * scale
                    else:
                        _box[1] = (_box[1] * img_size * scale + offset1) / img_size
                        _box[3] = _box[3] * scale
                    _box.append(obj_class)
                    boxes.append(_box)
                    _line = f.readline()
            box_data = np.zeros((max_num_box,5))
            if len(boxes) > 0:
                if len(boxes) > max_num_box: boxes = boxes[:max_num_box]
                box_data[:len(boxes)] = np.array(boxes)
            batch_label.append(box_data)

        batch_img = np.array(batch_img)
        batch_label = np.array(batch_label)
        return batch_img,batch_label