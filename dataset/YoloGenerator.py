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
            random_rotate = False,
            random_crop = False,
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
        self.random_crop = random_crop
        self.random_rotate = random_rotate
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
        batch_img,batch_box = self.load_batch(index)
        if self.debug:
            return batch_img, batch_box
        else:
            gt = preprocess_true_boxes(batch_box,(self.input_size,self.input_size),self.anchors,self.num_classes)
            return [batch_img,*gt],np.zeros(self.batch_size)


    def load_batch(self,index):
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
            train_batch = self.train_list[index * self.batch_size : index * self.batch_size + self.batch_size]

        return self.batch_img_label(train_batch,input_size)

    def batch_img_label(self,batch_list,img_size,max_num_box = 20):
        #load images
        batch_img = []
        batch_label = []
        for path in batch_list:
            img = cv2.imread(path)
            org_h = img.shape[0]
            org_w = img.shape[1]
            if self.random_crop:
                crop_x1 = np.random.randint(0,org_w / 4)
                crop_x2 = np.random.randint(crop_x1 + org_w / 2,org_w)
                crop_y1 = np.random.randint(0,org_h / 4)
                crop_y2 = np.random.randint(crop_y1 + org_h / 2, org_h)
                img = img[crop_y1:crop_y2,crop_x1:crop_x2,:]
                crop_h = img.shape[0]
                crop_w = img.shape[1]
            else:
                crop_x1 = 0
                crop_x2 = org_w
                crop_y1 = 0
                crop_y2 = org_h
                crop_h = org_h
                crop_w = org_w
            max_side = max(crop_h,crop_w)
            if crop_h > crop_w :
                scale = crop_w / max_side
                pts1 = np.array([[0, 0], [crop_w, 0], [0, crop_h]], dtype=np.float32)
                offset1 = img_size * (1 - scale) / 2
                offset2 = img_size * (1 + scale) / 2
                pts2 = np.array([[offset1, 0], [offset2, 0], [offset1, img_size]],
                                dtype=np.float32)
                M = cv2.getAffineTransform(pts1, pts2)
                img = cv2.warpAffine(img, M, (img_size, img_size))
            else:
                scale = crop_h / max_side
                pts1 = np.array([[0, 0], [crop_w, 0], [0, crop_h]], dtype=np.float32)
                offset1 = img_size * (1 - scale) / 2
                offset2 = img_size * (1 + scale) / 2
                pts2 = np.array([[0, offset1], [img_size, offset1], [0, offset2]],
                                dtype=np.float32)
                M = cv2.getAffineTransform(pts1, pts2)
                img = cv2.warpAffine(img, M, (img_size, img_size))

            img = img / 255.0
            batch_img.append(img)

            #load labels
            _path = path.split('.')
            with open(_path[0] + '.txt') as f:
                _line = f.readline()
                boxes = []
                while _line:
                    _line_split = _line.split()
                    obj_class = int(_line_split[0])
                    _box = [float(i) for i in _line_split[1:]]
                    _box[0] = (_box[0] * org_w - crop_x1) / crop_w
                    _box[1] = (_box[1] * org_h - crop_y1) / crop_h
                    _box[2] = _box[2] * org_w / crop_w
                    _box[3] = _box[3] * org_h / crop_h

                    x1 = (_box[0] - _box[2] / 2) * crop_w
                    y1 = (_box[1] - _box[3] / 2) * crop_h
                    x2 = (_box[0] + _box[2] / 2) * crop_w
                    y2 = (_box[1] + _box[3] / 2) * crop_h

                    if x2 < 0 or x1 > crop_w or y2 < 0 or y1 > crop_h:
                        _line = f.readline()
                        continue

                    if x1 < 0:
                        x1 = 0
                        if x2 - x1 <= 5:
                            _line = f.readline()
                            continue
                    if x2 > crop_w:
                        x2 = crop_w
                        if x2 - x1 <= 5:
                            _line = f.readline()
                            continue
                    if y1 < 0:
                        y1 = 0
                        if y2 - y1 <= 5:
                            _line = f.readline()
                            continue
                    if y2 > crop_h:
                        y2 = crop_h
                        if y2 - y1 <= 5:
                            _line = f.readline()
                            continue

                    _box[0] = (x2 + x1) / 2 / crop_w
                    _box[1] = (y2 + y1) / 2 / crop_h
                    _box[2] = (x2 - x1) / crop_w
                    _box[3] = (y2 - y1) / crop_h

                    if crop_h > crop_w:
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