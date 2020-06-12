import cv2
import numpy as np
from generator.YoloGenerator import YoloGenerator
from utils.visual_effect_preprocess import VisualEffect
from utils.misc_effect_preprocess import MiscEffect

with open('/home/cvos/Datasets/coco_car/train.txt') as f:
    _line = f.readlines()
train_set = [i.rstrip('\n') for i in _line]
train_generator = YoloGenerator(train_list = train_set,
                                anchors=[],
                                shuffle=False,
                                num_classes = 1,
                                batch_size=1,
                                multi_scale=True,
                                visual_effect = VisualEffect(),
                                misc_effect = MiscEffect(border_value=0),
                                debug=True)
for i in range(len(train_set)):
    _b_img,box = train_generator.__getitem__(i)
    img_1 = _b_img[0]
    for box in box[0]:
        cv2.rectangle(img_1,(int((box[0]-box[2]/2) * img_1.shape[1]),int((box[1]-box[3]/2) * img_1.shape[0])),
                            (int((box[0]+box[2]/2) * img_1.shape[1]),int((box[1]+box[3]/2) * img_1.shape[0])),(0,255,0))
        # cv2.putText(img_1,''+ str(box[4]),(int((box[0]-box[2]/2)*416),int((box[1]-box[3]/2)*416)),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255))
    cv2.imshow('vis',img_1)
    cv2.waitKey(1000)