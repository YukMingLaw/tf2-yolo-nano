import cv2
import numpy as np
from dataset.YoloGenerator import YoloGenerator

with open('./coco_car/train.txt') as f:
    _line = f.readlines()
train_set = [i.rstrip('\n') for i in _line]
train_generator = YoloGenerator(train_list=train_set,anchors=[],shuffle=True,batch_size=1,debug=True)
_b_img,box = train_generator.__getitem__(0)

img_1 = _b_img[0]
for box in box[0]:
    cv2.rectangle(img_1,(int((box[0]-box[2]/2)*416),int((box[1]-box[3]/2)*416)),(int((box[0]+box[2]/2)*416),int((box[1]+box[3]/2)*416)),(255,int(box[4]) * 50,0))
cv2.imshow('vis',img_1)
cv2.waitKey(0)