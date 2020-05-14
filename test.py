import tensorflow as tf
import numpy as np
from model.model import yoloNano
from model.base_layers import yolo_eval
import cv2

anchors = np.array([[6.,9.],[8.,13.],[11.,16.],[14.,22.],[17.,37.],[21.,26.],[29.,38.],[39.,62.],[79.,99.]],dtype='float32')

def main():
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    model, debug_model = yoloNano(anchors, input_size=416, num_classes=1)
    debug_model.load_weights('./model_save/save_model.h5')
    img = cv2.imread('4.jpg')
    img_org = cv2.resize(img,(416,416))
    img = img_org / 255.0
    img = img[np.newaxis,:]
    yolo_output = debug_model(img)
    boxes_, scores_, classes_ = yolo_eval(yolo_output,anchors,1,np.array([416,416]))
    for box in boxes_[0]:
        xmin = int(box[0])
        ymin = int(box[1])
        xmax = int(box[2])
        ymax = int(box[3])
        cv2.rectangle(img_org,(ymin,xmin),(ymax,xmax),(0,255,0))
    cv2.imshow('pred',img_org)
    cv2.waitKey(0)

if __name__ == '__main__':
    main()