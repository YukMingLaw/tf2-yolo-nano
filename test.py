import tensorflow as tf
import numpy as np
from model.model_full import yoloNano
from model.base_layers import yolo_eval
import cv2
import time

anchors = np.array([[6.,9.],[8.,13.],[11.,16.],[14,22],[17,37],[21,26],[29,38],[39,62],[79,99]],dtype='float32')
img_size = 416
num_classes = 1

def main():
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    train_model, test_model = yoloNano(anchors, input_size=416, num_classes = num_classes)
    test_model.summary()
    test_model.load_weights('./model_save/multi_coco_car(no crop no rotate).h5')
    img = cv2.imread('./test_img/Untitled Folder/2.jpg')
    org_h = img.shape[0]
    org_w = img.shape[1]
    max_side = max(org_h, org_w)
    if org_h > org_w:
        scale = org_w / max_side
        pts1 = np.array([[0, 0], [org_w, 0], [0, org_h]], dtype=np.float32)
        pts2 = np.array(
            [[img_size * (1 - scale) / 2, 0], [img_size * (1 + scale) / 2, 0],
             [img_size * (1 - scale) / 2, img_size]],
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
    pred_img = img[np.newaxis, :]
    yolo_output = test_model.predict(pred_img)

    boxes_, scores_, classes_ = yolo_eval(yolo_outputs=yolo_output, anchors=anchors, num_classes=num_classes,
                                          image_shape=np.array([img_size, img_size]), score_threshold=0.7)
    for box in boxes_[0]:
        ymin = int(box[0])
        xmin = int(box[1])
        ymax = int(box[2])
        xmax = int(box[3])
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0))
    cv2.imshow('pred', img)
    cv2.waitKey(0)

if __name__ == '__main__':
    main()
