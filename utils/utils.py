import numpy as np
import tensorflow as tf
import cv2
import math

def preprocess_true_boxes(true_boxes, input_shape, anchors, num_classes):
    '''Preprocess true boxes to training input format
    Parameters
    ----------
    true_boxes: array, shape=(m, T, 5)
        Absolute x_min, y_min, x_max, y_max, class_id relative to input_shape.
    input_shape: array-like, hw, multiples of 32
    anchors: array, shape=(N, 2), wh
    num_classes: integer
    Returns
    -------
    y_true: list of array, shape like yolo_outputs, xywh are reletive value
    '''
    assert (true_boxes[..., 4] < num_classes).all(), 'class id must be less than num_classes'
    num_layers = len(anchors)//3 # default setting
    anchor_mask = [[6,7,8], [3,4,5], [0,1,2]] if num_layers==3 else [[3,4,5], [1,2,3]]

    true_boxes = np.array(true_boxes, dtype='float32')
    input_shape = np.array(input_shape, dtype='int32')

    boxes_wh = true_boxes[..., 2:4]

    m = true_boxes.shape[0]
    grid_shapes = [input_shape//{0:32, 1:16, 2:8}[l] for l in range(num_layers)]
    y_true = [np.zeros((m,grid_shapes[l][0],grid_shapes[l][1],len(anchor_mask[l]),5+num_classes),dtype='float32') for l in range(num_layers)]

    # Expand dim to apply broadcasting.
    anchors = anchors / input_shape[0]
    anchors = np.expand_dims(anchors, 0)
    anchor_maxes = anchors / 2.
    anchor_mins = -anchor_maxes
    valid_mask = boxes_wh[..., 0] > 0

    for b in range(m):
        # Discard zero rows.
        wh = boxes_wh[b, valid_mask[b]]
        if len(wh)==0: continue
        # Expand dim to apply broadcasting.
        wh = np.expand_dims(wh, -2)
        box_maxes = wh / 2.
        box_mins = -box_maxes

        intersect_mins = np.maximum(box_mins, anchor_mins)
        intersect_maxes = np.minimum(box_maxes, anchor_maxes)
        intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
        box_area = wh[..., 0] * wh[..., 1]
        anchor_area = anchors[..., 0] * anchors[..., 1]
        iou = intersect_area / (box_area + anchor_area - intersect_area)

        # Find best anchor for each true box
        best_anchor = np.argmax(iou, axis=-1)

        for t, n in enumerate(best_anchor):
            for l in range(num_layers):
                if n in anchor_mask[l]:
                    i = np.floor(true_boxes[b,t,0]*grid_shapes[l][1]).astype('int32')
                    j = np.floor(true_boxes[b,t,1]*grid_shapes[l][0]).astype('int32')
                    if i == grid_shapes[l][1] or j == grid_shapes[l][0]:
                        i = i - 1
                        j = j - 1
                    k = anchor_mask[l].index(n)
                    c = true_boxes[b,t, 4].astype('int32')
                    y_true[l][b, j, i, k, 0:4] = true_boxes[b,t, 0:4]
                    y_true[l][b, j, i, k, 4] = 1
                    y_true[l][b, j, i, k, 5+c] = 1

    return y_true

def box_iou(b1, b2):
    '''Return iou tensor
    Parameters
    ----------
    b1: tensor, shape=(i1,...,iN, 4), xywh
    b2: tensor, shape=(j, 4), xywh
    Returns
    -------
    iou: tensor, shape=(i1,...,iN, j)
    '''

    # Expand dim to apply broadcasting.
    b1 = tf.expand_dims(b1, -2)
    b1_xy = b1[..., :2]
    b1_wh = b1[..., 2:4]
    b1_wh_half = b1_wh/2.
    b1_mins = b1_xy - b1_wh_half
    b1_maxes = b1_xy + b1_wh_half

    # Expand dim to apply broadcasting.
    b2 = tf.expand_dims(b2, 0)
    b2_xy = b2[..., :2]
    b2_wh = b2[..., 2:4]
    b2_wh_half = b2_wh/2.
    b2_mins = b2_xy - b2_wh_half
    b2_maxes = b2_xy + b2_wh_half

    intersect_mins = tf.maximum(b1_mins, b2_mins)
    intersect_maxes = tf.minimum(b1_maxes, b2_maxes)
    intersect_wh = tf.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    b1_area = b1_wh[..., 0] * b1_wh[..., 1]
    b2_area = b2_wh[..., 0] * b2_wh[..., 1]
    iou = intersect_area / (b1_area + b2_area - intersect_area)

    return iou

def rotate_img(img,angle):
    h = img.shape[0]
    w = img.shape[1]
    if w > h:
        x = (h / 2) / (h / w + math.fabs(math.tan(math.radians(angle))))
        scale = w / 2 / x
    else:
        x = (w / 2) / (w / h + math.fabs(math.tan(math.radians(angle))))
        scale = h / 2 / x

    M = cv2.getRotationMatrix2D((w/2,h/2),angle,scale)
    img = cv2.warpAffine(img, M, (w, h))
    return img,M,scale

def get_rottate_label(box,img_w,img_h,M):
    x = box[0] * img_w
    y = box[1] * img_h
    w = box[2] * img_w
    h = box[3] * img_h

    pts = np.zeros((4,2),dtype=np.float)

    pts[0][0] = x - w / 2
    pts[0][1] = y - h / 2
    pts[1][0] = x + w / 2
    pts[1][1] = y - h / 2
    pts[2][0] = x + w / 2
    pts[2][1] = y + h / 2
    pts[3][0] = x - w / 2
    pts[3][1] = y + h / 2

    pts = np.dot(M[:,:2], pts.transpose()) + M[:,2:]
    pts = pts.transpose()

    if M[0][1] > 0:
        pt_l_t_x = pts[0][0]
        pt_l_t_y = pts[1][1]
        pt_r_b_x = pts[2][0]
        pt_r_b_y = pts[3][1]
    else:
        pt_l_t_x = pts[3][0]
        pt_l_t_y = pts[0][1]
        pt_r_b_x = pts[1][0]
        pt_r_b_y = pts[2][1]
    x = (pt_r_b_x + pt_l_t_x) / 2 / img_w
    y = (pt_r_b_y + pt_l_t_y) / 2 / img_h
    w = (pt_r_b_x - pt_l_t_x) / img_w
    h = (pt_r_b_y - pt_l_t_y) / img_h

    return [x,y,w,h]


