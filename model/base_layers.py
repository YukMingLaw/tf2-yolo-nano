import tensorflow as tf
from tensorflow.keras.layers import Conv2D,DepthwiseConv2D,Dense,Input,BatchNormalization,AvgPool2D,UpSampling2D,Concatenate,LeakyReLU
from utils.utils import box_iou
from tensorflow.keras.regularizers import l2
import numpy as np
import math

def conv1x1(filters,bn=True,decay=0.001):
    if bn == True:
        return tf.keras.Sequential(
            [Conv2D(filters,kernel_size=(1,1),use_bias=False,padding='same',kernel_regularizer=l2(decay)),
            BatchNormalization(),
            LeakyReLU()]
        )
    else:
        return Conv2D(filters,kernel_size=(1,1),use_bias=False,padding='same',kernel_regularizer=l2(decay))

def conv3x3(filters,stride,bn=True,decay=0.001):
    if bn == True:
        return tf.keras.Sequential(
            [Conv2D(filters,kernel_size=(3,3),strides=stride,use_bias=False,padding='same',kernel_regularizer=l2(decay)),
            BatchNormalization(),
            LeakyReLU()]
        )
    else:
        return Conv2D(filters,kernel_size=(3,3),use_bias=False,padding='same',kernel_regularizer=l2(decay))

def sepconv3x3(neck_channels,output_channels,stride=(1,1),expantion=0.75,decay=0.001):
    return tf.keras.Sequential([
        #Conv2D(math.ceil(neck_channels * expantion),kernel_size=(1,1),use_bias=False,padding='same',kernel_regularizer=l2(decay)),
        Conv2D(math.ceil(output_channels * expantion),kernel_size=(1,1),use_bias=False,padding='same',kernel_regularizer=l2(decay)),
        BatchNormalization(),
        LeakyReLU(),
        DepthwiseConv2D(kernel_size=(3,3),padding='same',strides=stride,kernel_regularizer=l2(decay)),
        BatchNormalization(),
        LeakyReLU(),
        Conv2D(output_channels,kernel_size=(1,1),use_bias=False,padding='same',kernel_regularizer=l2(decay)),
        BatchNormalization()
    ])

class PEP(tf.keras.layers.Layer):
    def __init__(self,filters,neck_filters,**kwargs):
        super(PEP, self).__init__(**kwargs)
        self.filters = filters
        self.neck_filters = neck_filters
        self.conv = conv1x1(self.neck_filters)
        self.sepconv = sepconv3x3(self.neck_filters,self.filters)

    def get_config(self):
        config = {"filters":self.filters,"neck_filters":self.neck_filters,"conv": self.conv,"sepconv":self.sepconv}
        base_config = super(PEP, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def call(self, input):
        x = self.conv(input)
        x = self.sepconv(x)
        if input.shape[-1] == x.shape[-1]:
            return input + x
        else:
            return x

class EP(tf.keras.layers.Layer):
    def __init__(self,filters,stride=(1,1),**kwargs):
        super(EP, self).__init__(**kwargs)
        self.filters = filters
        self.stride = stride

    def build(self, input_shape):
        self.input_filters = input_shape[-1]
        self.sepconv = sepconv3x3(self.input_filters, self.filters, stride=self.stride)

    def call(self, input):
        if self.input_filters == self.filters:
            return input + self.sepconv(input)
        else:
            return self.sepconv(input)
    def get_config(self):
        config = {"sepconv" : self.sepconv,"input_filters" : self.input_filters,"stride":self.stride,"filters":self.filters}
        base_config = super(EP, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class FCA(tf.keras.layers.Layer):
    def __init__(self,reduction_ratio,decay=0.001,**kwargs):
        super(FCA, self).__init__(**kwargs)
        self.reduction_ratio = reduction_ratio
        self.decay = decay

    def build(self, input_shape):
        n,h,w,c = input_shape
        self.dense_units = c // self.reduction_ratio
        self.avg_pool = AvgPool2D(pool_size=(h,w))
        self.fc = tf.keras.Sequential([
            Dense(units=self.dense_units, activation='relu', use_bias=False,kernel_regularizer=l2(self.decay)),
            Dense(units=c,activation='sigmoid',use_bias=False,kernel_regularizer=l2(self.decay))
        ])

    def call(self, input):
        x = self.avg_pool(input)
        x = self.fc(x)
        return input * x

    def get_config(self):
        config = {"reduction_ratio " : self.reduction_ratio,"dense_units" : self.dense_units,"avg_pool":self.avg_pool,"fc":self.fc}
        base_config = super(FCA, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

def yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape):
    '''Get corrected boxes'''
    box_yx = box_xy[..., ::-1]
    box_hw = box_wh[..., ::-1]
    input_shape = tf.cast(input_shape, tf.keras.backend.dtype(box_yx))
    image_shape = tf.cast(image_shape, tf.keras.backend.dtype(box_yx))
    new_shape = tf.round(image_shape * tf.keras.backend.min(input_shape/image_shape))
    offset = (input_shape-new_shape)/2./input_shape
    scale = input_shape/new_shape
    box_yx = (box_yx - offset) * scale
    box_hw *= scale

    box_mins = box_yx - (box_hw / 2.)
    box_maxes = box_yx + (box_hw / 2.)
    boxes =  tf.keras.layers.concatenate([
        box_mins[..., 0:1],  # y_min
        box_mins[..., 1:2],  # x_min
        box_maxes[..., 0:1],  # y_max
        box_maxes[..., 1:2]  # x_max
    ])

    # Scale boxes back to original image shape.
    boxes *= tf.keras.layers.concatenate([image_shape, image_shape])
    return boxes

def yolo_boxes_and_scores(feats, anchors, num_classes, input_shape, image_shape):
    '''Process Conv layer output'''
    box_xy, box_wh, box_confidence, box_class_probs = yololayer(feats,anchors, num_classes, input_shape)
    boxes = yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape)
    boxes = tf.reshape(boxes, [-1, 4])
    box_scores = box_confidence * box_class_probs
    box_scores = tf.reshape(box_scores, [-1, num_classes])
    return boxes, box_scores

def yololayer(feats, anchors, num_classes, input_shape, calc_loss=False):
    """Convert final layer features to bounding box parameters."""
    num_anchors = len(anchors)
    # Reshape to batch, height, width, num_anchors, box_params.
    anchors_tensor = tf.reshape(tf.constant(anchors), [1, 1, 1, num_anchors, 2])

    grid_shape = tf.shape(feats)[1:3] # height, width
    grid_y = tf.tile(tf.reshape(tf.keras.backend.arange(0, stop=grid_shape[0]), [-1, 1, 1, 1]),[1, grid_shape[1], 1, 1])
    grid_x = tf.tile(tf.reshape(tf.keras.backend.arange(0, stop=grid_shape[1]), [1, -1, 1, 1]),[grid_shape[0], 1, 1, 1])
    grid = tf.keras.layers.concatenate([grid_x, grid_y])
    grid = tf.cast(grid, tf.float32)

    feats = tf.reshape(feats, [-1, grid_shape[0], grid_shape[1], num_anchors, num_classes + 5])

    # Adjust preditions to each spatial grid point and anchor size.
    box_xy = (tf.sigmoid(feats[..., :2]) + grid) / tf.cast(grid_shape[::-1], tf.float32)
    box_wh = tf.exp(feats[..., 2:4]) * anchors_tensor / tf.cast(input_shape[::-1], tf.float32)
    box_confidence = tf.sigmoid(feats[..., 4:5])
    box_class_probs = tf.sigmoid(feats[..., 5:])

    if calc_loss == True:
        return grid, feats, box_xy, box_wh
    return box_xy, box_wh, box_confidence, box_class_probs

def yolo_loss(args, anchors, num_classes, ignore_thresh=.5, print_loss=False):
    '''Return yolo_loss tensor
    Parameters
    ----------
    yolo_outputs: list of tensor, the output of yolo_body or tiny_yolo_body
    y_true: list of array, the output of preprocess_true_boxes
    anchors: array, shape=(N, 2), wh
    num_classes: integer
    ignore_thresh: float, the iou threshold whether to ignore object confidence loss
    Returns
    -------
    loss: tensor, shape=(1,)
    '''
    num_layers = len(anchors)//3 # default setting
    yolo_outputs = args[:num_layers]
    y_true = args[num_layers:]
    anchor_mask = [[6,7,8], [3,4,5], [0,1,2]] if num_layers==3 else [[3,4,5], [1,2,3]]
    input_shape = tf.cast(tf.shape(yolo_outputs[0])[1:3] * 32, tf.keras.backend.dtype(y_true[0]))
    grid_shapes = [tf.cast(tf.shape(yolo_outputs[l])[1:3], tf.keras.backend.dtype(y_true[0])) for l in range(num_layers)]
    loss = 0
    m = tf.shape(yolo_outputs[0])[0] # batch size, tensor
    mf = tf.cast(m, tf.keras.backend.dtype(yolo_outputs[0]))

    for l in range(num_layers):
        object_mask = y_true[l][..., 4:5]
        true_class_probs = y_true[l][..., 5:]

        grid, raw_pred, pred_xy, pred_wh = yololayer(yolo_outputs[l],anchors[anchor_mask[l]], num_classes, input_shape, calc_loss=True)
        pred_box = tf.keras.layers.concatenate([pred_xy, pred_wh])

        # Darknet raw box to calculate loss.
        raw_true_xy = y_true[l][..., :2]*grid_shapes[l][::-1] - grid
        raw_true_wh = tf.math.log(y_true[l][..., 2:4] / anchors[anchor_mask[l]] * input_shape[::-1])
        raw_true_wh = tf.keras.backend.switch(object_mask, raw_true_wh, tf.keras.backend.zeros_like(raw_true_wh)) # avoid log(0)=-inf
        box_loss_scale = 2 - y_true[l][...,2:3]*y_true[l][...,3:4]

        # Find ignore mask, iterate over each of batch.
        ignore_mask = tf.TensorArray(tf.keras.backend.dtype(y_true[0]), size=1, dynamic_size=True)
        object_mask_bool = tf.cast(object_mask, 'bool')
        def loop_body(b, ignore_mask):
            true_box = tf.boolean_mask(y_true[l][b,...,0:4], object_mask_bool[b,...,0])
            iou = box_iou(pred_box[b], true_box)
            best_iou = tf.keras.backend.max(iou, axis=-1)
            ignore_mask = ignore_mask.write(b, tf.cast(best_iou<ignore_thresh, tf.keras.backend.dtype(true_box)))
            return b+1, ignore_mask
        _, ignore_mask = tf.while_loop(lambda b,*args: b<m, loop_body, [0, ignore_mask])
        ignore_mask = ignore_mask.stack()
        ignore_mask = tf.expand_dims(ignore_mask, -1)

        # tf.keras.binary_crossentropy is helpful to avoid exp overflow.
        xy_loss = object_mask * box_loss_scale * tf.expand_dims(tf.keras.losses.binary_crossentropy(raw_true_xy, raw_pred[...,0:2], from_logits=True),axis=-1)
        wh_loss = object_mask * box_loss_scale * 0.5 * tf.expand_dims(tf.keras.losses.mean_squared_error(raw_true_wh,raw_pred[...,2:4]),axis=-1)
        confidence_loss = object_mask * tf.expand_dims(tf.keras.losses.binary_crossentropy(object_mask, raw_pred[...,4:5], from_logits=True),axis=-1)+ \
            (1-object_mask) * tf.expand_dims(tf.keras.losses.binary_crossentropy(object_mask, raw_pred[...,4:5], from_logits=True),axis=-1)* ignore_mask
        class_loss = object_mask * tf.expand_dims(tf.keras.losses.binary_crossentropy(true_class_probs, raw_pred[...,5:], from_logits=True),axis=-1)

        xy_loss = tf.keras.backend.sum(xy_loss) / mf
        wh_loss = tf.keras.backend.sum(wh_loss) / mf
        confidence_loss = tf.keras.backend.sum(confidence_loss) / mf
        class_loss = tf.keras.backend.sum(class_loss) / mf
        loss += xy_loss + wh_loss + confidence_loss + class_loss
    loss = tf.reshape(loss,[1])
    return loss

def yolo_eval(yolo_outputs,anchors,num_classes,image_shape,max_boxes=20,score_threshold=.6,iou_threshold=.5):
    """Evaluate YOLO model on given input and return filtered boxes."""

    num_layers = len(yolo_outputs)
    anchor_mask = [[6,7,8], [3,4,5], [0,1,2]] if num_layers==3 else [[3,4,5], [1,2,3]] # default setting
    input_shape = tf.shape(yolo_outputs[0])[1:3] * 32
    boxes = []
    box_scores = []
    for l in range(num_layers):
        _boxes, _box_scores = yolo_boxes_and_scores(yolo_outputs[l],anchors[anchor_mask[l]], num_classes, input_shape, image_shape)
        boxes.append(_boxes)
        box_scores.append(_box_scores)
    boxes = tf.keras.layers.concatenate(boxes, axis=0)
    box_scores = tf.keras.layers.concatenate(box_scores, axis=0)

    mask = box_scores >= score_threshold
    max_boxes_tensor = tf.constant(max_boxes, dtype='int32')
    boxes_ = []
    scores_ = []
    classes_ = []
    for c in range(num_classes):
        # TODO: use keras backend instead of tf.
        class_boxes = tf.boolean_mask(boxes, mask[:, c])
        class_box_scores = tf.boolean_mask(box_scores[:, c], mask[:, c])
        nms_index = tf.image.non_max_suppression(
            class_boxes, class_box_scores, max_boxes_tensor, iou_threshold=iou_threshold)
        class_boxes = tf.gather(class_boxes, nms_index)
        class_box_scores = tf.gather(class_box_scores, nms_index)
        classes = tf.ones_like(class_box_scores, 'int32') * c
        boxes_.append(class_boxes)
        scores_.append(class_box_scores)
        classes_.append(classes)
    if num_classes > 1:
        boxes_ = tf.keras.layers.concatenate(boxes_, axis=0)
        scores_ = tf.keras.layers.concatenate(scores_, axis=0)
        classes_ = tf.keras.layers.concatenate(classes_, axis=0)

    return boxes_, scores_, classes_
