import tensorflow as tf
from tensorflow.keras.layers import Conv2D,DepthwiseConv2D,Dense,Input,BatchNormalization,ReLU,AvgPool2D,UpSampling2D,Concatenate

def conv1x1(filters,bn=True):
    if bn == True:
        return tf.keras.Sequential(
            [Conv2D(filters,kernel_size=(1,1),use_bias=False,padding='same'),
            BatchNormalization(),
            ReLU()]
        )
    else:
        return Conv2D(filters,kernel_size=(1,1),use_bias=False,padding='same')

def conv3x3(filters,stride,bn=True):
    if bn == True:
        return tf.keras.Sequential(
            [Conv2D(filters,kernel_size=(3,3),strides=stride,use_bias=False,padding='same'),
            BatchNormalization(),
            ReLU()]
        )
    else:
        return Conv2D(filters,kernel_size=(1,1),use_bias=False,padding='same')

def sepconv3x3(neck_channels,output_channels,stride=(1,1)):
    return tf.keras.Sequential([
        Conv2D(neck_channels,kernel_size=(1,1),use_bias=False,padding='same'),
        BatchNormalization(),
        ReLU(),
        DepthwiseConv2D(kernel_size=(3,3),padding='same',strides=stride),
        BatchNormalization(),
        ReLU(),
        Conv2D(output_channels,kernel_size=(1,1),use_bias=False,padding='same'),
        BatchNormalization()
    ])

class PEP(tf.keras.layers.Layer):
    def __init__(self,filters,neck_filters):
        super(PEP, self).__init__()
        self.conv = conv1x1(neck_filters)
        self.sepconv = sepconv3x3(neck_filters,filters)

    def call(self, input):
        x = self.conv(input)
        x = self.sepconv(x)
        if input.shape[-1] == x.shape[-1]:
            return input + x
        else:
            return x

class EP(tf.keras.layers.Layer):
    def __init__(self,filters,stride=(1,1)):
        super(EP, self).__init__()
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

class FCA(tf.keras.layers.Layer):
    def __init__(self,reduction_ratio):
        super(FCA, self).__init__()
        self.reduction_ratio = reduction_ratio

    def build(self, input_shape):
        n,h,w,c = input_shape
        self.dense_units = c // self.reduction_ratio
        self.avg_pool = AvgPool2D(pool_size=(h,w))
        self.fc = tf.keras.Sequential([
            Dense(units=self.dense_units, activation='relu', use_bias=False),
            Dense(units=c,activation='sigmoid',use_bias=False)
        ])

    def call(self, input):
        x = self.avg_pool(input)
        x = self.fc(x)
        return input * x

def decode(conv_output,num_classes,stride,anchors):
    """
    return tensor of shape [batch_size, output_size, output_size, anchor_per_scale, 5 + num_classes]
            contains (x, y, w, h, score, probability)
    """
    conv_shape       = tf.shape(conv_output)
    batch_size       = conv_shape[0]
    output_size      = conv_shape[1]

    conv_output = tf.reshape(conv_output, (batch_size, output_size, output_size, 3, 5 + num_classes))

    conv_raw_dxdy = conv_output[:, :, :, :, 0:2]
    conv_raw_dwdh = conv_output[:, :, :, :, 2:4]
    conv_raw_conf = conv_output[:, :, :, :, 4:5]
    conv_raw_prob = conv_output[:, :, :, :, 5: ]

    y = tf.tile(tf.range(output_size, dtype=tf.int32)[:, tf.newaxis], [1, output_size])
    x = tf.tile(tf.range(output_size, dtype=tf.int32)[tf.newaxis, :], [output_size, 1])

    xy_grid = tf.concat([x[:, :, tf.newaxis], y[:, :, tf.newaxis]], axis=-1)
    xy_grid = tf.tile(xy_grid[tf.newaxis, :, :, tf.newaxis, :], [batch_size, 1, 1, 3, 1])
    xy_grid = tf.cast(xy_grid, tf.float32)

    pred_xy = (tf.sigmoid(conv_raw_dxdy) + xy_grid) * stride
    pred_wh = (tf.exp(conv_raw_dwdh) * anchors)
    pred_xywh = tf.concat([pred_xy, pred_wh], axis=-1)

    pred_conf = tf.sigmoid(conv_raw_conf)
    pred_prob = tf.sigmoid(conv_raw_prob)

    return tf.concat([pred_xywh, pred_conf, pred_prob], axis=-1)

def bbox_iou(boxes1, boxes2):

    boxes1_area = boxes1[..., 2] * boxes1[..., 3]
    boxes2_area = boxes2[..., 2] * boxes2[..., 3]

    boxes1 = tf.concat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                        boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
    boxes2 = tf.concat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                        boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

    left_up = tf.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down = tf.minimum(boxes1[..., 2:], boxes2[..., 2:])

    inter_section = tf.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area

    return 1.0 * inter_area / union_area

def bbox_giou(boxes1, boxes2):

    boxes1 = tf.concat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                        boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
    boxes2 = tf.concat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                        boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

    boxes1 = tf.concat([tf.minimum(boxes1[..., :2], boxes1[..., 2:]),
                        tf.maximum(boxes1[..., :2], boxes1[..., 2:])], axis=-1)
    boxes2 = tf.concat([tf.minimum(boxes2[..., :2], boxes2[..., 2:]),
                        tf.maximum(boxes2[..., :2], boxes2[..., 2:])], axis=-1)

    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    left_up = tf.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down = tf.minimum(boxes1[..., 2:], boxes2[..., 2:])

    inter_section = tf.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area
    iou = inter_area / union_area

    enclose_left_up = tf.minimum(boxes1[..., :2], boxes2[..., :2])
    enclose_right_down = tf.maximum(boxes1[..., 2:], boxes2[..., 2:])
    enclose = tf.maximum(enclose_right_down - enclose_left_up, 0.0)
    enclose_area = enclose[..., 0] * enclose[..., 1]
    giou = iou - 1.0 * (enclose_area - union_area) / enclose_area

    return giou

def compute_loss(pred, conv, label, bboxes,num_classes,stride):
    iuo_loss_thres = 0.5
    conv_shape  = tf.shape(conv)
    batch_size  = conv_shape[0]
    output_size = conv_shape[1]
    input_size  = stride * output_size
    conv = tf.reshape(conv, (batch_size, output_size, output_size, 3, 5 + num_classes))

    conv_raw_conf = conv[:, :, :, :, 4:5]
    conv_raw_prob = conv[:, :, :, :, 5:]

    pred_xywh     = pred[:, :, :, :, 0:4]
    pred_conf     = pred[:, :, :, :, 4:5]

    label_xywh    = label[:, :, :, :, 0:4]
    respond_bbox  = label[:, :, :, :, 4:5]
    label_prob    = label[:, :, :, :, 5:]

    giou = tf.expand_dims(bbox_giou(pred_xywh, label_xywh), axis=-1)
    input_size = tf.cast(input_size, tf.float32)

    bbox_loss_scale = 2.0 - 1.0 * label_xywh[:, :, :, :, 2:3] * label_xywh[:, :, :, :, 3:4] / (input_size ** 2)
    giou_loss = respond_bbox * bbox_loss_scale * (1- giou)

    iou = bbox_iou(pred_xywh[:, :, :, :, tf.newaxis, :], bboxes[:, tf.newaxis, tf.newaxis, tf.newaxis, :, :])
    max_iou = tf.expand_dims(tf.reduce_max(iou, axis=-1), axis=-1)

    respond_bgd = (1.0 - respond_bbox) * tf.cast( max_iou < iuo_loss_thres, tf.float32 )

    conf_focal = tf.pow(respond_bbox - pred_conf, 2)

    conf_loss = conf_focal * (
            respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox, logits=conv_raw_conf)
            +
            respond_bgd * tf.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox, logits=conv_raw_conf)
    )

    prob_loss = respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels=label_prob, logits=conv_raw_prob)

    giou_loss = tf.reduce_mean(tf.reduce_sum(giou_loss, axis=[1,2,3,4]))
    conf_loss = tf.reduce_mean(tf.reduce_sum(conf_loss, axis=[1,2,3,4]))
    prob_loss = tf.reduce_mean(tf.reduce_sum(prob_loss, axis=[1,2,3,4]))

    return giou_loss, conf_loss, prob_loss