import tensorflow as tf
from tensorflow.keras.layers import Input,UpSampling2D,Concatenate,Lambda,Conv2D,BatchNormalization,LeakyReLU,DepthwiseConv2D,Add,AvgPool2D,Dense
from .base_layers import yolo_loss

def yoloNano(anchors,input_size=416,num_classes=1):
    #fuck tensorflow 2.x
    #backbone
    input_0 = Input(shape=(input_size,input_size,3))
    input_gt = [Input(shape=(input_size//{0:32, 1:16, 2:8}[l], input_size//{0:32, 1:16, 2:8}[l],len(anchors)//3, num_classes+5)) for l in range(3)]
    x = Conv2D(filters=12,strides=(1,1),kernel_size=(3,3),use_bias=False,padding='same')(input_0)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Conv2D(filters=24,strides=(2,2),kernel_size=(3,3),use_bias=False,padding='same')(x)
    x = BatchNormalization()(x)
    x_0 = LeakyReLU()(x)
    #PEP(7)(208x208x24)
    x = Conv2D(filters=7,strides=(1,1),kernel_size=(1,1),use_bias=False,padding='same')(x_0)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Conv2D(filters=7,strides=(1,1),kernel_size=(1,1),use_bias=False,padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = DepthwiseConv2D(strides=(1,1),kernel_size=(3,3),use_bias=False,padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Conv2D(filters=24,strides=(1,1),kernel_size=(1,1),use_bias=False,padding='same')(x)
    x = BatchNormalization()(x)
    x = x_0 + x
    #EP(104x104x70)
    x = Conv2D(filters=24, strides=(1, 1), kernel_size=(1, 1), use_bias=False, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = DepthwiseConv2D(strides=(2, 2), kernel_size=(3, 3), use_bias=False, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Conv2D(filters=70, strides=(1, 1), kernel_size=(1, 1), use_bias=False, padding='same')(x)
    x_1 = BatchNormalization()(x)
    #PEP(25)(104x104x70)
    x = Conv2D(filters=25, strides=(1, 1), kernel_size=(1, 1), use_bias=False, padding='same')(x_1)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Conv2D(filters=25, strides=(1, 1), kernel_size=(1, 1), use_bias=False, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = DepthwiseConv2D(strides=(1, 1), kernel_size=(3, 3), use_bias=False, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Conv2D(filters=70, strides=(1, 1), kernel_size=(1, 1), use_bias=False, padding='same')(x)
    x = BatchNormalization()(x)
    x_2 = x_1 + x
    # PEP(24)(104x104x70)
    x = Conv2D(filters=24, strides=(1, 1), kernel_size=(1, 1), use_bias=False, padding='same')(x_2)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Conv2D(filters=24, strides=(1, 1), kernel_size=(1, 1), use_bias=False, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = DepthwiseConv2D(strides=(1, 1), kernel_size=(3, 3), use_bias=False, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Conv2D(filters=70, strides=(1, 1), kernel_size=(1, 1), use_bias=False, padding='same')(x)
    x = BatchNormalization()(x)
    x = x_2 + x
    # EP(52x52x150)
    x = Conv2D(filters=70, strides=(1, 1), kernel_size=(1, 1), use_bias=False, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = DepthwiseConv2D(strides=(2, 2), kernel_size=(3, 3), use_bias=False, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Conv2D(filters=150, strides=(1, 1), kernel_size=(1, 1), use_bias=False, padding='same')(x)
    x_3 = BatchNormalization()(x)
    # PEP(56)(52x52x150)
    x = Conv2D(filters=56, strides=(1, 1), kernel_size=(1, 1), use_bias=False, padding='same')(x_3)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Conv2D(filters=56, strides=(1, 1), kernel_size=(1, 1), use_bias=False, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = DepthwiseConv2D(strides=(1, 1), kernel_size=(3, 3), use_bias=False, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Conv2D(filters=150, strides=(1, 1), kernel_size=(1, 1), use_bias=False, padding='same')(x)
    x = BatchNormalization()(x)
    x = x_3 + x
    #Conv1x1
    x = Conv2D(filters=150,kernel_size=(1,1),strides=(1,1),use_bias=False,padding='same')(x)
    x = BatchNormalization()(x)
    x_4 = LeakyReLU()(x)
    #FCA(8)
    x = AvgPool2D(pool_size=(52,52))(x_4)
    x = Dense(units=150 // 8,activation='relu',use_bias=False)(x)
    x = Dense(units=150, activation='sigmoid', use_bias=False)(x)
    x_5 = x_4 * x
    #PEP(73)(52x52x150)
    x = Conv2D(filters=73, strides=(1, 1), kernel_size=(1, 1), use_bias=False, padding='same')(x_5)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Conv2D(filters=73, strides=(1, 1), kernel_size=(1, 1), use_bias=False, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = DepthwiseConv2D(strides=(1, 1), kernel_size=(3, 3), use_bias=False, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Conv2D(filters=150, strides=(1, 1), kernel_size=(1, 1), use_bias=False, padding='same')(x)
    x = BatchNormalization()(x)
    x_6 = x_5 + x
    # PEP(71)(52x52x150)
    x = Conv2D(filters=71, strides=(1, 1), kernel_size=(1, 1), use_bias=False, padding='same')(x_6)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Conv2D(filters=71, strides=(1, 1), kernel_size=(1, 1), use_bias=False, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = DepthwiseConv2D(strides=(1, 1), kernel_size=(3, 3), use_bias=False, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Conv2D(filters=150, strides=(1, 1), kernel_size=(1, 1), use_bias=False, padding='same')(x)
    x = BatchNormalization()(x)
    x_7 = x_6 + x
    # PEP(75)(52x52x150)
    x = Conv2D(filters=75, strides=(1, 1), kernel_size=(1, 1), use_bias=False, padding='same')(x_7)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Conv2D(filters=75, strides=(1, 1), kernel_size=(1, 1), use_bias=False, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = DepthwiseConv2D(strides=(1, 1), kernel_size=(3, 3), use_bias=False, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Conv2D(filters=150, strides=(1, 1), kernel_size=(1, 1), use_bias=False, padding='same')(x)
    x = BatchNormalization()(x)
    x_8 = x_7 + x #output 52x52x150
    #EP(26x26x325)
    x = Conv2D(filters=150, strides=(1, 1), kernel_size=(1, 1), use_bias=False, padding='same')(x_8)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = DepthwiseConv2D(strides=(2, 2), kernel_size=(3, 3), use_bias=False, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Conv2D(filters=325, strides=(1, 1), kernel_size=(1, 1), use_bias=False, padding='same')(x)
    x_9 = BatchNormalization()(x)
    # PEP(132)(26x26x325)
    x = Conv2D(filters=132, strides=(1, 1), kernel_size=(1, 1), use_bias=False, padding='same')(x_9)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Conv2D(filters=132, strides=(1, 1), kernel_size=(1, 1), use_bias=False, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = DepthwiseConv2D(strides=(1, 1), kernel_size=(3, 3), use_bias=False, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Conv2D(filters=325, strides=(1, 1), kernel_size=(1, 1), use_bias=False, padding='same')(x)
    x = BatchNormalization()(x)
    x_10 = x_9 + x
    # PEP(124)(26x26x325)
    x = Conv2D(filters=124, strides=(1, 1), kernel_size=(1, 1), use_bias=False, padding='same')(x_10)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Conv2D(filters=124, strides=(1, 1), kernel_size=(1, 1), use_bias=False, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = DepthwiseConv2D(strides=(1, 1), kernel_size=(3, 3), use_bias=False, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Conv2D(filters=325, strides=(1, 1), kernel_size=(1, 1), use_bias=False, padding='same')(x)
    x = BatchNormalization()(x)
    x_11 = x_10 + x
    # PEP(141)(26x26x325)
    x = Conv2D(filters=141, strides=(1, 1), kernel_size=(1, 1), use_bias=False, padding='same')(x_11)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Conv2D(filters=141, strides=(1, 1), kernel_size=(1, 1), use_bias=False, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = DepthwiseConv2D(strides=(1, 1), kernel_size=(3, 3), use_bias=False, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Conv2D(filters=325, strides=(1, 1), kernel_size=(1, 1), use_bias=False, padding='same')(x)
    x = BatchNormalization()(x)
    x_12 = x_11 + x
    # PEP(140)(26x26x325)
    x = Conv2D(filters=140, strides=(1, 1), kernel_size=(1, 1), use_bias=False, padding='same')(x_12)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Conv2D(filters=140, strides=(1, 1), kernel_size=(1, 1), use_bias=False, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = DepthwiseConv2D(strides=(1, 1), kernel_size=(3, 3), use_bias=False, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Conv2D(filters=325, strides=(1, 1), kernel_size=(1, 1), use_bias=False, padding='same')(x)
    x = BatchNormalization()(x)
    x_13 = x_12 + x
    # PEP(137)(26x26x325)
    x = Conv2D(filters=137, strides=(1, 1), kernel_size=(1, 1), use_bias=False, padding='same')(x_13)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Conv2D(filters=137, strides=(1, 1), kernel_size=(1, 1), use_bias=False, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = DepthwiseConv2D(strides=(1, 1), kernel_size=(3, 3), use_bias=False, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Conv2D(filters=325, strides=(1, 1), kernel_size=(1, 1), use_bias=False, padding='same')(x)
    x = BatchNormalization()(x)
    x_14 = x_13 + x
    # PEP(135)(26x26x325)
    x = Conv2D(filters=135, strides=(1, 1), kernel_size=(1, 1), use_bias=False, padding='same')(x_14)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Conv2D(filters=135, strides=(1, 1), kernel_size=(1, 1), use_bias=False, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = DepthwiseConv2D(strides=(1, 1), kernel_size=(3, 3), use_bias=False, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Conv2D(filters=325, strides=(1, 1), kernel_size=(1, 1), use_bias=False, padding='same')(x)
    x = BatchNormalization()(x)
    x_15 = x_14 + x
    # PEP(133)(26x26x325)
    x = Conv2D(filters=133, strides=(1, 1), kernel_size=(1, 1), use_bias=False, padding='same')(x_15)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Conv2D(filters=133, strides=(1, 1), kernel_size=(1, 1), use_bias=False, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = DepthwiseConv2D(strides=(1, 1), kernel_size=(3, 3), use_bias=False, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Conv2D(filters=325, strides=(1, 1), kernel_size=(1, 1), use_bias=False, padding='same')(x)
    x = BatchNormalization()(x)
    x_16 = x_15 + x
    # PEP(140)(26x26x325)
    x = Conv2D(filters=140, strides=(1, 1), kernel_size=(1, 1), use_bias=False, padding='same')(x_16)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Conv2D(filters=140, strides=(1, 1), kernel_size=(1, 1), use_bias=False, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = DepthwiseConv2D(strides=(1, 1), kernel_size=(3, 3), use_bias=False, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Conv2D(filters=325, strides=(1, 1), kernel_size=(1, 1), use_bias=False, padding='same')(x)
    x = BatchNormalization()(x)
    x_17 = x_16 + x #output 26x26x325
    # EP(13x13x545)
    x = Conv2D(filters=325, strides=(1, 1), kernel_size=(1, 1), use_bias=False, padding='same')(x_17)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = DepthwiseConv2D(strides=(2, 2), kernel_size=(3, 3), use_bias=False, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Conv2D(filters=545, strides=(1, 1), kernel_size=(1, 1), use_bias=False, padding='same')(x)
    x_18 = BatchNormalization()(x)
    # PEP(276)(13x13x545)
    x = Conv2D(filters=276, strides=(1, 1), kernel_size=(1, 1), use_bias=False, padding='same')(x_18)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Conv2D(filters=276, strides=(1, 1), kernel_size=(1, 1), use_bias=False, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = DepthwiseConv2D(strides=(1, 1), kernel_size=(3, 3), use_bias=False, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Conv2D(filters=545, strides=(1, 1), kernel_size=(1, 1), use_bias=False, padding='same')(x)
    x = BatchNormalization()(x)
    x_19 = x_18 + x
    #Conv1x1
    x = Conv2D(filters=230, kernel_size=(1, 1), strides=(1, 1), use_bias=False, padding='same')(x_19)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    # EP(13x13x489)
    x = Conv2D(filters=230, strides=(1, 1), kernel_size=(1, 1), use_bias=False, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = DepthwiseConv2D(strides=(1, 1), kernel_size=(3, 3), use_bias=False, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Conv2D(filters=489, strides=(1, 1), kernel_size=(1, 1), use_bias=False, padding='same')(x)
    x = BatchNormalization()(x)
    # PEP(213)(13x13x469)
    x = Conv2D(filters=213, strides=(1, 1), kernel_size=(1, 1), use_bias=False, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Conv2D(filters=213, strides=(1, 1), kernel_size=(1, 1), use_bias=False, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = DepthwiseConv2D(strides=(1, 1), kernel_size=(3, 3), use_bias=False, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Conv2D(filters=469, strides=(1, 1), kernel_size=(1, 1), use_bias=False, padding='same')(x)
    x = BatchNormalization()(x)
    # Conv1x1
    x = Conv2D(filters=189, kernel_size=(1, 1), strides=(1, 1), use_bias=False, padding='same')(x)
    x = BatchNormalization()(x)
    x_20 = LeakyReLU()(x) #output 13x13x189
    # EP(13x13x462)
    x = Conv2D(filters=189, strides=(1, 1), kernel_size=(1, 1), use_bias=False, padding='same')(x_20)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = DepthwiseConv2D(strides=(1, 1), kernel_size=(3, 3), use_bias=False, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Conv2D(filters=462, strides=(1, 1), kernel_size=(1, 1), use_bias=False, padding='same')(x)
    x = BatchNormalization()(x)
    # feature 13x13x[(num_classes+5)x3]
    feature_13x13 = Conv2D(filters=3 * (num_classes + 5), kernel_size=(1, 1), strides=(1, 1), use_bias=False, padding='same')(x)
    # Conv1x1
    x = Conv2D(filters=105, kernel_size=(1, 1), strides=(1, 1), use_bias=False, padding='same')(x_20)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    # upsampling 26x26x105
    x = UpSampling2D()(x)
    # concatenate
    x = Concatenate()([x,x_17])
    # PEP(113)(26x26x325)
    x = Conv2D(filters=113, strides=(1, 1), kernel_size=(1, 1), use_bias=False, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Conv2D(filters=113, strides=(1, 1), kernel_size=(1, 1), use_bias=False, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = DepthwiseConv2D(strides=(1, 1), kernel_size=(3, 3), use_bias=False, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Conv2D(filters=325, strides=(1, 1), kernel_size=(1, 1), use_bias=False, padding='same')(x)
    x = BatchNormalization()(x)
    # PEP(99)(26x26x207)
    x = Conv2D(filters=99, strides=(1, 1), kernel_size=(1, 1), use_bias=False, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Conv2D(filters=99, strides=(1, 1), kernel_size=(1, 1), use_bias=False, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = DepthwiseConv2D(strides=(1, 1), kernel_size=(3, 3), use_bias=False, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Conv2D(filters=207, strides=(1, 1), kernel_size=(1, 1), use_bias=False, padding='same')(x)
    x = BatchNormalization()(x)
    # Conv1x1
    x = Conv2D(filters=98, kernel_size=(1, 1), strides=(1, 1), use_bias=False, padding='same')(x)
    x = BatchNormalization()(x)
    x_21 = LeakyReLU()(x)
    # EP(13x13x183)
    x = Conv2D(filters=98, strides=(1, 1), kernel_size=(1, 1), use_bias=False, padding='same')(x_21)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = DepthwiseConv2D(strides=(1, 1), kernel_size=(3, 3), use_bias=False, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Conv2D(filters=183, strides=(1, 1), kernel_size=(1, 1), use_bias=False, padding='same')(x)
    x = BatchNormalization()(x)
    # feature 26x26x[(num_classes+5)x3]
    feature_26x26 = Conv2D(filters=3 * (num_classes + 5), kernel_size=(1, 1), strides=(1, 1), use_bias=False,padding='same')(x)
    # Conv1x1
    x = Conv2D(filters=47, kernel_size=(1, 1), strides=(1, 1), use_bias=False, padding='same')(x_21)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    #upsampling
    x = UpSampling2D()(x)
    #concatenate
    x = Concatenate()([x,x_8])
    # PEP(58)(52x52x132)
    x = Conv2D(filters=58, strides=(1, 1), kernel_size=(1, 1), use_bias=False, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Conv2D(filters=58, strides=(1, 1), kernel_size=(1, 1), use_bias=False, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = DepthwiseConv2D(strides=(1, 1), kernel_size=(3, 3), use_bias=False, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Conv2D(filters=132, strides=(1, 1), kernel_size=(1, 1), use_bias=False, padding='same')(x)
    x = BatchNormalization()(x)
    # PEP(52)(52x52x87)
    x = Conv2D(filters=52, strides=(1, 1), kernel_size=(1, 1), use_bias=False, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Conv2D(filters=52, strides=(1, 1), kernel_size=(1, 1), use_bias=False, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = DepthwiseConv2D(strides=(1, 1), kernel_size=(3, 3), use_bias=False, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Conv2D(filters=87, strides=(1, 1), kernel_size=(1, 1), use_bias=False, padding='same')(x)
    x = BatchNormalization()(x)
    # PEP(47)(52x52x93)
    x = Conv2D(filters=47, strides=(1, 1), kernel_size=(1, 1), use_bias=False, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Conv2D(filters=47, strides=(1, 1), kernel_size=(1, 1), use_bias=False, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = DepthwiseConv2D(strides=(1, 1), kernel_size=(3, 3), use_bias=False, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Conv2D(filters=93, strides=(1, 1), kernel_size=(1, 1), use_bias=False, padding='same')(x)
    x = BatchNormalization()(x)
    feature_52x52 = Conv2D(filters=3 * (num_classes + 5), kernel_size=(1, 1), strides=(1, 1), use_bias=False,padding='same')(x)
    #loss layer
    loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.5})([feature_13x13,feature_26x26,feature_52x52, *input_gt])

    debug_model = tf.keras.Model(inputs=input_0,outputs=[feature_13x13,feature_26x26,feature_52x52])
    train_model = tf.keras.Model(inputs=[input_0,*input_gt],outputs=loss)
    return train_model,debug_model

# import numpy as np
# anchors = np.array([[6.,9.],[8.,13.],[11.,16.],[14.,22.],[17.,37.],[21.,26.],[29.,38.],[39.,62.],[79.,99.]],dtype='float32')
# model,_ = yoloNano(anchors,input_size=416,num_classes=1)
# model.summary()