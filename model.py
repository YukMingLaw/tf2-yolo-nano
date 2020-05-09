import tensorflow as tf
from tensorflow.keras.layers import Conv2D,DepthwiseConv2D,Dense,Input,BatchNormalization,ReLU,AvgPool2D

def conv1x1(filters,bn=True):
    if bn == True:
        return tf.keras.Sequential(
            [Conv2D(filters,kernel_size=(1,1),use_bias=False,padding='same'),
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
        Conv2D(output_channels,kernel_size=(1,1),use_bias=False,padding='same')
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


def yoloNano(input_size=416):
    input_0 = Input(shape=(input_size,input_size,3))
    x = Conv2D(filters=12, kernel_size=(3,3), padding='same')(input_0)
    x = Conv2D(filters=24, kernel_size=(3,3), strides=2, padding='same')(x)
    x = PEP(filters=24, neck_filters=7)(x)
    x = EP(filters=70, stride=(2,2))(x)
    x = PEP(filters=70, neck_filters=25)(x)
    x = PEP(filters=70, neck_filters=24)(x)
    x = EP(filters=150, stride=(2, 2))(x)
    x = PEP(filters=150, neck_filters=56)(x)
    x = Conv2D(filters=150,kernel_size=(1,1),padding='same')(x)
    x = FCA(reduction_ratio=8)(x)
    x = PEP(filters=150, neck_filters=73)(x)
    x = PEP(filters=150, neck_filters=71)(x)
    x1 = PEP(filters=150, neck_filters=75)(x)
    x = EP(filters=325,stride=(2,2))(x1)
    x = PEP(filters=325,neck_filters=132)(x)
    x = PEP(filters=325,neck_filters=124)(x)
    x = PEP(filters=325,neck_filters=141)(x)
    x = PEP(filters=325,neck_filters=140)(x)
    x = PEP(filters=325,neck_filters=137)(x)
    x = PEP(filters=325,neck_filters=135)(x)
    x = PEP(filters=325, neck_filters=133)(x)
    x2 = PEP(filters=325, neck_filters=140)(x)
    x = EP(filters=545,stride=(2,2))(x2)
    x = PEP(filters=545, neck_filters=276)(x)
    x = Conv2D(filters=230, kernel_size=(1, 1), padding='same')(x)
    x = EP(filters=489)(x)
    x = PEP(filters=469,neck_filters=213)(x)
    x = Conv2D(filters=189, kernel_size=(1, 1), padding='same')(x)
    x = Conv2D(filters=105, kernel_size=(1, 1), padding='same')(x)

    model = tf.keras.Model(inputs=input_0,outputs=x)
    return model

_model = yoloNano(416)
_model.summary()