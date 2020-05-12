import tensorflow as tf

Input = tf.keras.layers.Input(shape=(416,416,3))
print(Input.shape)
print(Input.shape[::-1])