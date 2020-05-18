import tensorflow as tf
import numpy as np
from model.model_full import yoloNano
from dataset.YoloGenerator import YoloGenerator
import os

train_path = './coco/train.txt'
anchors = np.array([[6.,9.],[8.,13.],[11.,16.],[14.,22.],[17.,37.],[21.,26.],[29.,38.],[39.,62.],[79.,99.]],dtype='float32')

def create_callbacks():
    callbacks = []
    #add tensorboard callback
    tensorboard_callback = None
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir='./logs',
        histogram_freq=0,
        write_graph=True,
        write_grads=False,
        write_images=False,
        embeddings_freq=0,
        embeddings_layer_names=None,
        embeddings_metadata=None
    )
    callbacks.append(tensorboard_callback)
    # save the model
    # ensure directory created first; otherwise h5py will error after epoch.
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        os.path.join(
            './model_save',
            'save_model.h5'
        ),
        verbose=1,
    )
    callbacks.append(checkpoint)
    return callbacks

def main():
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    #get train set
    with open(train_path) as f:
        _line = f.readlines()
    train_set = [i.rstrip('\n') for i in _line]
    train_generator = YoloGenerator(train_list=train_set, anchors=anchors, num_classes=1, batch_size=16, input_size=416)

    #creat model
    model,debug_model = yoloNano(anchors, input_size=416, num_classes=1)

    #if you want to resume the train,open the code
    #model.load_weights('./model_save/save_model.h5')

    model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-3),loss={'yolo_loss':lambda y_true,y_pred:y_pred})

    callbacks = create_callbacks()

    # start training
    model.fit_generator(
        generator=train_generator,
        epochs=50,
        callbacks=callbacks
    )
    return 0

if __name__ == '__main__':
    main()