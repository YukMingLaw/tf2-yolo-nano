import tensorflow as tf
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# 这个代码网上说需要加上, 如果模型里有dropout , bn层的话, 我测试过加不加结果都一样, 保险起见还是加上吧
tf.keras.backend.set_learning_phase(0)

# 首先是定义你的模型
# model = tf.keras.models.load_model('./model_save/debug.h5')
#
# def get_flops(model):
#     run_meta = tf.RunMetadata()
#     opts = tf.profiler.ProfileOptionBuilder.float_operation()
#
#     # We use the Keras session graph in the call to the profiler.
#     flops = tf.profiler.profile(graph=tf.keras.backend.get_session().graph,run_meta=run_meta, cmd='op', options=opts)
#
#     return flops.total_float_ops  # Prints the "flops" of the model.
#
# print(get_flops(model))
#
# def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
#     from tensorflow.python.framework.graph_util import convert_variables_to_constants
#     graph = session.graph
#     with graph.as_default():
#         freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
#         output_names = output_names or []
#         output_names += [v.op.name for v in tf.global_variables()]
#         # Graph -> GraphDef ProtoBuf
#         input_graph_def = graph.as_graph_def(add_shapes=True)
#         if clear_devices:
#             for node in input_graph_def.node:
#                 node.device = ""
#         frozen_graph = convert_variables_to_constants(session, input_graph_def,output_names, freeze_var_names)
#         return frozen_graph
#
# frozen_graph = freeze_session(tf.keras.backend.get_session(), output_names=[out.op.name for out in model.outputs])
# tf.train.write_graph(frozen_graph, "pb", "yolonano.pb", as_text=False)


#test pb
from tensorflow.python.platform import gfile
from model.base_layers import yolo_eval
sess = tf.Session()
with gfile.FastGFile("./pb/" + "yolov3.pb", 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    for i, n in enumerate(graph_def.node):
        print("Name of the node - %s" % n.name)
        print('node: ',i)
    sess.graph.as_default()
    tf.import_graph_def(graph_def, name='')  # 导入计算图

# 需要有一个初始化的过程
sess.run(tf.global_variables_initializer())
#exit()
input_x = sess.graph.get_tensor_by_name("Placeholder:0")
feature_13x13 = sess.graph.get_tensor_by_name("yolov3/yolov3_head/feature_map_1:0")
feature_26x26 = sess.graph.get_tensor_by_name("yolov3/yolov3_head/feature_map_2:0")
feature_52x52 = sess.graph.get_tensor_by_name("yolov3/yolov3_head/feature_map_3:0")

# input_x = sess.graph.get_tensor_by_name("input_1:0")
# feature_13x13 = sess.graph.get_tensor_by_name("conv2d_68/Conv2D:0")
# feature_26x26 = sess.graph.get_tensor_by_name("conv2d_79/Conv2D:0")
# feature_52x52 = sess.graph.get_tensor_by_name("conv2d_90/Conv2D:0")

import cv2
import numpy as np
import time

anchors = np.array([[6.,9.],[8.,13.],[11.,16.],[14.,22.],[17.,37.],[21.,26.],[29.,38.],[39.,62.],[79.,99.]],dtype='float32')
img = cv2.imread('./test_img/4.jpg')
img = cv2.resize(img,(416,416))
image = img.astype(np.float32)
input = image / 255.0
input = input[np.newaxis,:]

start = time.time()
for i in range(100):
    # start = time.time()
    yolo_output = sess.run([feature_13x13, feature_26x26, feature_52x52], feed_dict={input_x: input})
    # end = time.time()
    # print('pred time:', end - start)
    # boxes_, scores_, classes_ = yolo_eval(yolo_output, anchors, 1, np.array([416, 416]))
    # with tf.Session() as sess2:
    #     boxes = boxes_[0].eval()
    # for box in boxes:
    #     xmin = int(box[0])
    #     ymin = int(box[1])
    #     xmax = int(box[2])
    #     ymax = int(box[3])
    #     cv2.rectangle(img, (ymin, xmin), (ymax, xmax), (0, 255, 0))
    # cv2.imshow('pred', img)
    # cv2.waitKey(2000)
end = time.time()
print('total time:',(end-start))
