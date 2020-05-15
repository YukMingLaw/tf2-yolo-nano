import tensorflow as tf
import os
from model.base_layers import PEP,EP,FCA
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# 这个代码网上说需要加上, 如果模型里有dropout , bn层的话, 我测试过加不加结果都一样, 保险起见还是加上吧
tf.keras.backend.set_learning_phase(0)

# 首先是定义你的模型
model = tf.keras.models.load_model('model_save/save_model.h5.old')

def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    from tensorflow.python.framework.graph_util import convert_variables_to_constants
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        # Graph -> GraphDef ProtoBuf
        input_graph_def = graph.as_graph_def(add_shapes=True)
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = convert_variables_to_constants(session, input_graph_def,output_names, freeze_var_names)
        return frozen_graph

frozen_graph = freeze_session(tf.keras.backend.get_session(), output_names=[out.op.name for out in model.outputs])
tf.train.write_graph(frozen_graph, "pb", "yolonano.pb", as_text=False)

# from tensorflow.python.platform import gfile
#
# sess = tf.Session()
# with gfile.FastGFile("./pb/" + "mobile_center.pb", 'rb') as f:
#     graph_def = tf.GraphDef()
#     graph_def.ParseFromString(f.read())
#     sess.graph.as_default()
#     tf.import_graph_def(graph_def, name='')  # 导入计算图
#
# # 需要有一个初始化的过程
# sess.run(tf.global_variables_initializer())
#
# input_x = sess.graph.get_tensor_by_name("input_1:0")
# print (input_x)
# out_sigmoid = sess.graph.get_tensor_by_name("conv2d/Sigmoid:0")
# print (out_sigmoid)

# import cv2
# import numpy as np

# img = cv2.imread("1.jpg")
# img_w = img.shape[1]
# img_h = img.shape[0]
# image = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
# c = np.array([image.shape[1] / 2., image.shape[0] / 2.], dtype=np.float32)
# s = max(image.shape[0], image.shape[1]) * 1.0
# trans_input = get_affine_transform(c, s, (512, 512))
# image = cv2.warpAffine(image, trans_input, (512, 512), flags=cv2.INTER_LINEAR)
# image = cv2.resize(image,(512,512))
# cv2.imshow('img',image)
# # cv2.waitKey(0)
# image = image.astype(np.float32)
# input = image / 255.0
# input = input[np.newaxis,:]
#
# img_out_softmax = sess.run(out_sigmoid, feed_dict={input_x:input})
# print(img_out_softmax.shape)
# cv2.imshow('pred',img_out_softmax[0]);
# cv2.waitKey(0)
#
# print "img_out_softmax:",img_out_softmax
# prediction_labels = np.argmax(img_out_softmax, axis=1)
# print "label:",prediction_labels
