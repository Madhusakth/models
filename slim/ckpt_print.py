import tensorflow as tf

from tensorflow.python.tools import inspect_checkpoint as chkp


sess = tf.InteractiveSession()
saver = tf.train.import_meta_graph('mobilenet-v2/original-v2/mobilenet_v2_1.0_224_quant.ckpt.meta')
for tensor in tf.get_default_graph().get_operations():
    if "weights" in tensor.name:
        print(tensor.name)
#saver.restore(sess,'sample/model.ckpt-0')#tf.train.latest_checkpoint('./'))
print('loaded checkpoint')

#first_weights = tf.contrib.framework.load_variable('conv.ckpt-18000', 'first_weights:0')

#v1 = sess.graph.get_tensor_by_name('InceptionV3/Conv2d_1a_3x3/weights:0')
#print(sess.run(v1)[0,0,0,0:4])



#saver.save(sess, "model.ckpt")
