import tensorflow as tf

pb_file = '/Users/alexdodd/Documents/CMPE 258 Deep Learning/traffic-sign-recognition/output/trafficsignnet.model/keras_metadata.pb'

tf.
graph_def = tf.compat.v1.GraphDef()

# try:
with tf.io.gfile.GFile(pb_file, 'rb') as f:
    graph_def.ParseFromString(f.read())
# except:
#     with tf.io.gfile.FastGFile(pb_file, 'rb') as f:
#         graph_def.ParseFromString(f.read())

# Delete weights
for i in reversed(range(len(graph_def.node))):
    if graph_def.node[i].op == 'Const':
        del graph_def.node[i]

graph_def.library.Clear()

tf.compat.v1.train.write_graph(graph_def, "", 'model.pbtxt', as_text=True)