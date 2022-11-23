import tensorflow.compat.v1 as tf
tf.disable_eager_execution()   #可以用于从TensorFlow 1.x到2.x的复杂迁移项目的程序开头

tf.reset_default_graph()#清除default graph和不断增加的结点
a = tf.Variable(1,name="a")
b = tf.add(a,1,name="b")
c = tf.multiply(b,4,name="c")
d = tf.subtract(c,b,name="d")

logdir = 'D:\VScodeProject\pythonProject\TensorFlowDemo\study'

#生成一个写日志的writer,并将当前的Tensorflow计算图写入日志
writer = tf.summary.FileWriter(logdir,tf.get_default_graph())
writer.close()