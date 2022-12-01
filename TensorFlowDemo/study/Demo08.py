#tensorflowc查看计算图结构
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()   #可以用于从TensorFlow 1.x到2.x的复杂迁移项目的程序开头

#清除default graph和不断增加的结点
tf.reset_default_graph()

#logdir改为自己机器上的合适的路径
logdir = "D:\VScodeProject\pythonProject\TensorFlowDemo\log"

#定义一个简单的计算图,实现向量加法的操作
input1 = tf.constant([1.0,2.0,3.0],name="input1")
input2 = tf.Variable(tf.random_uniform([3]),name="input2")
output = tf.add_n([input1,input2],name="add")

#生成一个写日志的writer，并将当前的Tensorflow计算图写入日志
writer = tf.summary.FileWriter(logdir,tf.get_default_graph())
writer.close()