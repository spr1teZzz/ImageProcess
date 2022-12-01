import tensorflow.compat.v1 as tf
tf.disable_eager_execution()   #可以用于从TensorFlow 1.x到2.x的复杂迁移项目的程序开头
a = tf.constant(1.0,name='a')
b = tf.constant(2.5,name='b')
c = tf.add(a,b,name='c')
sess = tf.Session()
c_value = sess.run(c)
print(c_value)
sess.close()

node1 = tf.Variable(3.0,tf.float32,name="node1")
node2 = tf.Variable(4.0,tf.float32,name="node2")
result = tf.add(node1,node2,name="add")

sess = tf.Session()
#变量初始化
init = tf.global_variables_initializer()
sess.run(init)
print(sess.run(result))
