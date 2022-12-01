import tensorflow.compat.v1 as tf
tf.disable_eager_execution()   #可以用于从TensorFlow 1.x到2.x的复杂迁移项目的程序开头

node1 = tf.constant(3.0,tf.float32,name="node1")
node2 = tf.constant(4.0,tf.float32,name="node2")
result = tf.add(node1,node2)

#创建一个会话，并通过python中的上下文管理器来管理这个会话
# with tf.Session() as sess:
#     print(sess.run(result))
#不需要在调用Session.close()函数来关闭会话
#当上下文退出时会话关闭和资源释放也自动完成了

# sess = tf.Session()
# with sess.as_default():
#     print(result.eval())#eval函数 #7.0

# sess = tf.Session()
# #下面两个命令有相同的功能
# print(sess.run(result))
# # print(result.eval(session = sess))
# #print(result.eval())#报错,session = sess ==>把sess注册成了缺省的会话

sess = tf.InteractiveSession()
print(result.eval())
print(result)
sess.close()
#res:7.0    Tensor("Add:0", shape=(), dtype=float32)