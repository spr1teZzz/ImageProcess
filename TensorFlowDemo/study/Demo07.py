#占位符 placeholder
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()   #可以用于从TensorFlow 1.x到2.x的复杂迁移项目的程序开头

# a = tf.placeholder(tf.float32,name="a")
# b = tf.placeholder(tf.float32,name="b")
# c = tf.multiply(a,b,name="c")

# init = tf.global_variables_initializer()
# with tf.Session() as sess:
#     sess.run(init)
#     #通过feed_dict的参数传值,按字典格式
#     result = sess.run(c,feed_dict={a:8.0,b:3.5})
#     print(result)


a = tf.placeholder(tf.float32,name="a")
b = tf.placeholder(tf.float32,name="b")
c = tf.multiply(a,b,name="c")
d = tf.subtract(a,b,name="d")
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    #通过feed_dict的参数传值,按字典格式
    # result = sess.run([c,d],feed_dict={a:[8.0,2.0,3.5],b:[1.5,2.0,4.0]})
    # print(result)
    # print(result[0])
    rc,rd = sess.run([c,d],feed_dict={a:[8.0,2.0,3.5],b:[1.5,2.0,4.0]})
    print("value of c=",rc,"value of d=",rd)