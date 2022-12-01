import tensorflow.compat.v1 as tf
tf.disable_eager_execution()   #可以用于从TensorFlow 1.x到2.x的复杂迁移项目的程序开头

value = tf.Variable(0,name="value")
sum = tf.Variable(0,name="sum")
one = tf.constant(1)
new_value = tf.add(value,one)
new_sum = tf.add(value,sum)
update_value = tf.assign(value,new_value)
update_sum = tf.assign(sum,new_sum)
#初始化变量
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for _ in range(10):
        sess.run(update_value)
        sess.run(update_sum)
        #print(sess.run(value))    
    print(sess.run(sum))  

