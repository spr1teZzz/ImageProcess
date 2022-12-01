import tensorflow.compat.v1 as tf
tf.disable_eager_execution()   #可以用于从TensorFlow 1.x到2.x的复杂迁移项目的程序开头

tens1 = tf.constant([1,2,3])
#创建一个会话
sess = tf.Session()
#得到张量的取值
try:
    print(sess.run(tens1))
except:
    print("Exception!")
finally:
    #关闭会话
    sess.close()
