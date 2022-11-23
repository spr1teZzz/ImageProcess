import tensorflow.compat.v1 as tf
tf.disable_eager_execution()   #可以用于从TensorFlow 1.x到2.x的复杂迁移项目的程序开头

# hello = tf.constant("hello world!")
# sess = tf.Session()
# print(sess.run(hello))



# node1 = tf.constant(3.0,tf.float32,name="node1")
# node2 = tf.constant(4.0,tf.float32,name="node2")
# node3 = tf.add(node1,node2)
# # print(node3) #Tensor("Add:0", shape=(), dtype=float32)
# # print(node1) #Tensor("node1:0", shape=(), dtype=float32)
# # print(node2) #Tensor("node2:0", shape=(), dtype=float32)
# sess = tf.Session()
# print(sess.run(node1))
# print(sess.run(node2))
# print(sess.run(node3))
# #3.0 4.0 7.0
# sess.close()


# tens1 = tf.constant([[[1,2,2],[2,2,3]],
#                     [[3,5,6],[5,4,3]],
#                     [[7,0,1],[9,1,9]],
#                     [[11,12,7],[1,3,14]]],name="tens1")
# print(tens1)
#Tensor("tens1:0", shape=(4, 2, 3), dtype=int32) #4:数组第一层 2:数组第二层 3:数组第三层 


# scalar = tf.constant(100)
# vector = tf.constant([1,2,3,4,5])
# matrix = tf.constant([[1,2,3],[4,5,6]])
# cube_matrix = tf.constant([[[1],[2],[3]],[[4],[5],[6]],[[7],[8],[9]]])
# print(scalar.get_shape())
# print(vector.get_shape())
# print(matrix.get_shape())
# print(cube_matrix.get_shape())
# #() (5,) (2, 3) (3, 3, 1)


# tens1 = tf.constant([[[1,2],[2,3]],[[3,4],[5,6]]])
# sess = tf.Session()
# # print(sess.run(tens1)[1,1,0]) #res: 5
# print(sess.run(tens1)) #res :[[[1 2][2 3]][[3 4][5 6]]]
# sess.close()


a = tf.constant([1,2])
# a = tf.constant([1.0,2.0])
#TypeError: Input 'y' of 'AddV2' Op has type int32 that does not match type float32 of argument 'x'.
b = tf.constant([3,4])
res = a+b
sess = tf.Session()
print(sess.run(res))#[4 6]